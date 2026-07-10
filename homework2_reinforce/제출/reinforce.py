import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    """
    행동(action)의 평균(mean)과 표준편차(std)를 출력합니다.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        # 초기 탐색 폭을 충분히 확보하기 위해 log_std를 0.0으로 시작
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean_layer(x)
        # 중요: std가 너무 작아지면(결정론적 붕괴) 로컬 미니멈(벽에 박혀 대기)에 갇히게 됩니다.
        # 하한선을 -1.2(std 약 0.3)로 제한하여 항상 최소한의 탐색을 보장합니다.
        log_std = torch.clamp(self.log_std, min=-2.0, max=0.5)
        std = torch.exp(log_std)
        return mean, std

class CriticNetwork(nn.Module):
    """
    상태 가치 V(s)를 출력합니다.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)

class REINFORCEAgent:
    """
    배치(Batch) 업데이트 기반의 PPO 에이전트:
    단일 에피소드 단위 업데이트의 무작위성(랜덤 시드 취약성)을 극복하기 위해,
    여러 에피소드의 데이터를 모아서 한 번에 업데이트(Batch Update)를 수행합니다.
    이를 통해 재현성을 극도로 높이고 로컬 미니멈(트랩 회피를 위한 벽 박힘)을 탈출합니다.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 5e-4,          # 배치 업데이트를 하므로 안정적인 5e-4 사용
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 5,
        update_every: int = 8,     # 8개 에피소드마다 한번씩 모아서 업데이트 수행
        device: torch.device = torch.device('cpu')
    ):
        self.device = device
        
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr * 2.5}
        ])
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.update_every = update_every
        self.episode_count = 0

        # 단일 에피소드 버퍼
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []

        # 배치 업데이트용 누적 버퍼
        self.batch_states: list[torch.Tensor] = []
        self.batch_actions: list[torch.Tensor] = []
        self.batch_log_probs: list[torch.Tensor] = []
        self.batch_returns: list[torch.Tensor] = []

        self.last_loss = 0.0

    def reset_episode(self) -> None:
        """새 에피소드 시작 시 단일 에피소드 버퍼 비우기"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()

    def select_action(self, state: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        state_unsqueeze = state.to(self.device).unsqueeze(0)
        mean, std = self.actor(state_unsqueeze)
        
        if eval_mode:
            action = torch.clamp(mean, -1.0, 1.0)
            return action.squeeze(0).cpu()

        dist = Normal(mean, std)
        action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        self.states.append(state.to(self.device))
        self.actions.append(action.squeeze(0))
        self.log_probs.append(log_prob.squeeze(0))
        
        action_clamped = torch.clamp(action, -1.0, 1.0)
        return action_clamped.squeeze(0).cpu()

    def finish_episode(self) -> float:
        if not self.rewards:
            return self.last_loss

        # 1) 현재 에피소드의 할인 누적 보상(G_t) 계산
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 에피소드 보상 정규화
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-9)

        # 배치 버퍼에 현재 에피소드 데이터 추가
        self.batch_states.extend(self.states)
        self.batch_actions.extend(self.actions)
        self.batch_log_probs.extend(self.log_probs)
        self.batch_returns.append(returns)

        # 단일 에피소드 버퍼 리셋
        self.reset_episode()
        self.episode_count += 1

        # 지정된 에피소드 수만큼 모이지 않았다면 업데이트 건너뜀
        if self.episode_count % self.update_every != 0:
            return self.last_loss

        # 2) 배치 데이터를 이용한 PPO 업데이트 수행
        states = torch.stack(self.batch_states)
        actions = torch.stack(self.batch_actions)
        old_log_probs = torch.stack(self.batch_log_probs).detach()
        returns = torch.cat(self.batch_returns)

        total_loss = 0.0

        for _ in range(self.k_epochs):
            mean, std = self.actor(states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            
            values = self.critic(states)
            
            # Advantage 계산 및 정규화
            advantages = returns - values.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratios = torch.exp(log_probs - old_log_probs)

            # Surrogate Loss 계산
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values, returns)
            # 탐색을 장려하기 위해 엔트로피 가중치를 0.02로 약간 올림
            entropy_bonus = 0.02 * dist_entropy.mean()
            
            loss = actor_loss + 0.5 * critic_loss - entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        # 누적 배치 버퍼 비우기
        self.batch_states.clear()
        self.batch_actions.clear()
        self.batch_log_probs.clear()
        self.batch_returns.clear()

        self.last_loss = total_loss / self.k_epochs
        return self.last_loss

    def save(self, path: str) -> None:
        """정책 및 가치 네트워크 파라미터를 파일에 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """저장된 파라미터를 불러와 네트워크에 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        if 'actor' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.actor.load_state_dict(checkpoint)

    def inference(self, state: torch.Tensor) -> torch.Tensor:
        """
        평가 모드에서 행동(action)을 반환합니다.
        내부적으로 select_action(eval_mode=True) 호출
        """
        return self.select_action(state, eval_mode=True)
    
    @property
    def policy(self):
        """기존 코드(train_r.py 의 시각화 부분 등)와의 호환성을 위한 프로퍼티"""
        return self.actor
