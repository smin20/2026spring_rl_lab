import os
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from algos.utils import log_map, plot_continuous_policy, plot_value_heatmap
from env.gridworld_c2 import GridWorldEnv_c2


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ActorNetwork(nn.Module):
    """
    Deterministic policy network for DDPG.

    Input: normalized state, shape (batch, state_dim)
    Output: continuous action, shape (batch, action_dim), each value in [-1, 1]
    """

    def __init__(self, state_dim=2, action_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        ########## TODO 1: Actor forward pass ##########
        # Hint:
        # - This is like REINFORCE's policy forward pass, but deterministic.
        # - The actor should return one action vector directly, not mean/std.
        # - The continuous environment expects each action component in [-1, 1].
        raise NotImplementedError("TODO 1: implement ActorNetwork.forward")
        ########## END TODO 1 ##########


class CriticNetwork(nn.Module):
    """
    Action-value network for DDPG.

    Input: normalized state and continuous action
    Output: scalar Q(s, a), shape (batch, 1)
    """

    def __init__(self, state_dim=2, action_dim=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        ########## TODO 2: Critic forward pass ##########
        # Hint:
        # - Deep SARSA estimates Q from a state and a discrete action index.
        # - DDPG estimates Q(s, a) from a state vector and a continuous action vector.
        # - Build one feature tensor from state and action, then pass it to self.net.
        raise NotImplementedError("TODO 2: implement CriticNetwork.forward")
        ########## END TODO 2 ##########


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*batch)

        states_np = np.stack(states).astype(np.float32)
        next_states_np = np.stack(next_states).astype(np.float32)
        rewards_np = np.array(rewards, dtype=np.float32)[:, None]
        dones_np = np.array(dones, dtype=np.float32)[:, None]

        actions_np = np.asarray(actions, dtype=np.float32)

        return (
            torch.from_numpy(states_np),
            torch.from_numpy(actions_np),
            torch.from_numpy(rewards_np),
            torch.from_numpy(next_states_np),
            torch.from_numpy(dones_np),
        )

    def __len__(self):
        return len(self.buffer)


def compute_critic_value_grid(env, agent, resolution):
    height, width = env.height, env.width
    xs = np.arange(resolution / 2, width, resolution)
    ys = np.arange(resolution / 2, height, resolution)
    states = np.stack([[y, x] for y in ys for x in xs], axis=0).astype(np.float32)
    states_norm = states / agent.state_scale

    agent.actor.eval()
    agent.critic.eval()
    with torch.no_grad():
        state_tensor = torch.tensor(states_norm, dtype=torch.float32, device=agent.device)
        actions = agent.actor(state_tensor)
        q_values = agent.critic(state_tensor, actions).cpu().numpy()
    agent.actor.train()
    agent.critic.train()

    values = q_values.reshape(len(ys), len(xs))
    return values, xs, ys


class DDPGAgent:
    """
    DDPG skeleton for the continuous-action GridWorld.

    Students fill the TODO blocks that define the actor, critic, target, noise,
    and soft-update mechanics.
    """

    def __init__(
        self,
        env: GridWorldEnv_c2,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        noise_std: float = 1.0,
        noise_decay: float = 0.9995,
        noise_min: float = 0.05,
        warmup_steps: int = 2000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        device: torch.device = None,
    ):
        self.env = env
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.noise_min = noise_min
        self.warmup_steps = warmup_steps
        self.total_steps = 0
        self.batch_size = batch_size
        self.action_dim = 2

        self.state_scale = np.array([env.height, env.width], dtype=np.float32)
        self.reward_scale = 1.0

        self.actor = ActorNetwork().to(self.device)
        self.critic = CriticNetwork().to(self.device)
        self.target_actor = ActorNetwork().to(self.device)
        self.target_critic = CriticNetwork().to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state, eval=False):
        if not eval:
            self.total_steps += 1
            if self.total_steps <= self.warmup_steps:
                return np.random.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

        state_norm = np.array(state, dtype=np.float32) / self.state_scale
        state_v = torch.tensor(state_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

        action = None
        ########## TODO 3: Select a continuous action ##########
        # Hint:
        # - Use the actor without tracking gradients, just as inference uses no gradients.
        # - In training mode, add zero-mean Gaussian exploration noise.
        # - Return a NumPy action clipped to the environment action range.
        ########## END TODO 3 ##########
        if action is None:
            raise NotImplementedError("TODO 3: implement DDPG action selection")
        return action

    def learn(self, state, action, reward, next_state, done=False):
        state_norm = np.array(state, dtype=np.float32) / self.state_scale
        next_state_norm = np.array(next_state, dtype=np.float32) / self.state_scale
        action_np = np.asarray(action, dtype=np.float32)
        reward_scaled = reward / self.reward_scale

        self.memory.push(state_norm, action_np, reward_scaled, next_state_norm, float(done))
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        target_q = None
        ########## TODO 4A: Build the critic target ##########
        # Hint:
        # - This is the off-policy TD target, similar in spirit to Deep SARSA/DQN.
        # - Use target networks for the next-state action and next-state value.
        # - Terminal transitions should not bootstrap from the next state.
        ########## END TODO 4A ##########
        if target_q is None:
            raise NotImplementedError("TODO 4A: compute the critic target")

        critic_loss = None
        ########## TODO 4B: Update the critic ##########
        # Hint:
        # - Compare the critic's current Q estimate with target_q.
        # - Use the same optimizer pattern as Deep SARSA and DQN:
        #   zero gradients, backpropagate loss, then step the optimizer.
        ########## END TODO 4B ##########
        if critic_loss is None:
            raise NotImplementedError("TODO 4B: update the critic")

        actor_loss = None
        ########## TODO 4C: Update the actor ##########
        # Hint:
        # - The actor should choose actions that the critic scores highly.
        # - Optimizers minimize losses, so convert "maximize Q" into a loss.
        # - Use the current actor and current critic, not the target networks.
        ########## END TODO 4C ##########
        if actor_loss is None:
            raise NotImplementedError("TODO 4C: update the actor")

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

        if self.total_steps > self.warmup_steps:
            self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, target_net, source_net):
        ########## TODO 4D: Soft-update a target network ##########
        # Hint:
        # - DQN copied the full online network into the target network sometimes.
        # - DDPG moves the target network a small fraction toward the online network.
        # - Math form: target <- (1 - tau) * target + tau * source.
        raise NotImplementedError("TODO 4D: implement soft target-network update")
        ########## END TODO 4D ##########

    def reset_episode(self):
        pass

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "target_actor_state_dict": self.target_actor.state_dict(),
                "target_critic_state_dict": self.target_critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.target_actor.load_state_dict(
            checkpoint.get("target_actor_state_dict", checkpoint["actor_state_dict"])
        )
        self.target_critic.load_state_dict(
            checkpoint.get("target_critic_state_dict", checkpoint["critic_state_dict"])
        )

    def inference(self, state):
        return self.select_action(state, eval=True)


class DDPGTrainer:
    def __init__(self, args):
        self.args = args
        self.set_seeds()

    def set_seeds(self):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

    def initialize(self):
        os.makedirs(self.args.logdir, exist_ok=True)
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.args.logdir, self.args.algo, self.args.save_name)
        )
        config_path = os.path.join("configs", f"{self.args.map}.yaml")
        self.env = GridWorldEnv_c2(config_path, step_size_m=self.args.step_size)
        self.agent = DDPGAgent(self.env)

    def train(self):
        map_img, height, width = log_map(self.writer, self.env)

        for ep in range(1, self.args.episodes + 1):
            state = self.env.reset()
            total_reward = 0.0

            for t in range(self.args.max_steps):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                terminal = done or (t == self.args.max_steps - 1)
                losses = self.agent.learn(state, action, reward, next_state, done=terminal)
                if losses is not None:
                    self.writer.add_scalar("Loss/Critic", losses["critic_loss"], ep)
                    self.writer.add_scalar("Loss/Actor", losses["actor_loss"], ep)

                state = next_state

                if self.args.render:
                    self.env.render(tick=5000)
                if done:
                    break

            if ep % self.args.heatmap_interval == 0:
                values, xs, ys = compute_critic_value_grid(
                    self.env, self.agent, self.args.resolution
                )
                plot_value_heatmap(self.writer, values, xs, ys, ep)
                plot_continuous_policy(
                    self.writer,
                    self.env,
                    self.agent,
                    map_img,
                    height,
                    width,
                    self.args.resolution,
                    ep,
                )

            self.writer.add_scalar("Reward", total_reward, ep)
            self.writer.add_scalar("ExplorationNoiseStd", self.agent.noise_std, ep)

            if ep % 100 == 0:
                print(
                    f"[DDPG] Episode: {ep}, Reward: {total_reward:.2f}, "
                    f"Noise std: {self.agent.noise_std:.3f}"
                )

    def save(self):
        os.makedirs("checkpoints", exist_ok=True)
        self.agent.save(f"checkpoints/{self.args.algo}_{self.args.save_name}.pth")
        self.writer.close()
