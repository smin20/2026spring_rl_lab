"""
학습 과정 시각화 스크립트
- 각 알고리즘(SARSA, Q-Learning) × 각 맵(HW1_1, HW1_2)에 대해
- 에피소드별 보상 및 성공률 그래프를 생성하여 PNG로 저장
"""
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np

# Minimal environment without pygame rendering
import json

class TileType:
    NORMAL = 0
    WALL = 1
    TRAP = 2
    GOAL = 3

class Action:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
ALL_ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

class SimpleGridWorld:
    def __init__(self, map_file):
        with open(f'env/maps/{map_file}', 'r') as f:
            data = json.load(f)
        self.width = data['width']
        self.height = data['height']
        self.grid = [[TileType.NORMAL]*self.width for _ in range(self.height)]
        for y, x in data.get('walls', []):
            self.grid[y][x] = TileType.WALL
        for y, x in data.get('traps', []):
            self.grid[y][x] = TileType.TRAP
        gy, gx = data['goal']
        self.grid[gy][gx] = TileType.GOAL
        self.agent_pos = [0, 0]
        self.done = False

    def reset(self):
        self.agent_pos = [0, 0]
        self.done = False
        return list(self.agent_pos)

    def step(self, action):
        if self.done:
            return list(self.agent_pos), 0, True
        moves = {Action.UP: (-1,0), Action.DOWN: (1,0), Action.LEFT: (0,-1), Action.RIGHT: (0,1)}
        dy, dx = moves[action]
        ny, nx = self.agent_pos[0]+dy, self.agent_pos[1]+dx
        if 0 <= ny < self.height and 0 <= nx < self.width:
            if self.grid[ny][nx] != TileType.WALL:
                self.agent_pos = [ny, nx]
        tile = self.grid[self.agent_pos[0]][self.agent_pos[1]]
        reward = -1
        if tile == TileType.TRAP:
            reward = -100
            self.done = True
        elif tile == TileType.GOAL:
            reward = 100
            self.done = True
        return list(self.agent_pos), reward, self.done


def train_sarsa(env, episodes=500, alpha=0.3, gamma=0.99, min_epsilon=0.01, decay_rate=0.98):
    Q = defaultdict(lambda: {a: 0.0 for a in ALL_ACTIONS})
    max_steps = 500
    initial_epsilon = 1.0

    episode_rewards = []
    episode_successes = []

    for ep in range(episodes):
        epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** ep))
        state = tuple(env.reset())
        if random.random() < epsilon:
            action = random.choice(ALL_ACTIONS)
        else:
            action = max(Q[state], key=Q[state].get)

        done = False
        total_reward = 0
        steps = 0
        while not done and steps < max_steps:
            next_state, reward, done = env.step(action)
            next_state = tuple(next_state)
            total_reward += reward
            if random.random() < epsilon:
                next_action = random.choice(ALL_ACTIONS)
            else:
                next_action = max(Q[next_state], key=Q[next_state].get)
            td_target = reward + gamma * Q[next_state][next_action]
            Q[state][action] += alpha * (td_target - Q[state][action])
            state, action = next_state, next_action
            steps += 1

        episode_rewards.append(total_reward)
        episode_successes.append(1 if reward == 100 else 0)

    return episode_rewards, episode_successes


def train_qlearning(env, episodes=500, alpha=0.3, gamma=0.99, min_epsilon=0.01, decay_rate=0.98):
    Q = defaultdict(lambda: {a: 0.0 for a in ALL_ACTIONS})
    max_steps = 100
    initial_epsilon = 1.0

    episode_rewards = []
    episode_successes = []

    for ep in range(episodes):
        epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** ep))
        state = tuple(env.reset())
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < max_steps:
            if random.random() < epsilon:
                action = random.choice(ALL_ACTIONS)
            else:
                action = max(Q[state], key=Q[state].get)
            next_state, reward, done = env.step(action)
            next_state = tuple(next_state)
            if done and reward == 100:
                Q[state][action] += alpha * (reward - Q[state][action])
            else:
                max_next = max(Q[next_state].values())
                Q[state][action] += alpha * (reward + gamma * max_next - Q[state][action])
            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_successes.append(1 if reward == 100 else 0)

    return episode_rewards, episode_successes


def moving_average(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode='valid')


def main():
    save_dir = 'figures'
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 튜닝된 파라미터로 개별 그래프 그리기
    configs = [
        ('SARSA', 'HW1_1.json', lambda env: train_sarsa(env, alpha=0.3, min_epsilon=0.01, decay_rate=0.98)),
        ('SARSA', 'HW1_2.json', lambda env: train_sarsa(env, alpha=0.3, min_epsilon=0.01, decay_rate=0.98)),
        ('Q-Learning', 'HW1_1.json', lambda env: train_qlearning(env, alpha=0.3, min_epsilon=0.01, decay_rate=0.98)),
        ('Q-Learning', 'HW1_2.json', lambda env: train_qlearning(env, alpha=0.3, min_epsilon=0.01, decay_rate=0.98)),
    ]
    
    # 임시 그래프 그리기용 함수
    def plot_training_curves(rewards, successes, algo_name, map_name, save_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        raw_color = '#B0BEC5'
        smooth_color = '#1976D2' if 'SARSA' in algo_name else '#D32F2F'
        success_color = '#43A047' if 'SARSA' in algo_name else '#F57C00'
        episodes = range(1, len(rewards)+1)
        
        ax1.plot(episodes, rewards, alpha=0.25, color=raw_color, linewidth=0.8, label='Raw')
        smoothed = moving_average(rewards, 20)
        ax1.plot(range(20, len(rewards)+1), smoothed, color=smooth_color, linewidth=2.2, label='Moving Avg (20ep)')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title(f'{algo_name} — Episode Reward ({map_name})', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        window = 50
        success_rate = moving_average(successes, window) * 100
        ax2.plot(range(window, len(successes)+1), success_rate, color=success_color, linewidth=2.2)
        ax2.fill_between(range(window, len(successes)+1), success_rate, alpha=0.15, color=success_color)
        ax2.axhline(y=80, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='80% threshold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title(f'{algo_name} — Success Rate ({map_name})', fontweight='bold')
        ax2.set_ylim(-5, 105)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f'{algo_name.lower().replace("-", "_")}_{map_name.replace(".json","")}.png'
        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()

    for algo_name, map_name, train_fn in configs:
        env = SimpleGridWorld(map_name)
        rewards, successes = train_fn(env)
        plot_training_curves(rewards, successes, algo_name, map_name, save_dir)
        print(f"Generated tuned plot for {algo_name} ({map_name})")

    # 2. [기본값 vs 튜닝값] 성능 비교 시각화 (HW1_1 기준)
    print("\nRunning comparison experiments...")
    env = SimpleGridWorld('HW1_1.json')
    
    # SARSA 학습 실행
    sarsa_def_rewards, sarsa_def_succ = train_sarsa(env, alpha=0.1, min_epsilon=0.05, decay_rate=0.99)
    env.reset()
    sarsa_tune_rewards, sarsa_tune_succ = train_sarsa(env, alpha=0.3, min_epsilon=0.01, decay_rate=0.98)
    
    # Q-Learning 학습 실행
    env.reset()
    ql_def_rewards, ql_def_succ = train_qlearning(env, alpha=0.1, min_epsilon=0.05, decay_rate=0.99)
    env.reset()
    ql_tune_rewards, ql_tune_succ = train_qlearning(env, alpha=0.3, min_epsilon=0.01, decay_rate=0.98)

    # 그래프 그리기
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    w = 50  # 성공률 moving average 윈도우 크기
    
    # (1) SARSA Reward 비교
    axes[0][0].plot(range(20, 501), moving_average(sarsa_def_rewards, 20), color='#78909C', linestyle='--', linewidth=2, label='Default (alpha=0.1, decay=0.99)')
    axes[0][0].plot(range(20, 501), moving_average(sarsa_tune_rewards, 20), color='#1976D2', linewidth=2.5, label='Tuned (alpha=0.3, decay=0.98)')
    axes[0][0].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[0][0].set_title('SARSA — Reward Comparison (HW1_1)', fontsize=12, fontweight='bold')
    axes[0][0].set_xlabel('Episode')
    axes[0][0].set_ylabel('Avg Reward (Moving Avg 20)')
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)

    # (2) SARSA Success Rate 비교
    axes[1][0].plot(range(w, 501), moving_average(sarsa_def_succ, w)*100, color='#90A4AE', linestyle='--', linewidth=2, label='Default')
    axes[1][0].plot(range(w, 501), moving_average(sarsa_tune_succ, w)*100, color='#43A047', linewidth=2.5, label='Tuned')
    axes[1][0].axhline(y=80, color='red', linestyle=':', label='Goal (80%)')
    axes[1][0].set_title('SARSA — Success Rate Comparison (HW1_1)', fontsize=12, fontweight='bold')
    axes[1][0].set_xlabel('Episode')
    axes[1][0].set_ylabel('Success Rate (%)')
    axes[1][0].set_ylim(-5, 105)
    axes[1][0].legend()
    axes[1][0].grid(True, alpha=0.3)

    # (3) Q-Learning Reward 비교
    axes[0][1].plot(range(20, 501), moving_average(ql_def_rewards, 20), color='#78909C', linestyle='--', linewidth=2, label='Default (alpha=0.1, decay=0.99)')
    axes[0][1].plot(range(20, 501), moving_average(ql_tune_rewards, 20), color='#D32F2F', linewidth=2.5, label='Tuned (alpha=0.3, decay=0.98)')
    axes[0][1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    axes[0][1].set_title('Q-Learning — Reward Comparison (HW1_1)', fontsize=12, fontweight='bold')
    axes[0][1].set_xlabel('Episode')
    axes[0][1].set_ylabel('Avg Reward (Moving Avg 20)')
    axes[0][1].legend()
    axes[0][1].grid(True, alpha=0.3)

    # (4) Q-Learning Success Rate 비교
    axes[1][1].plot(range(w, 501), moving_average(ql_def_succ, w)*100, color='#90A4AE', linestyle='--', linewidth=2, label='Default')
    axes[1][1].plot(range(w, 501), moving_average(ql_tune_succ, w)*100, color='#F57C00', linewidth=2.5, label='Tuned')
    axes[1][1].axhline(y=80, color='red', linestyle=':', label='Goal (80%)')
    axes[1][1].set_title('Q-Learning — Success Rate Comparison (HW1_1)', fontsize=12, fontweight='bold')
    axes[1][1].set_xlabel('Episode')
    axes[1][1].set_ylabel('Success Rate (%)')
    axes[1][1].set_ylim(-5, 105)
    axes[1][1].legend()
    axes[1][1].grid(True, alpha=0.3)

    plt.suptitle('Hyperparameter Tuning Effect Comparison (HW1_1)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    cmp_path = os.path.join(save_dir, 'tuning_comparison.png')
    plt.savefig(cmp_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved tuning comparison plot to: {cmp_path}")
    print("Done!")

if __name__ == '__main__':
    main()
