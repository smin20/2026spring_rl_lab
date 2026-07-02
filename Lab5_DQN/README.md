## Lab 5. DQN

This lab introduces off-policy deep value-based reinforcement learning with
experience replay and a target network.
Implemented algorithm:

- **DQN**
- **DDPG skeleton** for self-guided continuous-control practice

The DDPG file is intentionally incomplete. Fill the blocks marked with
`########## TODO` in `algos/ddpg.py`.

### Continuous GridWorld Environment

#### Discrete-Action Continuous GridWorld (`gridworld_c1.py`)
- **State:** continuous \((row, col)\) in meters
- **Actions:** 8 directions (N, NE, E, SE, S, SW, W, NW), fixed step length
- **Rewards:**
  - Move: **-1**
  - Trap: **-100**, episode ends
  - Goal: **+100**, episode ends

Each cell in the map can be one of:
- **Normal** (0): free to move, reward -1
- **Wall** (1): blocks movement, reward -1
- **Trap** (2): ends episode, reward -100
- **Goal** (3): ends episode, reward +100

#### Continuous-Action GridWorld (`gridworld_c2.py`)
- **State:** continuous \((row, col)\) in meters
- **Actions:** 2D vector \((\Delta row, \Delta col)\) in \([-1, 1]^2\)
- **Used by:** DDPG skeleton

### Training

To train a DQN agent, run:
```bash
python train.py --algo dqn
```

To work on the DDPG skeleton, first fill the `########## TODO` blocks in
`algos/ddpg.py`, then run:

```bash
python train.py --algo ddpg
```

**Arguments**
- --algo (str, required): Choose the learning algorithm
	- Options: dqn, ddpg
- --map (str, optional): Select GridWorld Map (Choice: map0, map1, map2, map3). Default is map1.
- --save_name (str): Filename to save the policy and the plots. Default is example.
- --render (action flag): If specified, render the agent's behavior during training.
- --logdir (str, optional): Directory to save tensorboard log files. Default is `runs`.
- --seed (int, optional): Seed for training. Default is 42.

**Parameter Arguments**
- --episodes (int) Number of episodes. Default is 500.
- --max-steps (int) Number of maximum step per episode. Default is 100.
- --step-size (float) Movement length for each discrete action. Default is 0.2.
- --heatmap-interval (int) Episode interval for heatmap and policy plots. Default is 500.
- --resolution (float) Grid resolution for heatmap and policy plots. Default is 0.1.

### TensorBoard

Run the following script to visualize TensorBoard logs.
```bash
tensorboard --logdir runs/
```

TensorBoard logs include:
- **Reward Curve**
- **Epsilon Decay**
- **Loss**
- **StateValueHeatmap** (figure visualization)
- **PolicyArrows** (figure visualization)

For DDPG, the logs use actor/critic losses and exploration noise instead of
epsilon.

### DDPG Skeleton TODOs

Students should focus on the algorithmic pieces:

1. `ActorNetwork.forward`: map state to continuous action with `tanh`.
2. `CriticNetwork.forward`: evaluate `Q(s, a)` from state-action pairs.
3. `ReplayBuffer.sample`: store continuous actions as float vectors.
4. `DDPGAgent.select_action`: add Gaussian exploration noise and clip actions.
5. `DDPGAgent.learn`: implement critic target, critic update, and actor update.
6. `DDPGAgent._soft_update`: update target actor and target critic with `tau`.

### Rendering a Trained Policy

```bash
python test.py --algo dqn
```

For a completed DDPG implementation:

```bash
python test.py --algo ddpg
```

The trained policy is saved as:

```text
checkpoints/dqn_{save_name}.pth
```
