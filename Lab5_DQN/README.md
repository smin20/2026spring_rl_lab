## Lab 5. DQN

This lab introduces off-policy deep value-based reinforcement learning with
experience replay and a target network.
Implemented algorithm:

- **DQN**

DDPG can be added to this lab later as the continuous-control extension.

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

### Training

To train a DQN agent, run:
```bash
python train.py --algo dqn
```

**Arguments**
- --algo (str, required): Choose the learning algorithm
	- Options: dqn
- --map (str, optional): Select GridWorld Map (Choice: map0, map1, map2, map3). Default is map1.
- --save_name (str): Filename to save the policy and the plots. Default is example.
- --render (action flag): If specified, render the agent's behavior during training.
- --logdir (str, optional): Directory to save tensorboard log files. Default is `runs`.
- --seed (int, optional): Seed for training. Default is 42.

**Parameter Arguments**
- --episodes (int) Number of episodes. Default is 1500.
- --max-steps (int) Number of maximum step per episode. Default is 100.
- --step-size (float) Movement length for each discrete action. Default is 1.0.

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

### Rendering a Trained Policy

```bash
python test.py --algo dqn
```

The trained policy is saved as:

```text
checkpoints/dqn_{save_name}.pth
```
