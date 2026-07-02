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
- --episodes (int) Number of episodes. Default is 1000.
- --max-steps (int) Number of maximum step per episode. Default is 100.
- --step-size (float) Movement length in meters. Defaults are dqn=0.2 and ddpg=1.0.
- --heatmap-interval (int) Episode interval for heatmap and policy plots. Default is 100.
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

Students should focus on the algorithmic pieces. The TODO hints point back to
Deep SARSA, DQN, and REINFORCE patterns without giving copy-paste solutions:

1. `ActorNetwork.forward`: map a state to one deterministic continuous action.
   - Look back at REINFORCE's policy network shape.
   - Difference: DDPG returns an action directly, not a distribution.
2. `CriticNetwork.forward`: evaluate `Q(s, a)` from state-action pairs.
   - Look back at Deep SARSA's Q-network.
   - Difference: the action is part of the input vector, not an index.
3. `DDPGAgent.select_action`: use the actor, add exploration noise, and keep actions valid.
   - Look back at REINFORCE for continuous actions.
   - Look back at DQN/Deep SARSA for the exploration-vs-evaluation split.
4. `DDPGAgent.learn` and `_soft_update`: implement the DDPG update.
   - The critic target is the TD target idea from Deep SARSA/DQN.
   - The critic update is the same loss/optimizer pattern as Deep SARSA/DQN.
   - The actor update maximizes the critic's value estimate for actor-chosen actions.
   - The soft update is DQN's target-network idea, but blended gradually with `tau`.

The replay buffer, warmup exploration, noise decay, save/load, logging, and
trainer loop are provided so the TODOs stay focused on the DDPG algorithm.

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
