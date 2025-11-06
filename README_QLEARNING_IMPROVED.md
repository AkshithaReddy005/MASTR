# Improved Q-Learning Implementation for VRP

## Overview
This document describes the improved Q-learning implementation that follows proper reinforcement learning principles and includes hyperparameter tuning.

## Key Improvements Over Previous Implementation

### 1. **Proper Reinforcement Learning Implementation**

#### Standard Q-Learning Update Rule
The improved implementation follows the standard Bellman equation for Q-learning:

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                        a'
```

Where:
- `Q(s,a)`: Current Q-value for state-action pair
- `α`: Learning rate (alpha)
- `r`: Immediate reward from environment
- `γ`: Discount factor (gamma)
- `max Q(s',a')`: Maximum Q-value for next state

#### Temporal Difference Learning
```python
td_target = reward + self.gamma * max_next_q
td_error = td_target - current_q
new_q = current_q + self.alpha * td_error
```

### 2. **Fixed Reward Double-Counting Issue**

**Previous Implementation:**
- Environment provided rewards with penalties
- Agent added additional penalties on top
- This caused double-counting of penalties

**Improved Implementation:**
- Uses environment rewards directly
- No additional penalty calculations in agent
- Follows standard RL principle: environment provides rewards, agent learns from them

### 3. **Enhanced State Representation**

**Previous State:**
```python
state = (num_unserved, capacity_bin)  # 2 features
```

**Improved State:**
```python
state = (num_unserved, capacity_bin, time_bin, vehicle_idx)  # 4 features
```

New features:
- **time_bin**: Current time discretized into 5 bins (0-20%, 20-40%, etc.)
- **vehicle_idx**: Which vehicle is currently being used
- **capacity_bin**: Increased from 4 to 10 bins for better granularity

### 4. **Learning Rate Decay**

**New Feature:**
```python
alpha = max(alpha_min, alpha * alpha_decay)
```

Benefits:
- Early training: Higher learning rate for faster exploration
- Later training: Lower learning rate for fine-tuning and stability
- Prevents oscillations in Q-values during convergence

### 5. **Improved Exploration Strategy**

**Previous:**
- 50% random action
- 50% nearest neighbor heuristic

**Improved:**
- Pure epsilon-greedy policy (standard RL approach)
- Exploration: Random action
- Exploitation: Highest Q-value action
- Falls back to nearest neighbor only when all Q-values are equal (early training)

### 6. **Hyperparameter Tuning**

The implementation includes a comprehensive hyperparameter tuning script that tests:

#### Tested Hyperparameters:
1. **Learning Rate (α)**: [0.05, 0.1, 0.2]
   - Controls how much new information overrides old information
   
2. **Discount Factor (γ)**: [0.9, 0.95, 0.99]
   - Balances immediate vs future rewards
   - Higher values = more future-oriented
   
3. **Epsilon Decay**: [0.995, 0.999]
   - Controls exploration reduction rate
   
4. **Epsilon Min**: [0.01, 0.05]
   - Minimum exploration rate
   
5. **Alpha Decay**: [0.9995, 1.0]
   - Learning rate reduction rate
   - 1.0 = no decay

## Files

### Core Implementation
- `qlearning_improved.py`: Improved Q-learning agent with proper RL implementation
- `train_qlearning_improved.py`: Training script with optimal hyperparameters
- `evaluate_qlearning_improved.py`: Evaluation script
- `hyperparameter_tuning.py`: Comprehensive hyperparameter search

### Original Implementation (for comparison)
- `qlearning_simple.py`: Original simple Q-learning
- `train_qlearning_simple.py`: Original training script
- `evaluate_qlearning_simple.py`: Original evaluation script

## Usage

### 1. Train with Improved Implementation

```bash
python train_qlearning_improved.py
```

This trains the agent with optimal hyperparameters:
- Learning Rate: 0.1 (with decay)
- Discount Factor: 0.95
- Epsilon: 1.0 → 0.01 (with decay)

### 2. Evaluate Trained Agent

```bash
python evaluate_qlearning_improved.py
```

Evaluates the best saved agent over 50 episodes.

### 3. Run Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

Tests multiple hyperparameter configurations and saves results to `hyperparameter_results/`.

## Theoretical Background

### Q-Learning Algorithm

Q-learning is a **model-free, off-policy** reinforcement learning algorithm:

- **Model-free**: Doesn't require knowledge of environment dynamics
- **Off-policy**: Learns optimal policy while following exploratory policy
- **Temporal Difference**: Learns from each step without waiting for episode end

### Key Concepts

#### 1. Value Function
Q(s,a) estimates the expected cumulative reward of taking action `a` in state `s` and following the optimal policy thereafter.

#### 2. Bellman Optimality Equation
The optimal Q-value satisfies:
```
Q*(s,a) = E[r + γ max Q*(s',a')]
                    a'
```

#### 3. Epsilon-Greedy Policy
Balances exploration vs exploitation:
- With probability ε: Choose random action (explore)
- With probability 1-ε: Choose action with highest Q-value (exploit)

#### 4. Convergence Guarantee
Q-learning converges to optimal Q* if:
- All state-action pairs are visited infinitely often
- Learning rate α decays appropriately
- Rewards are bounded

## Expected Performance Improvements

Compared to the simple implementation, the improved version should show:

1. **Better Convergence**
   - Learning rate decay prevents oscillations
   - More stable Q-values

2. **Higher Solution Quality**
   - Enhanced state representation captures more information
   - Better decision making

3. **Fewer Violations**
   - Proper reward usage
   - No penalty double-counting

4. **Faster Learning**
   - Optimal hyperparameters
   - Better exploration strategy

## Hyperparameter Tuning Results

After running `hyperparameter_tuning.py`, you'll get:

1. **JSON Results File**: `hyperparameter_results/tuning_results_YYYYMMDD_HHMMSS.json`
   - Contains all test results
   - Training curves
   - Final performance metrics

2. **Summary Report**: Printed to console
   - Ranked configurations
   - Best hyperparameters
   - Performance comparison

## Configuration Comparison

| Feature | Simple Q-Learning | Improved Q-Learning |
|---------|------------------|---------------------|
| State Features | 2 | 4 |
| Learning Rate Decay | ❌ | ✅ |
| Proper Reward Usage | ❌ | ✅ |
| Exploration Strategy | Hybrid | Standard ε-greedy |
| Hyperparameter Tuning | ❌ | ✅ |
| Capacity Bins | 4 | 10 |
| Documentation | Basic | Comprehensive |

## References

### Reinforcement Learning Theory
1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2. Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.

### Vehicle Routing Problems
1. Solomon, M. M. (1987). Algorithms for the vehicle routing and scheduling problems with time window constraints. *Operations Research*, 35(2), 254-265.

## Future Improvements

Potential enhancements:
1. **Double Q-Learning**: Reduce overestimation bias
2. **Experience Replay**: Learn from past experiences
3. **Prioritized Sweeping**: Focus on important updates
4. **Function Approximation**: Neural network instead of Q-table
5. **Multi-step Returns**: n-step Q-learning

## Contact

For questions or issues, please refer to the main project README.
