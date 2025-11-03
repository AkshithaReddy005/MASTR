# Getting Started with MASTR

This guide will help you get up and running with MASTR quickly.

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, but recommended for faster training)
- 8GB+ RAM
- Kaggle API credentials (for dataset download)

## Installation

### 1. Install Dependencies

```bash
cd MASTR/MASTR
pip install -r requirements.txt
```

### 2. Set Up Kaggle API (Optional)

If you want to use the Kaggle dataset:

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account Settings → API → Create New API Token
3. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

```bash
# Linux/Mac
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\
```

## Quick Start (5 minutes)

Run the quick start script to see MASTR in action:

```bash
python MASTR/scripts/quick_start.py
```

This will:
- Create a small MVRPSTW environment
- Initialize the MAAM model
- Train for 100 iterations (demo)
- Evaluate and display results

## Full Training (30-60 minutes)

For better results, train the model properly:

```bash
python MASTR/train/train_rl.py
```

**Training parameters:**
- Iterations: 1000
- Episodes per iteration: 32
- Total training time: ~30-60 minutes (GPU) or 2-3 hours (CPU)

**Monitor training progress:**

```bash
tensorboard --logdir runs/maam_training
```

Open http://localhost:6006 in your browser.

## Evaluation

### Basic Evaluation

```bash
python MASTR/scripts/evaluate.py --model-path checkpoints/best_model.pt
```

### Compare with OR-Tools Baseline

```bash
python MASTR/scripts/evaluate.py \
    --model-path checkpoints/best_model.pt \
    --compare-ortools \
    --visualize \
    --num-episodes 100
```

This will:
- Evaluate MAAM on 100 test episodes
- Compare with OR-Tools baseline
- Generate route visualizations
- Save results to `results/`

## Interactive Exploration

Launch the Jupyter notebook for interactive analysis:

```bash
jupyter notebook MASTR/notebooks/demo.ipynb
```

The notebook includes:
- Environment setup and visualization
- Model training walkthrough
- Solution comparison
- Performance analysis

## Usage Examples

### Example 1: Train on Custom Problem Size

```python
from env.mvrp_env import MVRPSTWEnv
from model.maam_model import MAAM
from train.train_rl import REINFORCETrainer

# Create larger environment
env = MVRPSTWEnv(
    num_customers=50,
    num_vehicles=5,
    vehicle_capacity=150.0
)

# Create model
model = MAAM(embed_dim=256, num_heads=16, num_encoder_layers=4)

# Train
trainer = REINFORCETrainer(model, env)
trainer.train(num_iterations=2000)
```

### Example 2: Use Configuration Presets

```python
from config import get_config, print_config
from env.mvrp_env import MVRPSTWEnv
from model.maam_model import MAAM
from train.train_rl import REINFORCETrainer

# Get large problem configuration
env_cfg, model_cfg, train_cfg, eval_cfg = get_config("large")
print_config(env_cfg, model_cfg, train_cfg, eval_cfg)

# Create environment and model from config
env = MVRPSTWEnv(**env_cfg.__dict__)
model = MAAM(**model_cfg.__dict__)
trainer = REINFORCETrainer(model, env, **train_cfg.__dict__)

# Train
trainer.train(
    num_iterations=train_cfg.num_iterations,
    episodes_per_iter=train_cfg.episodes_per_iter,
    eval_interval=train_cfg.eval_interval
)
```

### Example 3: Solve a Single Instance

```python
import torch
from env.mvrp_env import MVRPSTWEnv
from model.maam_model import MAAM
from train.train_rl import REINFORCETrainer
from utils.metrics import evaluate_solution, print_solution_summary

# Load trained model
env = MVRPSTWEnv(num_customers=20, num_vehicles=3)
model = MAAM(input_dim=8, embed_dim=128)
trainer = REINFORCETrainer(model, env)
trainer.load_checkpoint('checkpoints/best_model.pt')

# Solve instance
env.reset(seed=42)
done = False
obs = env._get_obs()

while not done:
    customer_features, vehicle_state = trainer._parse_observation(obs)
    mask = trainer._get_mask(obs)
    
    with torch.no_grad():
        action, _ = model.sample_action(
            customer_features, vehicle_state, mask, greedy=True
        )
    
    obs, reward, done, _, _ = env.step(action.item())

# Get solution
routes = env.get_routes()
cost = env.get_solution_cost()

# Evaluate
metrics = evaluate_solution(
    routes, env.customer_locs, env.depot, env.demands,
    env.start_times, env.end_times,
    env.penalties_early, env.penalties_late,
    env.vehicle_capacity
)

print_solution_summary(metrics)
```

## Troubleshooting

### Issue: CUDA out of memory

**Solution:** Reduce batch size or model size:
```python
# Smaller model
model = MAAM(embed_dim=64, num_heads=4, num_encoder_layers=2)

# Smaller batch
trainer.train(episodes_per_iter=16)  # Instead of 32
```

### Issue: Training is slow

**Solutions:**
1. Use GPU: Ensure PyTorch is installed with CUDA support
2. Reduce problem size: Start with fewer customers
3. Use fewer episodes per iteration

### Issue: Model not converging

**Solutions:**
1. Train longer (1000+ iterations)
2. Adjust learning rate
3. Increase episodes per iteration for better gradient estimates
4. Check reward scaling

### Issue: Kaggle dataset download fails

**Solution:** Use synthetic data instead:
```python
# The environment generates synthetic instances automatically
env = MVRPSTWEnv(num_customers=20, num_vehicles=3)
# No need to load external data
```

## Next Steps

1. **Experiment with hyperparameters**: Try different model sizes, learning rates
2. **Scale up**: Test on larger problem instances (50-100 customers)
3. **Customize rewards**: Modify reward function in `env/mvrp_env.py`
4. **Add features**: Extend customer features (e.g., priority, service time)
5. **Deploy**: Use trained model for real-world routing

## Resources

- **Documentation**: See `README.md` for detailed project overview
- **Code**: All modules are documented with docstrings
- **Examples**: Check `notebooks/demo.ipynb` for interactive examples
- **Issues**: Report bugs at GitHub Issues

## Support

For questions or issues:
- Check the FAQ in README.md
- Review code documentation
- Open an issue on GitHub
- Contact: your.email@example.com

---

**Happy Routing! 🚚📦**
