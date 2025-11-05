import os
import yaml
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from env.mvrp_env import MVRPSTWEnv

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def make_env(env_config, seed=0):
    def _init():
        env = MVRPSTWEnv(**env_config)
        env.reset(seed=seed + int(1000 * np.random.random()))
        return env
    return _init

def main():
    # Load configuration
    config = load_config("../config/ppo_fast.yaml")
    
    # Set random seeds
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # Setup environment
    env_config = {
        'num_customers': config['env']['num_customers'][0],  # Start with smallest instance
        'num_vehicles': config['env']['num_vehicles'],
        'vehicle_capacity': config['env']['vehicle_capacity'],
        'max_time': config['env']['max_time']
    }
    
    # Create vectorized environments
    n_envs = 8  # Adjust based on your CPU cores
    env = DummyVecEnv([make_env(env_config, i) for i in range(n_envs)])
    
    # Initialize model
    model = PPO(
        config['model']['policy'],
        env,
        learning_rate=config['training']['learning_rate'],
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        n_epochs=config['training']['n_epochs'],
        gamma=config['training']['gamma'],
        ent_coef=config['training']['ent_coef'],
        vf_coef=config['training']['vf_coef'],
        max_grad_norm=config['training']['max_grad_norm'],
        clip_range=config['training']['clip_range'],
        clip_range_vf=config['training']['clip_range_vf'],
        policy_kwargs={
            'net_arch': config['model']['net_arch']
        },
        verbose=1,
        tensorboard_log=config['logging']['tensorboard_log']
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./checkpoints/",
        log_path="./logs/",
        eval_freq=config['logging']['eval_freq'],
        deterministic=True,
        render=False,
        n_eval_episodes=config['logging']['n_eval_episodes']
    )
    
    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("checkpoints/final_model")
    print("Training completed. Models saved to checkpoints/")

if __name__ == "__main__":
    main()
