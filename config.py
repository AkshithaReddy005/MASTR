"""
Configuration file for MASTR
Centralized hyperparameters and settings
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvironmentConfig:
    """Environment configuration"""
    num_customers: int = 20
    num_vehicles: int = 3
    vehicle_capacity: float = 100.0
    grid_size: float = 100.0
    max_time: float = 480.0  # 8 hours in minutes
    penalty_early: float = 1.0
    penalty_late: float = 2.0
    seed: Optional[int] = None


@dataclass
class ModelConfig:
    """MAAM model configuration"""
    input_dim: int = 8  # [x, y, demand, start_time, end_time, penalty_early, penalty_late, visited]
    embed_dim: int = 128
    num_heads: int = 8
    num_encoder_layers: int = 3
    ff_dim: int = 512
    dropout: float = 0.1
    tanh_clipping: float = 10.0


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimizer
    learning_rate: float = 1e-4
    baseline_lr: float = 1e-3
    
    # RL
    gamma: float = 0.99  # Discount factor
    
    # Training loop
    num_iterations: int = 1000
    episodes_per_iter: int = 32
    eval_interval: int = 50
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    
    # Logging
    log_dir: str = "runs/maam_training"
    
    # Device
    device: str = "cuda"  # or "cpu"


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    num_episodes: int = 100
    seed: int = 42
    
    # OR-Tools baseline
    ortools_time_limit: int = 30  # seconds
    compare_with_ortools: bool = True
    
    # Visualization
    visualize: bool = True
    save_plots: bool = True
    plot_dir: str = "results"


@dataclass
class DataConfig:
    """Data configuration"""
    # Kaggle dataset
    dataset_name: str = "abhilashg23/vehicle-routing-problem-ga-dataset"
    dataset_file: str = "VRP - C101.csv"
    
    # Paths
    raw_data_dir: str = "MASTR/data/raw"
    processed_data_dir: str = "MASTR/data/processed"
    
    # Processing
    add_time_windows: bool = True
    time_window_min: float = 60.0
    time_window_max: float = 180.0


# Default configurations
DEFAULT_ENV_CONFIG = EnvironmentConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_EVAL_CONFIG = EvaluationConfig()
DEFAULT_DATA_CONFIG = DataConfig()


def get_config(config_type: str = "default"):
    """
    Get configuration preset
    
    Args:
        config_type: One of ["default", "small", "large", "test"]
    
    Returns:
        Tuple of (env_config, model_config, training_config)
    """
    if config_type == "default":
        return (
            DEFAULT_ENV_CONFIG,
            DEFAULT_MODEL_CONFIG,
            DEFAULT_TRAINING_CONFIG,
            DEFAULT_EVAL_CONFIG
        )
    
    elif config_type == "small":
        # Small problem for quick testing
        env_config = EnvironmentConfig(
            num_customers=10,
            num_vehicles=2,
            vehicle_capacity=80.0
        )
        model_config = ModelConfig(
            embed_dim=64,
            num_heads=4,
            num_encoder_layers=2,
            ff_dim=256
        )
        training_config = TrainingConfig(
            num_iterations=100,
            episodes_per_iter=16,
            eval_interval=20
        )
        eval_config = EvaluationConfig(
            num_episodes=20
        )
        return env_config, model_config, training_config, eval_config
    
    elif config_type == "large":
        # Large problem for challenging scenarios
        env_config = EnvironmentConfig(
            num_customers=50,
            num_vehicles=5,
            vehicle_capacity=150.0,
            grid_size=150.0
        )
        model_config = ModelConfig(
            embed_dim=256,
            num_heads=16,
            num_encoder_layers=4,
            ff_dim=1024
        )
        training_config = TrainingConfig(
            num_iterations=2000,
            episodes_per_iter=64,
            eval_interval=100,
            learning_rate=5e-5
        )
        eval_config = EvaluationConfig(
            num_episodes=200,
            ortools_time_limit=60
        )
        return env_config, model_config, training_config, eval_config
    
    elif config_type == "test":
        # Minimal config for unit testing
        env_config = EnvironmentConfig(
            num_customers=5,
            num_vehicles=2,
            vehicle_capacity=50.0,
            grid_size=50.0
        )
        model_config = ModelConfig(
            embed_dim=32,
            num_heads=2,
            num_encoder_layers=1,
            ff_dim=128
        )
        training_config = TrainingConfig(
            num_iterations=10,
            episodes_per_iter=4,
            eval_interval=5
        )
        eval_config = EvaluationConfig(
            num_episodes=5
        )
        return env_config, model_config, training_config, eval_config
    
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def print_config(env_config, model_config, training_config, eval_config=None):
    """Print configuration summary"""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    print("\n[Environment]")
    print(f"  Customers:        {env_config.num_customers}")
    print(f"  Vehicles:         {env_config.num_vehicles}")
    print(f"  Vehicle Capacity: {env_config.vehicle_capacity}")
    print(f"  Grid Size:        {env_config.grid_size}")
    print(f"  Max Time:         {env_config.max_time} min")
    
    print("\n[Model]")
    print(f"  Embedding Dim:    {model_config.embed_dim}")
    print(f"  Num Heads:        {model_config.num_heads}")
    print(f"  Encoder Layers:   {model_config.num_encoder_layers}")
    print(f"  FF Dim:           {model_config.ff_dim}")
    print(f"  Dropout:          {model_config.dropout}")
    
    print("\n[Training]")
    print(f"  Iterations:       {training_config.num_iterations}")
    print(f"  Episodes/Iter:    {training_config.episodes_per_iter}")
    print(f"  Learning Rate:    {training_config.learning_rate}")
    print(f"  Gamma:            {training_config.gamma}")
    print(f"  Device:           {training_config.device}")
    
    if eval_config:
        print("\n[Evaluation]")
        print(f"  Num Episodes:     {eval_config.num_episodes}")
        print(f"  Compare OR-Tools: {eval_config.compare_with_ortools}")
        print(f"  Visualize:        {eval_config.visualize}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test configurations
    print("Testing configuration presets...\n")
    
    for config_type in ["default", "small", "large", "test"]:
        print(f"\n{'#'*60}")
        print(f"Config Type: {config_type.upper()}")
        print(f"{'#'*60}")
        
        env_cfg, model_cfg, train_cfg, eval_cfg = get_config(config_type)
        print_config(env_cfg, model_cfg, train_cfg, eval_cfg)
