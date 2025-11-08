"""
Configuration management utilities.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class DataConfig:
    """Data configuration."""
    metadata_path: str
    audio_dir: str
    feature_dir: str
    feature_type: str = "mfcc"  # 'mfcc' or 'hubert'
    hubert_layer: Optional[int] = None
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration."""
    architecture: str = "mlp"  # 'mlp', 'cnn', 'bilstm', 'transformer'
    input_dim: int = 200
    num_classes: int = 10
    hidden_dims: List[int] = None
    dropout: float = 0.3
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


@dataclass
class TrainingConfig:
    """Training configuration."""
    max_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"
    early_stopping_patience: int = 10
    checkpoint_dir: str = "models/checkpoints"
    log_dir: str = "experiments/logs"


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int = 42
    use_wandb: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            experiment_name=config_dict['experiment_name'],
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            seed=config_dict.get('seed', 42),
            use_wandb=config_dict.get('use_wandb', False)
        )
    
    def to_yaml(self, yaml_path):
        """Save configuration to YAML file."""
        config_dict = {
            'experiment_name': self.experiment_name,
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'seed': self.seed,
            'use_wandb': self.use_wandb
        }
        
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Example configuration
def create_default_config():
    """Create default configuration."""
    config = ExperimentConfig(
        experiment_name="mfcc_baseline",
        data=DataConfig(
            metadata_path="data/splits/train.csv",
            audio_dir="data/processed",
            feature_dir="data/features/mfcc",
            feature_type="mfcc"
        ),
        model=ModelConfig(
            architecture="mlp",
            input_dim=200,
            num_classes=10
        ),
        training=TrainingConfig()
    )
    return config


if __name__ == "__main__":
    # Test configuration
    config = create_default_config()
    config.to_yaml("experiments/configs/mfcc_baseline.yaml")
    print("✓ Default configuration saved")
    
    # Load it back
    loaded_config = ExperimentConfig.from_yaml("experiments/configs/mfcc_baseline.yaml")
    print(f"✓ Configuration loaded: {loaded_config.experiment_name}")
