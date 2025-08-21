import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "configs/experiments.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_float_maybe_frac(s: str) -> float:
    """Allow inputs like '4/255' or '0.0156'."""
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)

def get_experiment_config(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    """Get configuration for a specific experiment."""
    if experiment_name not in config["experiments"]:
        raise ValueError(f"Experiment '{experiment_name}' not found in config")
    
    exp_config = config["experiments"][experiment_name].copy()
    
    # Merge with base config
    exp_config.update({
        "model": config["model"],
        "data": config["data"],
        "wandb": config["wandb"],
        # Ensure save_images block is available to runners (for saving pairs)
        "save_images": config.get("save_images", {})
    })
    
    return exp_config

# Alias for backward compatibility
load_experiment_config = get_experiment_config
