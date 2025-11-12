"""Utility functions for TinyLlama fine-tuning application.

This module provides configuration loading, logging setup, model inference utilities,
data preprocessing helpers, and performance monitoring functions.
"""

import os
import json
import yaml
import logging
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import torch
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration dataclass for training parameters."""

    # Model configuration
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_length: int = 512

    # Training parameters
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None

    # Data parameters
    train_file: str = "data/train.json"
    eval_file: str = "data/eval.json"
    max_samples: int = 1000

    # Output parameters
    output_dir: str = "outputs"
    run_name: str = "tinyllama-finetune"

    def __post_init__(self):
        """Set default target modules if None."""
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "v_proj"]


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        TrainingConfig object with loaded configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested config structure
        flat_config = {}

        # Model config
        if 'model' in config_dict:
            flat_config['model_name'] = config_dict['model'].get(
                'name', TrainingConfig.model_name)
            flat_config['max_length'] = config_dict['model'].get(
                'max_length', TrainingConfig.max_length)

        # Training config
        if 'training' in config_dict:
            training = config_dict['training']
            flat_config.update({
                'batch_size': training.get('batch_size', TrainingConfig.batch_size),
                'learning_rate': training.get('learning_rate', TrainingConfig.learning_rate),
                'num_epochs': training.get('num_epochs', TrainingConfig.num_epochs),
                'gradient_accumulation_steps': training.get('gradient_accumulation_steps', TrainingConfig.gradient_accumulation_steps),
                'warmup_steps': training.get('warmup_steps', TrainingConfig.warmup_steps),
                'save_steps': training.get('save_steps', TrainingConfig.save_steps),
                'eval_steps': training.get('eval_steps', TrainingConfig.eval_steps),
                'logging_steps': training.get('logging_steps', TrainingConfig.logging_steps),
            })

        # LoRA config
        if 'lora' in config_dict:
            lora = config_dict['lora']
            flat_config.update({
                'lora_r': lora.get('r', TrainingConfig.lora_r),
                'lora_alpha': lora.get('alpha', TrainingConfig.lora_alpha),
                'lora_dropout': lora.get('dropout', TrainingConfig.lora_dropout),
                'lora_target_modules': lora.get('target_modules', TrainingConfig.lora_target_modules),
            })

        # Data config
        if 'data' in config_dict:
            data = config_dict['data']
            flat_config.update({
                'train_file': data.get('train_file', TrainingConfig.train_file),
                'eval_file': data.get('eval_file', TrainingConfig.eval_file),
                'max_samples': data.get('max_samples', TrainingConfig.max_samples),
            })

        # Output config
        if 'output' in config_dict:
            output = config_dict['output']
            flat_config.update({
                'output_dir': output.get('output_dir', TrainingConfig.output_dir),
                'run_name': output.get('run_name', TrainingConfig.run_name),
            })

        return TrainingConfig(**flat_config)

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """Set up structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Optional custom log format

    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )

    # Create logger
    logger = logging.getLogger("tinyllama_finetune")
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


def get_device() -> torch.device:
    """Get the best available device for training.

    Returns:
        torch.device object (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.

    Returns:
        Dictionary with memory usage information
    """
    usage = {}

    if torch.cuda.is_available():
        usage['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        usage['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3   # GB
        usage['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / \
            1024**3  # GB

    # Note: For system memory, you might want to use psutil if available
    # For simplicity, we'll just include GPU memory here

    return usage


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save
        filepath: Path to save the file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load data from JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is malformed
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path to ensure exists
    """
    os.makedirs(directory, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


class Timer:
    """Simple timer context manager for performance monitoring."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.start_time is not None:
            duration = self.end_time - self.start_time
            print(f"{self.name} completed in {duration:.2f} seconds")

    @property
    def duration(self) -> Optional[float]:
        """Get duration if timer has completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def validate_config(config: TrainingConfig) -> None:
    """Validate configuration parameters.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if config.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")

    if config.max_length <= 0:
        raise ValueError("max_length must be positive")

    if config.lora_r <= 0:
        raise ValueError("lora_r must be positive")

    if not (0 <= config.lora_dropout <= 1):
        raise ValueError("lora_dropout must be between 0 and 1")

    # Check if data files exist
    if not os.path.exists(config.train_file):
        raise FileNotFoundError(
            f"Training file not found: {config.train_file}")
