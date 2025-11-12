#!/usr/bin/env python3
"""Main script for TinyLlama fine-tuning application.

This script provides command-line interface for training, evaluation, and inference
with TinyLlama 1.1B model using LoRA fine-tuning.
"""

from src.trainer import Trainer
from src.data_loader import create_dataloaders, validate_dataset_format, preview_dataset
from src.model_handler import ModelHandler
from src.utils import (
    TrainingConfig,
    load_config,
    setup_logging,
    validate_config,
    get_device,
    Timer
)
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="TinyLlama 1.1B Fine-tuning Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python main.py --mode train
  
  # Train with custom config
  python main.py --mode train --config configs/custom_config.yaml
  
  # Evaluate model
  python main.py --mode eval --model_path outputs/checkpoint-1000
  
  # Run inference
  python main.py --mode inference --model_path outputs/final --prompt "Explain machine learning"
  
  # Resume training from checkpoint
  python main.py --mode train --resume_from_checkpoint outputs/checkpoints/checkpoint-500
        """
    )

    # Main arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "inference"],
        help="Mode to run: train, eval, or inference"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to configuration file (default: configs/training_config.yaml)"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to pretrained model (for eval/inference mode)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Override model name from config"
    )

    # Training arguments
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Resume training from checkpoint directory"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size from config"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Override learning rate from config"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Override number of epochs from config"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        help="Limit number of training samples"
    )

    # Data arguments
    parser.add_argument(
        "--train_file",
        type=str,
        help="Override training data file from config"
    )

    parser.add_argument(
        "--eval_file",
        type=str,
        help="Override evaluation data file from config"
    )

    # Inference arguments
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for inference mode"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum length for generated text (default: 100)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation (default: 0.7)"
    )

    # Other arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory from config"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to log file (optional)"
    )

    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )

    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )

    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA fine-tuning (use full fine-tuning)"
    )

    parser.add_argument(
        "--preview_data",
        action="store_true",
        help="Preview dataset before training"
    )

    parser.add_argument(
        "--validate_data",
        action="store_true",
        help="Validate dataset format before training"
    )

    return parser.parse_args()


def load_and_override_config(args: argparse.Namespace, logger) -> TrainingConfig:
    """Load configuration and apply command line overrides.

    Args:
        args: Parsed command line arguments

    Returns:
        Configuration with overrides applied
    """
    # Load base configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        logger.warning(
            f"Config file not found: {args.config}. Using default configuration.")
        config = TrainingConfig()

    # Apply command line overrides
    if args.model_name:
        config.model_name = args.model_name
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.max_samples:
        config.max_samples = args.max_samples
    if args.train_file:
        config.train_file = args.train_file
    if args.eval_file:
        config.eval_file = args.eval_file
    if args.output_dir:
        config.output_dir = args.output_dir

    return config


def train_mode(args: argparse.Namespace, config: TrainingConfig, logger) -> Dict[str, Any]:
    """Run training mode.

    Args:
        args: Command line arguments
        config: Training configuration

    Returns:
        Training results
    """
    logger.info("=== TRAINING MODE ===")

    # Validate configuration
    validate_config(config)

    # Preview/validate data if requested
    if args.preview_data:
        preview_dataset(config.train_file)

    if args.validate_data:
        if not validate_dataset_format(config.train_file):
            raise ValueError("Training data validation failed")
        logger.info("Training data validation passed")

    # Create model handler
    model_handler = ModelHandler(config)

    # Load tokenizer and model
    with Timer("Loading tokenizer"):
        tokenizer = model_handler.load_tokenizer()

    with Timer("Loading model"):
        model = model_handler.load_model(
            use_lora=not args.no_lora,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit
        )

    # Print model info
    model_info = model_handler.get_model_info()
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")

    # Create dataloaders
    with Timer("Creating dataloaders"):
        train_dataloader, eval_dataloader = create_dataloaders(
            config=config,
            tokenizer=tokenizer,
            eval_file=args.eval_file or config.eval_file
        )

    # Create trainer
    trainer = Trainer(
        config=config,
        model_handler=model_handler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        trainer.resume_from_checkpoint(args.resume_from_checkpoint)

    # Start training
    with Timer("Training"):
        results = trainer.train()

    logger.info("Training completed successfully!")
    return results


def eval_mode(args: argparse.Namespace, config: TrainingConfig, logger) -> Dict[str, Any]:
    """Run evaluation mode.

    Args:
        args: Command line arguments
        config: Training configuration

    Returns:
        Evaluation results
    """
    logger.info("=== EVALUATION MODE ===")

    if not args.model_path:
        raise ValueError("--model_path is required for evaluation mode")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    # Create model handler
    model_handler = ModelHandler(config)

    # Load model and tokenizer
    with Timer("Loading model"):
        model = model_handler.load_pretrained_model(
            args.model_path,
            is_lora_checkpoint=True  # Assume LoRA checkpoint unless specified
        )

    # Print model info
    model_info = model_handler.get_model_info()
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")

    # Create evaluation dataloader
    eval_file = args.eval_file or config.eval_file
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    with Timer("Creating evaluation dataloader"):
        _, eval_dataloader = create_dataloaders(
            config=config,
            tokenizer=model_handler.tokenizer,
            eval_file=eval_file
        )

    if eval_dataloader is None:
        raise ValueError("Failed to create evaluation dataloader")

    # Create trainer for evaluation
    trainer = Trainer(
        config=config,
        model_handler=model_handler,
        train_dataloader=eval_dataloader,  # Dummy, won't be used
        eval_dataloader=eval_dataloader
    )

    # Run evaluation
    with Timer("Evaluation"):
        results = trainer._evaluate()

    logger.info("Evaluation Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")

    return results


def inference_mode(args: argparse.Namespace, config: TrainingConfig, logger) -> Dict[str, Any]:
    """Run inference mode.

    Args:
        args: Command line arguments
        config: Training configuration

    Returns:
        Inference results
    """
    logger.info("=== INFERENCE MODE ===")

    if not args.model_path:
        raise ValueError("--model_path is required for inference mode")

    if not args.prompt:
        raise ValueError("--prompt is required for inference mode")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    # Create model handler
    model_handler = ModelHandler(config)

    # Load model and tokenizer
    with Timer("Loading model"):
        model = model_handler.load_pretrained_model(
            args.model_path,
            is_lora_checkpoint=True
        )

    # Print model info
    model_info = model_handler.get_model_info()
    logger.info("Model Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")

    # Generate text
    logger.info(f"Prompt: {args.prompt}")
    logger.info("Generating response...")

    with Timer("Text generation"):
        generated_text = model_handler.generate_text(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            do_sample=True
        )

    logger.info(f"Generated text: {generated_text}")

    return {
        "prompt": args.prompt,
        "generated_text": generated_text,
        "max_length": args.max_length,
        "temperature": args.temperature
    }


def main():
    """Main entry point."""
    logger = None
    args = None

    try:
        # Parse arguments
        args = parse_arguments()

        # Setup logging
        logger = setup_logging(
            log_level=args.log_level,
            log_file=args.log_file
        )

        logger.info("TinyLlama Fine-tuning Application")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Device: {get_device()}")

        # Load configuration
        config = load_and_override_config(args, logger)

        # Run mode-specific logic
        if args.mode == "train":
            results = train_mode(args, config, logger)
        elif args.mode == "eval":
            results = eval_mode(args, config, logger)
        elif args.mode == "inference":
            results = inference_mode(args, config, logger)
        else:
            # Save results
            raise ValueError(f"Unknown mode: {args.mode}")
        if args.mode != "inference":
            results_file = os.path.join(
                config.output_dir, f"{args.mode}_results.json")
            os.makedirs(os.path.dirname(results_file), exist_ok=True)

            import json
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Results saved to {results_file}")

        logger.info("Application completed successfully!")

    except KeyboardInterrupt:
        if logger:
            logger.info("Application interrupted by user")
        else:
            print("Application interrupted by user")
        sys.exit(1)

    except Exception as e:
        if logger:
            logger.error(f"Application error: {e}")
            if args and args.log_level == "DEBUG":
                import traceback
                logger.error(traceback.format_exc())
        else:
            print(f"Application error: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
