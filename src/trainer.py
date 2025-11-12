"""Training logic for TinyLlama fine-tuning.

This module implements the training loop with gradient accumulation, evaluation,
checkpointing, metrics logging, and memory optimization techniques.
"""

import os
import math
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW 
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from tqdm.auto import tqdm
import json

from .utils import TrainingConfig, get_memory_usage, format_time, Timer
from .model_handler import ModelHandler


logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class for fine-tuning TinyLlama."""

    def __init__(
        self,
        config: TrainingConfig,
        model_handler: ModelHandler,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None
    ):
        """Initialize the trainer.

        Args:
            config: Training configuration
            model_handler: Model handler instance
            train_dataloader: Training data loader
            eval_dataloader: Optional evaluation data loader
        """
        self.config = config
        self.model_handler = model_handler
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.model = model_handler.model
        self.tokenizer = model_handler.tokenizer
        self.device = model_handler.device

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        self.training_stats = []

        # Initialize optimizer and scheduler (will be set in train())
        self.optimizer = None
        self.scheduler = None

        # Output directories
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Trainer initialized with {len(self.train_dataloader)} training batches")
        if self.eval_dataloader:
            logger.info(
                f"Evaluation dataloader has {len(self.eval_dataloader)} batches")

    def _setup_optimizer_and_scheduler(self) -> None:
        """Set up optimizer and learning rate scheduler."""
        # Count total training steps
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        total_steps = total_steps // self.config.gradient_accumulation_steps

        logger.info(f"Total training steps: {total_steps}")

        # Set up optimizer
        # Get parameters that require gradients
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            eps=1e-8
        )

        # Set up scheduler
        if self.config.warmup_steps > 0:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )

        logger.info(f"Optimizer: AdamW with lr={self.config.learning_rate}")
        logger.info(f"Scheduler: {'Linear' if self.config.warmup_steps > 0 else 'Cosine'} "
                    f"with {self.config.warmup_steps} warmup steps")

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch.

        Args:
            batch: Batch of tokenized inputs

        Returns:
            Loss tensor
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)
        return outputs.loss

    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation on the validation set.

        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}

        logger.info("Running evaluation...")
        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            eval_pbar = tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                leave=False,
                disable=len(self.eval_dataloader) < 10
            )

            for batch in eval_pbar:
                try:
                    loss = self._compute_loss(batch)
                    total_loss += loss.item()
                    total_batches += 1

                    eval_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                except Exception as e:
                    logger.warning(f"Error in evaluation batch: {e}")
                    continue

        if total_batches == 0:
            logger.warning("No valid evaluation batches")
            return {"eval_loss": float('inf')}

        avg_loss = total_loss / total_batches
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')

        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
            "eval_batches": total_batches
        }

        logger.info(
            f"Evaluation complete: loss={avg_loss:.4f}, perplexity={perplexity:.2f}")
        return metrics

    def _save_checkpoint(self, checkpoint_dir: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Save model checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint
            metrics: Optional metrics to save with checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        try:
            # Save model and tokenizer
            self.model_handler.save_model(
                checkpoint_dir, save_full_model=False)

            # Save training state
            state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_eval_loss": self.best_eval_loss,
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "config": self.config.__dict__,
            }

            if metrics:
                state["metrics"] = metrics

            state_path = os.path.join(checkpoint_dir, "training_state.json")
            with open(state_path, 'w') as f:
                # Convert non-serializable objects to strings
                serializable_state = {}
                for key, value in state.items():
                    try:
                        json.dumps(value)
                        serializable_state[key] = value
                    except (TypeError, ValueError):
                        if key.endswith("_state_dict"):
                            # Skip state dicts as they're not JSON serializable
                            continue
                        serializable_state[key] = str(value)

                json.dump(serializable_state, f, indent=2)

            logger.info(f"Checkpoint saved to {checkpoint_dir}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def _log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log training metrics.

        Args:
            metrics: Metrics to log
            step: Current step
        """
        # Add to training stats
        metrics["step"] = step
        self.training_stats.append(metrics.copy())

        # Log to console
        log_str = f"Step {step}"
        for key, value in metrics.items():
            if key != "step" and isinstance(value, (int, float)):
                if "loss" in key.lower():
                    log_str += f", {key}: {value:.4f}"
                elif "lr" in key.lower():
                    log_str += f", {key}: {value:.2e}"
                else:
                    log_str += f", {key}: {value}"

        logger.info(log_str)

        # Log memory usage
        if step % (self.config.logging_steps * 5) == 0:
            memory_usage = get_memory_usage()
            if memory_usage:
                memory_str = ", ".join(
                    [f"{k}: {v:.1f}GB" for k, v in memory_usage.items()])
                logger.info(f"Memory usage - {memory_str}")

    def train(self) -> Dict[str, Any]:
        """Main training loop.

        Returns:
            Training statistics
        """
        logger.info("Starting training...")

        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler()

        # Enable training mode
        self.model.train()

        # Training progress bar
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        pbar = tqdm(total=total_steps, desc="Training")

        running_loss = 0.0
        best_checkpoint_dir = None

        try:
            for epoch in range(self.config.num_epochs):
                self.epoch = epoch
                logger.info(
                    f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

                epoch_loss = 0.0
                batch_count = 0

                for batch_idx, batch in enumerate(self.train_dataloader):
                    try:
                        # Forward pass
                        loss = self._compute_loss(batch)

                        # Scale loss by accumulation steps
                        loss = loss / self.config.gradient_accumulation_steps

                        # Backward pass
                        loss.backward()

                        # Update running loss
                        running_loss += loss.item()
                        epoch_loss += loss.item()
                        batch_count += 1

                        # Gradient accumulation
                        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 1.0)

                            # Optimizer step
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad()

                            self.global_step += 1

                            # Update progress bar
                            pbar.update(
                                self.config.gradient_accumulation_steps)
                            pbar.set_postfix({
                                "loss": running_loss / self.config.gradient_accumulation_steps,
                                "lr": self.scheduler.get_last_lr()[0]
                            })

                            # Logging
                            if self.global_step % self.config.logging_steps == 0:
                                avg_loss = running_loss / self.config.gradient_accumulation_steps
                                metrics = {
                                    "train_loss": avg_loss,
                                    "learning_rate": self.scheduler.get_last_lr()[0],
                                    "epoch": epoch + (batch_idx + 1) / len(self.train_dataloader)
                                }
                                self._log_metrics(metrics, self.global_step)
                                running_loss = 0.0

                            # Evaluation
                            if (self.eval_dataloader and
                                    self.global_step % self.config.eval_steps == 0):

                                eval_metrics = self._evaluate()
                                if eval_metrics:
                                    self._log_metrics(
                                        eval_metrics, self.global_step)

                                    # Save best model
                                    eval_loss = eval_metrics.get(
                                        "eval_loss", float('inf'))
                                    if eval_loss < self.best_eval_loss:
                                        self.best_eval_loss = eval_loss
                                        best_checkpoint_dir = str(
                                            self.checkpoint_dir / f"best-step-{self.global_step}")
                                        self._save_checkpoint(
                                            best_checkpoint_dir, eval_metrics)
                                        logger.info(
                                            f"New best model saved with eval_loss: {eval_loss:.4f}")

                                # Return to training mode
                                self.model.train()

                            # Save checkpoint
                            if self.global_step % self.config.save_steps == 0:
                                checkpoint_dir = str(
                                    self.checkpoint_dir / f"checkpoint-{self.global_step}")
                                self._save_checkpoint(checkpoint_dir)

                    except Exception as e:
                        logger.error(
                            f"Error in training batch {batch_idx}: {e}")
                        continue

                # End of epoch logging
                if batch_count > 0:
                    avg_epoch_loss = epoch_loss / batch_count
                    logger.info(
                        f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

                # Run evaluation at end of epoch
                if self.eval_dataloader:
                    eval_metrics = self._evaluate()
                    if eval_metrics:
                        self._log_metrics(eval_metrics, self.global_step)
                    self.model.train()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

        finally:
            pbar.close()

        # Save final model
        final_dir = str(self.output_dir / "final")
        self._save_checkpoint(final_dir)
        logger.info(f"Final model saved to {final_dir}")

        # Save training statistics
        stats_path = str(self.output_dir / "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

        training_summary = {
            "total_steps": self.global_step,
            "epochs_completed": self.epoch + 1,
            "best_eval_loss": self.best_eval_loss,
            "best_checkpoint": best_checkpoint_dir,
            "final_checkpoint": final_dir,
            "training_stats_file": stats_path
        }

        logger.info("Training completed!")
        logger.info(f"Total steps: {self.global_step}")
        logger.info(f"Best eval loss: {self.best_eval_loss:.4f}")
        logger.info(f"Best model: {best_checkpoint_dir}")

        return training_summary

    def resume_from_checkpoint(self, checkpoint_dir: str) -> None:
        """Resume training from a checkpoint.

        Args:
            checkpoint_dir: Directory containing the checkpoint
        """
        logger.info(f"Resuming training from {checkpoint_dir}")

        try:
            # Load model
            self.model_handler.load_pretrained_model(
                checkpoint_dir, is_lora_checkpoint=True)
            self.model = self.model_handler.model

            # Load training state
            state_path = os.path.join(checkpoint_dir, "training_state.json")
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)

                self.global_step = state.get("global_step", 0)
                self.epoch = state.get("epoch", 0)
                self.best_eval_loss = state.get("best_eval_loss", float('inf'))

                logger.info(
                    f"Resumed from step {self.global_step}, epoch {self.epoch}")

        except Exception as e:
            logger.error(f"Error resuming from checkpoint: {e}")
            raise

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics.

        Returns:
            Dictionary with training statistics
        """
        if not self.training_stats:
            return {}

        # Calculate summary statistics
        train_losses = [s["train_loss"]
                        for s in self.training_stats if "train_loss" in s]
        eval_losses = [s["eval_loss"]
                       for s in self.training_stats if "eval_loss" in s]

        stats = {
            "total_steps": self.global_step,
            "epochs_completed": self.epoch + 1,
            "best_eval_loss": self.best_eval_loss,
            "num_train_losses": len(train_losses),
            "num_eval_losses": len(eval_losses),
        }

        if train_losses:
            stats.update({
                "final_train_loss": train_losses[-1],
                "min_train_loss": min(train_losses),
                "avg_train_loss": sum(train_losses) / len(train_losses)
            })

        if eval_losses:
            stats.update({
                "final_eval_loss": eval_losses[-1],
                "min_eval_loss": min(eval_losses),
                "avg_eval_loss": sum(eval_losses) / len(eval_losses)
            })

        return stats
