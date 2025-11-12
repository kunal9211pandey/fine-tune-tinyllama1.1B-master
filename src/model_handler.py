"""Model handling for TinyLlama fine-tuning.

This module handles loading TinyLlama 1.1B model from HuggingFace,
setting up LoRA for efficient fine-tuning, device management, and model saving/loading.
"""

import os
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)

from .utils import TrainingConfig, get_device, count_parameters


logger = logging.getLogger(__name__)


class ModelHandler:
    """Handles model loading, LoRA setup, and device management."""

    def __init__(self, config: TrainingConfig):
        """Initialize the model handler.

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = get_device()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.is_lora_model = False

        logger.info(f"Using device: {self.device}")

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and configure the tokenizer.

        Returns:
            Configured tokenizer
        """
        logger.info(f"Loading tokenizer: {self.config.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                use_fast=True
            )

            # Set special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")

            # Ensure we have all required special tokens
            special_tokens = {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }

            tokens_to_add = {}
            for token_name, token_value in special_tokens.items():
                if getattr(self.tokenizer, token_name) is None:
                    tokens_to_add[token_name] = token_value

            if tokens_to_add:
                self.tokenizer.add_special_tokens(tokens_to_add)
                logger.info(f"Added special tokens: {tokens_to_add}")

            logger.info(
                f"Tokenizer loaded successfully. Vocab size: {len(self.tokenizer)}")
            return self.tokenizer

        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise

    def load_model(
        self,
        use_lora: bool = True,
        use_4bit: bool = False,
        use_8bit: bool = False
    ) -> PreTrainedModel:
        """Load and configure the model.

        Args:
            use_lora: Whether to use LoRA fine-tuning
            use_4bit: Whether to use 4-bit quantization
            use_8bit: Whether to use 8-bit quantization

        Returns:
            Configured model
        """
        logger.info(f"Loading model: {self.config.model_name}")

        # Configure quantization if requested
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization")
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization")

        try:
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "device_map": "auto" if quantization_config else None,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )

            # Move to device if not using device_map
            if not quantization_config:
                self.model = self.model.to(self.device)

            # Resize token embeddings if tokenizer was modified
            if self.tokenizer and len(self.tokenizer) != self.model.config.vocab_size:
                logger.info(
                    f"Resizing token embeddings from {self.model.config.vocab_size} to {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer))

            # Configure for training
            self.model.config.use_cache = False  # Disable cache for training

            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")

            # Set up LoRA if requested
            if use_lora:
                self._setup_lora()

            # Count parameters
            param_count = count_parameters(self.model)
            logger.info(f"Model loaded successfully:")
            logger.info(f"  Total parameters: {param_count['total']:,}")
            logger.info(
                f"  Trainable parameters: {param_count['trainable']:,}")
            logger.info(f"  Frozen parameters: {param_count['frozen']:,}")

            return self.model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _setup_lora(self) -> None:
        """Set up LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        logger.info("Setting up LoRA configuration")

        try:
            # Prepare model for k-bit training if quantized
            if hasattr(self.model, "is_loaded_in_8bit") and self.model.is_loaded_in_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
                logger.info("Prepared model for 8-bit training")
            elif hasattr(self.model, "is_loaded_in_4bit") and self.model.is_loaded_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
                logger.info("Prepared model for 4-bit training")

            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)
            self.is_lora_model = True

            # Print trainable parameters
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())

            logger.info(f"LoRA setup complete:")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(
                f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")

        except Exception as e:
            logger.error(f"Error setting up LoRA: {e}")
            raise

    def save_model(self, output_dir: str, save_full_model: bool = False) -> None:
        """Save the model and tokenizer.

        Args:
            output_dir: Directory to save the model
            save_full_model: Whether to save the full model or just LoRA weights
        """
        if self.model is None:
            raise ValueError("No model loaded to save")

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model to {output_dir}")

        try:
            if self.is_lora_model and not save_full_model:
                # Save only LoRA weights
                self.model.save_pretrained(output_dir)
                logger.info("Saved LoRA weights")
            else:
                # Save full model
                if self.is_lora_model:
                    # Merge LoRA weights and save full model
                    merged_model = self.model.merge_and_unload()
                    merged_model.save_pretrained(output_dir)
                    logger.info("Merged LoRA weights and saved full model")
                else:
                    # Save regular fine-tuned model
                    self.model.save_pretrained(output_dir)
                    logger.info("Saved full model")

            # Save tokenizer
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_dir)
                logger.info("Saved tokenizer")

            # Save configuration
            config_path = os.path.join(output_dir, "training_config.json")
            import json
            with open(config_path, 'w') as f:
                # Convert config to dict, handling non-serializable types
                config_dict = {}
                for key, value in self.config.__dict__.items():
                    try:
                        json.dumps(value)  # Test if serializable
                        config_dict[key] = value
                    except (TypeError, ValueError):
                        config_dict[key] = str(value)

                json.dump(config_dict, f, indent=2)
            logger.info("Saved training configuration")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_pretrained_model(
        self,
        model_path: str,
        is_lora_checkpoint: bool = True
    ) -> PreTrainedModel:
        """Load a pretrained model for inference or continued training.

        Args:
            model_path: Path to the saved model
            is_lora_checkpoint: Whether the checkpoint contains LoRA weights

        Returns:
            Loaded model
        """
        logger.info(f"Loading pretrained model from {model_path}")

        try:
            if is_lora_checkpoint:
                # Load base model first
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    trust_remote_code=True
                )

                # Load LoRA weights
                self.model = PeftModel.from_pretrained(base_model, model_path)
                self.is_lora_model = True
                logger.info("Loaded LoRA checkpoint")
            else:
                # Load full fine-tuned model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                self.is_lora_model = False
                logger.info("Loaded full model checkpoint")

            self.model = self.model.to(self.device)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            return self.model

        except Exception as e:
            logger.error(f"Error loading pretrained model: {e}")
            raise

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, list]:
        """Generate text using the loaded model.

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text(s)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        self.model.eval()

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=len(inputs.input_ids[0]) + max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode outputs
            generated_texts = []
            for output in outputs:
                # Remove input tokens from output
                generated_tokens = output[len(inputs.input_ids[0]):]
                generated_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text.strip())

            return generated_texts[0] if num_return_sequences == 1 else generated_texts

        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}

        param_count = count_parameters(self.model)

        info = {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "is_lora_model": self.is_lora_model,
            "total_parameters": param_count["total"],
            "trainable_parameters": param_count["trainable"],
            "frozen_parameters": param_count["frozen"],
            "trainable_percentage": 100 * param_count["trainable"] / param_count["total"],
        }

        if hasattr(self.model, "config"):
            info.update({
                "vocab_size": getattr(self.model.config, "vocab_size", "Unknown"),
                "hidden_size": getattr(self.model.config, "hidden_size", "Unknown"),
                "num_layers": getattr(self.model.config, "num_hidden_layers", "Unknown"),
                "num_attention_heads": getattr(self.model.config, "num_attention_heads", "Unknown"),
            })

        return info

    def enable_training_mode(self) -> None:
        """Set the model to training mode."""
        if self.model is not None:
            self.model.train()

    def enable_eval_mode(self) -> None:
        """Set the model to evaluation mode."""
        if self.model is not None:
            self.model.eval()


def create_model_handler(config: TrainingConfig) -> ModelHandler:
    """Factory function to create a ModelHandler instance.

    Args:
        config: Training configuration

    Returns:
        Configured ModelHandler instance
    """
    return ModelHandler(config)
