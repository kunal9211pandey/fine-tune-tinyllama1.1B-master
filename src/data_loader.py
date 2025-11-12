"""Data loading and preprocessing for TinyLlama fine-tuning.

This module handles JSON dataset loading, prompt templating, tokenization,
and train/validation splitting for instruction-following datasets.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import Dataset as HFDataset
import random

from .utils import TrainingConfig, load_json


logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """Dataset class for instruction-following data."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt_template: Optional[str] = None
    ):
        """Initialize the dataset.

        Args:
            data: List of instruction examples
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            prompt_template: Template for formatting prompts
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or self._get_default_template()

        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_default_template(self) -> str:
        """Get default Alpaca-style prompt template."""
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        )

    def _format_prompt(self, example: Dict[str, str]) -> str:
        """Format a single example using the prompt template.

        Args:
            example: Single training example

        Returns:
            Formatted prompt string
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        # Handle cases where input might be empty
        if not input_text.strip():
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Response:\n{output}"
            )
            return template.format(
                instruction=instruction,
                output=output_text
            )

        return self.prompt_template.format(
            instruction=instruction,
            input=input_text,
            output=output_text
        )

    def _tokenize_example(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text example.

        Args:
            text: Text to tokenize

        Returns:
            Dictionary with tokenized inputs
        """
        # Tokenize the full text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # For causal language modeling, labels are the same as input_ids
        # but we typically mask the instruction part and only compute loss on the response
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # For causal LM, labels = input_ids
        }

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset.

        Args:
            idx: Index of the example

        Returns:
            Tokenized example
        """
        example = self.data[idx]
        formatted_text = self._format_prompt(example)
        return self._tokenize_example(formatted_text)


class DataCollator:
    """Custom data collator for dynamic padding."""

    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_multiple_of: Optional[int] = None):
        """Initialize the data collator.

        Args:
            tokenizer: Tokenizer to use for padding
            pad_to_multiple_of: Pad to multiple of this value
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples.

        Args:
            batch: List of tokenized examples

        Returns:
            Batched and padded examples
        """
        # Extract individual components
        input_ids = [example["input_ids"] for example in batch]
        attention_masks = [example["attention_mask"] for example in batch]
        labels = [example["labels"] for example in batch]

        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            # -100 is ignored in loss calculation
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }


def load_dataset_from_json(
    file_path: str,
    max_samples: Optional[int] = None,
    validation_split: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load and split dataset from JSON file.

    Args:
        file_path: Path to JSON file
        max_samples: Maximum number of samples to load
        validation_split: Fraction of data to use for validation
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_data, val_data)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data format is invalid
    """
    logger.info(f"Loading dataset from {file_path}")

    try:
        data = load_json(file_path)
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise ValueError(f"Invalid JSON format: {e}")

    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of examples")

    # Validate data format
    for i, example in enumerate(data[:5]):  # Check first 5 examples
        if not isinstance(example, dict):
            raise ValueError(f"Example {i} is not a dictionary")

        required_keys = {"instruction", "output"}
        if not required_keys.issubset(example.keys()):
            missing_keys = required_keys - example.keys()
            raise ValueError(
                f"Example {i} missing required keys: {missing_keys}")

    logger.info(f"Loaded {len(data)} examples from dataset")

    # Limit samples if specified
    if max_samples and len(data) > max_samples:
        random.seed(seed)
        data = random.sample(data, max_samples)
        logger.info(f"Limited dataset to {max_samples} samples")

    # Split into train/validation
    if validation_split > 0:
        random.seed(seed)
        random.shuffle(data)

        val_size = int(len(data) * validation_split)
        val_data = data[:val_size]
        train_data = data[val_size:]

        logger.info(
            f"Split dataset: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data
    else:
        logger.info(
            f"Using all {len(data)} examples for training (no validation split)")
        return data, []


def create_dataloaders(
    config: TrainingConfig,
    tokenizer: PreTrainedTokenizer,
    eval_file: Optional[str] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation dataloaders.

    Args:
        config: Training configuration
        tokenizer: Tokenizer to use
        eval_file: Optional separate evaluation file

    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    logger.info("Creating dataloaders...")

    # Load training data
    if eval_file and Path(eval_file).exists():
        # Use separate files for train and eval
        train_data = load_json(config.train_file)
        eval_data = load_json(eval_file)

        # Limit samples if specified
        if config.max_samples:
            if len(train_data) > config.max_samples:
                train_data = train_data[:config.max_samples]

        logger.info(
            f"Using separate eval file: {len(train_data)} train, {len(eval_data)} eval")
    else:
        # Split single file into train/eval
        train_data, eval_data = load_dataset_from_json(
            config.train_file,
            max_samples=config.max_samples,
            validation_split=0.1
        )

    # Create datasets
    train_dataset = InstructionDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=config.max_length
    )

    eval_dataset = None
    if eval_data:
        eval_dataset = InstructionDataset(
            data=eval_data,
            tokenizer=tokenizer,
            max_length=config.max_length
        )

    # Create data collator
    data_collator = DataCollator(tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=torch.cuda.is_available(),
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            pin_memory=torch.cuda.is_available(),
            num_workers=0
        )

    logger.info(f"Created dataloaders: train_batches={len(train_dataloader)}, "
                f"eval_batches={len(eval_dataloader) if eval_dataloader else 0}")

    return train_dataloader, eval_dataloader


def validate_dataset_format(file_path: str) -> bool:
    """Validate that a dataset file has the correct format.

    Args:
        file_path: Path to dataset file

    Returns:
        True if format is valid, False otherwise
    """
    try:
        data = load_json(file_path)

        if not isinstance(data, list):
            logger.error("Dataset must be a list")
            return False

        if len(data) == 0:
            logger.error("Dataset is empty")
            return False

        # Check required fields in first few examples
        required_fields = {"instruction", "output"}
        for i, example in enumerate(data[:min(5, len(data))]):
            if not isinstance(example, dict):
                logger.error(f"Example {i} is not a dictionary")
                return False

            if not required_fields.issubset(example.keys()):
                missing = required_fields - example.keys()
                logger.error(f"Example {i} missing required fields: {missing}")
                return False

        logger.info(f"Dataset format is valid: {len(data)} examples")
        return True

    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        return False


def preview_dataset(file_path: str, num_examples: int = 3) -> None:
    """Preview examples from a dataset file.

    Args:
        file_path: Path to dataset file
        num_examples: Number of examples to show
    """
    try:
        data = load_json(file_path)
        logger.info(
            f"Dataset preview ({num_examples} examples from {len(data)} total):")

        for i, example in enumerate(data[:num_examples]):
            logger.info(f"\nExample {i+1}:")
            logger.info(
                f"  Instruction: {example.get('instruction', 'N/A')[:100]}...")
            logger.info(f"  Input: {example.get('input', 'N/A')[:100]}...")
            logger.info(f"  Output: {example.get('output', 'N/A')[:100]}...")

    except Exception as e:
        logger.error(f"Error previewing dataset: {e}")


# Utility function for creating custom prompt templates
def create_custom_template(
    instruction_prefix: str = "### Instruction:\n",
    input_prefix: str = "### Input:\n",
    response_prefix: str = "### Response:\n",
    system_message: str = ""
) -> str:
    """Create a custom prompt template.

    Args:
        instruction_prefix: Prefix for instruction section
        input_prefix: Prefix for input section
        response_prefix: Prefix for response section
        system_message: Optional system message at the beginning

    Returns:
        Formatted prompt template string
    """
    template_parts = []

    if system_message:
        template_parts.append(system_message.strip())
        template_parts.append("")

    template_parts.extend([
        f"{instruction_prefix}{{instruction}}",
        "",
        f"{input_prefix}{{input}}",
        "",
        f"{response_prefix}{{output}}"
    ])

    return "\n".join(template_parts)
