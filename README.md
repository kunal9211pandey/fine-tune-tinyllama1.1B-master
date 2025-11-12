# TinyLlama 1.1B Fine-tuning Application

A clean and simple Python proof-of-concept application for fine-tuning TinyLlama 1.1B model using modern ML frameworks with LoRA (Low-Rank Adaptation) for efficient fine-tuning.

## Features

- 🚀 **Easy to Use**: Simple command-line interface for training, evaluation, and inference
- 🎯 **Memory Efficient**: LoRA fine-tuning works on GPUs with 8GB+ VRAM
- 🔧 **Configurable**: YAML-based configuration with command-line overrides
- 📊 **Comprehensive Logging**: Real-time training progress and metrics
- 💾 **Checkpointing**: Automatic model saving and resuming from checkpoints
- 🔍 **Evaluation**: Built-in evaluation metrics and model comparison
- 🎨 **Flexible Data**: Support for custom instruction-following datasets

## Project Structure

```
tinyllama-finetune/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Dataset loading and preprocessing
│   ├── model_handler.py     # Model loading and LoRA setup
│   ├── trainer.py           # Training loop implementation
│   └── utils.py             # Utility functions and configuration
├── data/
│   ├── train.json           # Training dataset
│   └── eval.json            # Evaluation dataset
├── outputs/
│   └── checkpoints/         # Model checkpoints
├── configs/
│   └── training_config.yaml # Training configuration
├── requirements.txt         # Python dependencies
├── main.py                  # Main application entry point
└── README.md                # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended, 8GB+ VRAM)
- Git

### Setup

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd tinyllama-finetune
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python main.py --help
   ```

## Quick Start

### 1. Training

Train with default configuration:

```bash
python main.py --mode train
```

Train with custom settings:

```bash
python main.py --mode train --batch_size 2 --learning_rate 1e-4 --num_epochs 5
```

### 2. Evaluation

Evaluate a trained model:

```bash
python main.py --mode eval --model_path outputs/checkpoints/best-step-1000
```

### 3. Inference

Generate text with your fine-tuned model:

```bash
python main.py --mode inference --model_path outputs/final --prompt "Explain quantum computing"
```

## Configuration

### Default Configuration (`configs/training_config.yaml`)

```yaml
model:
  name: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
  max_length: 512

training:
  batch_size: 4
  learning_rate: 2e-4
  num_epochs: 3
  gradient_accumulation_steps: 4
  warmup_steps: 100
  save_steps: 500
  eval_steps: 500
  logging_steps: 50

lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: ['q_proj', 'v_proj']

data:
  train_file: 'data/train.json'
  eval_file: 'data/eval.json'
  max_samples: 1000

output:
  output_dir: 'outputs'
  run_name: 'tinyllama-finetune'
```

### Command Line Overrides

Most configuration parameters can be overridden via command line:

```bash
python main.py --mode train \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --max_samples 500 \
  --config configs/custom_config.yaml
```

## Data Format

The application expects JSON files with instruction-following format:

```json
[
  {
    "instruction": "Explain the concept of machine learning",
    "input": "",
    "output": "Machine learning is a subset of artificial intelligence..."
  },
  {
    "instruction": "Translate the following text",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
  }
]
```

### Required Fields

- `instruction`: The task description
- `output`: The expected response

### Optional Fields

- `input`: Additional context or input data (can be empty)

## Advanced Usage

### Custom Configuration

Create your own configuration file:

```yaml
# configs/my_config.yaml
model:
  name: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
  max_length: 1024

training:
  batch_size: 8
  learning_rate: 5e-5
  num_epochs: 10
  gradient_accumulation_steps: 2

lora:
  r: 32
  alpha: 64
  target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj']
```

Use it with:

```bash
python main.py --mode train --config configs/my_config.yaml
```

### Memory Optimization

For limited GPU memory, try these options:

```bash
# Use 4-bit quantization
python main.py --mode train --use_4bit

# Use 8-bit quantization
python main.py --mode train --use_8bit

# Reduce batch size
python main.py --mode train --batch_size 1 --gradient_accumulation_steps 8
```

### Resume Training

Resume from a checkpoint:

```bash
python main.py --mode train --resume_from_checkpoint outputs/checkpoints/checkpoint-1000
```

### Data Validation

Validate your dataset before training:

```bash
python main.py --mode train --validate_data --preview_data
```

### Logging and Debugging

Enable detailed logging:

```bash
python main.py --mode train --log_level DEBUG --log_file training.log
```

## Model Performance

### Memory Requirements

| Configuration | GPU Memory | Training Speed |
| ------------- | ---------- | -------------- |
| Default LoRA  | ~6-8GB     | ~2-3 steps/sec |
| 8-bit + LoRA  | ~4-6GB     | ~1-2 steps/sec |
| 4-bit + LoRA  | ~3-4GB     | ~1 step/sec    |

### Recommended Settings

**For 8GB GPU:**

```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 4
lora:
  r: 16
  alpha: 32
```

**For 12GB+ GPU:**

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 2
lora:
  r: 32
  alpha: 64
```

## Output Files

After training, you'll find:

```
outputs/
├── checkpoints/
│   ├── best-step-X/         # Best model checkpoint
│   ├── checkpoint-X/        # Regular checkpoints
│   └── final/               # Final model
├── training_stats.json      # Training metrics
└── train_results.json       # Training summary
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce `batch_size`
   - Increase `gradient_accumulation_steps`
   - Use `--use_4bit` or `--use_8bit`

2. **Import Errors**

   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **Model Loading Issues**

   - Verify internet connection for initial model download
   - Check available disk space (model ~2GB)

4. **Data Format Errors**
   - Use `--validate_data` to check dataset format
   - Ensure JSON is properly formatted

### Performance Tips

1. **Faster Training:**

   - Use larger batch sizes if memory allows
   - Enable mixed precision training (automatic)
   - Use faster storage (SSD) for data

2. **Better Results:**
   - Increase number of epochs for small datasets
   - Use larger LoRA rank (r) for complex tasks
   - Provide more diverse training examples

## Examples

### Example 1: Basic Training

```bash
# Start basic training
python main.py --mode train

# Monitor progress in outputs/training_stats.json
# Models saved to outputs/checkpoints/
```

### Example 2: Custom Dataset

```bash
# Prepare your data in data/my_dataset.json
python main.py --mode train \
  --train_file data/my_dataset.json \
  --max_samples 2000 \
  --num_epochs 5
```

### Example 3: Inference Pipeline

```bash
# Train model
python main.py --mode train --num_epochs 3

# Evaluate best checkpoint
python main.py --mode eval --model_path outputs/checkpoints/best-step-1500

# Run inference
python main.py --mode inference \
  --model_path outputs/final \
  --prompt "Write a Python function to sort a list" \
  --max_length 200 \
  --temperature 0.8
```

## API Reference

### Command Line Arguments

| Argument          | Description                          | Default                        |
| ----------------- | ------------------------------------ | ------------------------------ |
| `--mode`          | Operation mode: train/eval/inference | Required                       |
| `--config`        | Configuration file path              | `configs/training_config.yaml` |
| `--model_path`    | Path to pretrained model             | None                           |
| `--batch_size`    | Training batch size                  | From config                    |
| `--learning_rate` | Learning rate                        | From config                    |
| `--num_epochs`    | Number of training epochs            | From config                    |
| `--max_samples`   | Limit training samples               | From config                    |
| `--use_4bit`      | Enable 4-bit quantization            | False                          |
| `--use_8bit`      | Enable 8-bit quantization            | False                          |
| `--no_lora`       | Disable LoRA (full fine-tuning)      | False                          |

### Configuration Options

See `configs/training_config.yaml` for all available options and their descriptions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [TinyLlama](https://github.com/jzhang38/TinyLlama) for the base model
- [Hugging Face](https://huggingface.co/) for transformers and datasets
- [Microsoft LoRA](https://github.com/microsoft/LoRA) for the LoRA implementation
- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tinyllama_finetune,
  title={TinyLlama Fine-tuning Application},
  author={kunal Pandey},
  year={2025},
  url={ https://github.com/kunal9211pandey/fine-tune-tinyllama1.1B-master}
}
```
