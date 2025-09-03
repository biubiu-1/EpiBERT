# EpiBERT PyTorch Lightning Implementation

This directory contains a PyTorch Lightning implementation of EpiBERT that **exactly matches** the original TensorFlow model parameters and architecture.

## Model Parameter Alignment

The Lightning implementation automatically configures the correct parameters based on the model type:

### Pretraining Model (`model_type="pretraining"`)
Matches `src/models/epibert_atac_pretrain.py`:
- **Attention heads**: 8  
- **Transformer layers**: 8
- **Dropout rate**: 0.20
- **Pointwise dropout**: 0.10
- **Sequence filters**: [512, 640, 640, 768, 896, 1024]
- **d_model**: 1024 (derived from sequence filters)
- **ATAC filters**: [32, 64]

### Fine-tuning Model (`model_type="finetuning"`)  
Matches `src/models/epibert_rampage_finetune.py`:
- **Attention heads**: 4
- **Transformer layers**: 7  
- **Dropout rate**: 0.2
- **Pointwise dropout**: 0.2
- **Sequence filters**: [768, 896, 1024, 1024, 1152, 1280]
- **d_model**: 1280 (derived from sequence filters)
- **ATAC filters**: [32, 64]

## Files Overview

- **`epibert_lightning.py`**: Core EpiBERT model using PyTorch Lightning with parameter verification
- **`data_module.py`**: Data loading and preprocessing module  
- **`train_lightning.py`**: Training script with automatic model type configuration
- **`requirements_lightning.txt`**: Required dependencies
- **`execute_lightning_pretraining.sh`**: Run script for pretraining (uses pretraining model)
- **`execute_lightning_finetuning.sh`**: Run script for fine-tuning (uses fine-tuning model)

## Quick Start

### Installation
```bash
pip install -r requirements_lightning.txt
```

### Pretraining (8 heads, 8 layers, d_model=1024)
```bash
./execute_lightning_pretraining.sh <data_dir> <output_dir> [gpus] [project_name]
```

### Fine-tuning (4 heads, 7 layers, d_model=1280)
```bash
./execute_lightning_finetuning.sh <data_dir> <pretrained_checkpoint> <output_dir> [gpus] [project_name]
```

### Custom Training
```bash
# Use auto-configured pretraining model
python train_lightning.py --model_type pretraining --data_dir /path/to/data

# Use auto-configured fine-tuning model  
python train_lightning.py --model_type finetuning --data_dir /path/to/data

# Override specific parameters
python train_lightning.py --model_type pretraining --num_heads 16 --dropout_rate 0.1
```

## Python API Usage

```python
from epibert_lightning import EpiBERTLightning

# Pretraining model (auto-configured to match original TensorFlow)
model_pretrain = EpiBERTLightning(model_type="pretraining")

# Fine-tuning model (auto-configured to match original TensorFlow)
model_finetune = EpiBERTLightning(model_type="finetuning")

# Manual parameter specification (overrides auto-configuration)
model_custom = EpiBERTLightning(
    model_type="pretraining",
    num_heads=16,  # Override default
    learning_rate=5e-4
)
```

## Model Architecture

EpiBERT Lightning maintains the original architecture with exact parameter fidelity:
- **Convolutional stems** for sequence and ATAC processing
- **Multi-head transformer** with rotary positional embeddings  
- **Masking strategy** for self-supervised learning
- **Multi-modal fusion** of sequence, ATAC, and motif data

## Key Features

- ✅ **Exact parameter matching** with original TensorFlow models
- ✅ **Automatic configuration** based on model type (pretraining/finetuning)
- ✅ **Manual override support** for research experiments  
- ✅ **Production-ready** training scripts and execution workflows
- ✅ **Multi-GPU support** with distributed training
- ✅ **Mixed precision training** for improved performance
- ✅ **Comprehensive logging** with Weights & Biases integration
- ✅ **Parameter verification** at model instantiation

## Data Format

This implementation expects data in PyTorch-compatible format (HDF5 or PyTorch tensors). For converting from TFRecord format, see the data conversion utilities in the main data_processing directory.

## Architecture Verification

The implementation includes automatic parameter verification to ensure exact alignment with the original TensorFlow models. All model parameters are validated at instantiation to guarantee architectural fidelity.