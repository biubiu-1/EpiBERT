# EpiBERT PyTorch Lightning Implementation

This directory contains the production-ready PyTorch Lightning implementation of EpiBERT, providing a modern, maintainable alternative to the original TensorFlow implementation.

## Files Overview

- **`epibert_lightning.py`**: Core EpiBERT model using PyTorch Lightning
- **`data_module.py`**: Data loading and preprocessing module  
- **`train_lightning.py`**: Training script with command-line interface
- **`requirements_lightning.txt`**: Required dependencies
- **`execute_lightning_pretraining.sh`**: Run script for pretraining
- **`execute_lightning_finetuning.sh`**: Run script for fine-tuning

## Quick Start

### Installation
```bash
pip install -r requirements_lightning.txt
```

### Pretraining
```bash
./execute_lightning_pretraining.sh
```

### Fine-tuning
```bash
./execute_lightning_finetuning.sh
```

### Custom Training
```bash
python train_lightning.py \
    --data_dir /path/to/data \
    --batch_size 4 \
    --gpus 2 \
    --max_epochs 100 \
    --learning_rate 1e-4
```

## Model Architecture

EpiBERT Lightning maintains the original architecture:
- **Convolutional stems** for sequence and ATAC processing
- **Multi-head transformer** with rotary positional embeddings  
- **Masking strategy** for self-supervised learning
- **Multi-modal fusion** of sequence, ATAC, and motif data

## Key Features

- **Simplified training**: Lightning handles optimization, distributed training, and checkpointing
- **Modern tooling**: Built-in WandB logging, mixed precision, gradient clipping
- **Multi-GPU support**: Automatic distributed training
- **Production ready**: Clean, maintainable codebase following best practices

## Data Format

This implementation expects data in PyTorch-compatible format (HDF5 or PyTorch tensors). For converting from TFRecord format, see the data conversion utilities in the main data_processing directory.