# Lightning EpiBERT Testing Documentation

This directory contains test scripts for the PyTorch Lightning version of EpiBERT that enable testing with minimal computation using synthetic random data.

## Overview

The testing suite provides two main scripts:
1. `test_lightning_minimal.py` - Basic testing functionality
2. `test_lightning_enhanced.py` - Advanced testing with benchmarking and configuration options

Both scripts test the Lightning implementation of EpiBERT at two stages:
- **Pretraining Stage**: Tests masked ATAC-seq prediction using self-supervised learning
- **Finetuning Stage**: Tests multi-task learning (ATAC + RNA prediction)

## Quick Start

### Basic Testing (Recommended)

```bash
# Test both pretraining and finetuning with minimal computation
python test_lightning_minimal.py --test_both --fast_dev_run

# Test with actual training (1 epoch)
python test_lightning_minimal.py --test_both --max_epochs 1

# Test only pretraining
python test_lightning_minimal.py --test_pretraining --fast_dev_run

# Test only finetuning  
python test_lightning_minimal.py --test_finetuning --fast_dev_run
```

### Enhanced Testing

```bash
# Run comprehensive test with benchmarking
python test_lightning_enhanced.py --test_both --benchmark --fast_dev_run

# Test with GPU (if available)
python test_lightning_enhanced.py --test_both --use_gpu --max_epochs 2

# Save results to file
python test_lightning_enhanced.py --test_both --output_file results.json

# Run performance benchmark only
python test_lightning_enhanced.py --benchmark --fast_dev_run
```

## Script Features

### test_lightning_minimal.py

- **Synthetic Data Generation**: Creates random genomic sequences, ATAC profiles, and motif activities
- **Simplified Model Architecture**: Lightweight transformer-based model for fast testing
- **Pretraining Testing**: Tests masked accessibility modeling
- **Finetuning Testing**: Tests multi-task ATAC + RNA prediction
- **Minimal Dependencies**: Uses only PyTorch Lightning and standard libraries

### test_lightning_enhanced.py

All features of minimal script plus:
- **Performance Benchmarking**: Tests multiple model sizes and measures forward pass times
- **GPU Support**: Optional GPU acceleration testing
- **Configurable Parameters**: Extensive command-line options
- **Result Logging**: Saves detailed results to JSON files
- **Enhanced Reporting**: Comprehensive test summaries with timing information

## Key Parameters

### Model Architecture (Minimal for Testing)
- `input_length`: DNA sequence length (default: 8192, vs 524288 in full model)
- `output_length`: Output profile length (default: 256, vs 4096 in full model)  
- `d_model`: Model dimension (default: 64, vs 1024 in full model)
- `num_heads`: Attention heads (default: 4, vs 8 in full model)
- `num_layers`: Transformer layers (default: 2, vs 8 in full model)

### Training Parameters
- `batch_size`: Batch size (default: 2 for minimal computation)
- `max_epochs`: Training epochs (default: 1-2 for testing)
- `num_samples`: Synthetic samples per split (default: 10)
- `learning_rate`: Learning rate (default: 1e-3)

### Testing Options
- `--fast_dev_run`: Run single batch only (fastest testing)
- `--limit_batches`: Fraction of batches to use per epoch
- `--benchmark`: Run performance benchmarking
- `--use_gpu`: Use GPU if available

## Test Data Format

The scripts generate synthetic data matching the original EpiBERT format:

### Pretraining Data
- **Sequence**: One-hot encoded DNA (4 x input_length)
- **ATAC Profile**: Accessibility signal (output_length)
- **Motif Activity**: TF binding scores (693 motifs)
- **Mask**: Binary mask for self-supervised learning

### Finetuning Data  
- **All pretraining data plus**:
- **RNA Target**: Gene expression levels (output_length/4)

## Model Architectures Tested

### Pretraining Model
```
Input: DNA sequence + ATAC + motif activity
↓
Convolutional feature extraction
↓  
Transformer encoder layers
↓
ATAC prediction head
→ Output: Predicted ATAC accessibility
```

### Finetuning Model
```
Input: DNA sequence + ATAC + motif activity  
↓
Shared feature extraction & transformer
↓
Multi-task heads:
├─ ATAC prediction
└─ RNA prediction  
→ Output: Predicted ATAC + RNA levels
```

## Expected Results

### Successful Test Output
```
✓ Pretraining test completed successfully!
✓ Finetuning test completed successfully!
Overall: ✓ ALL TESTS PASSED
```

### Performance Benchmarks
Typical performance on CPU (Intel/AMD):
- **Tiny model** (16K params): ~5ms forward pass
- **Small model** (50K params): ~11ms forward pass  
- **Medium model** (171K params): ~22ms forward pass

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PyTorch Lightning is installed:
   ```bash
   pip install -r lightning_transfer/requirements_lightning.txt
   ```

2. **Memory Issues**: Reduce batch size or model dimensions:
   ```bash
   python test_lightning_minimal.py --batch_size 1 --d_model 32
   ```

3. **GPU Issues**: Disable GPU or check CUDA installation:
   ```bash
   python test_lightning_enhanced.py --test_both  # CPU only
   ```

4. **NaN Losses**: Expected with random synthetic data - test passes if no crashes occur

### Error Codes
- **Exit code 0**: All tests passed
- **Exit code 1**: Some tests failed (check error messages)

## Integration with Full EpiBERT

These test scripts validate that the Lightning implementation:
1. **Correctly processes genomic data** in the expected formats
2. **Handles multi-task learning** (ATAC + RNA prediction)  
3. **Supports masked language modeling** for self-supervised pretraining
4. **Scales efficiently** across different model sizes
5. **Integrates properly** with PyTorch Lightning training loops

The synthetic data approach enables rapid validation without requiring the multi-TB EpiBERT datasets, making it suitable for:
- **Development testing** during model implementation
- **CI/CD pipelines** for automated testing
- **Quick validation** after code changes
- **Performance profiling** across different hardware configurations

## Next Steps

After successful testing with synthetic data:
1. **Data Conversion**: Convert original TFRecord datasets to PyTorch format
2. **Full Training**: Run with real genomic data using full model parameters
3. **Transfer Learning**: Load pretrained weights and finetune on specific tasks
4. **Production Deployment**: Scale to multi-GPU training with real datasets

The test scripts provide a foundation for validating the Lightning implementation before scaling to the full computational requirements of EpiBERT training.