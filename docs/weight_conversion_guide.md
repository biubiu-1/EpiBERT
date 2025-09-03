# TensorFlow to PyTorch Weight Conversion Guide

This guide explains how to convert and use EpiBERT TensorFlow checkpoints with the PyTorch Lightning implementation.

## Quick Start

### 1. Convert TensorFlow Checkpoint
```bash
# Convert a TensorFlow checkpoint to PyTorch format
./scripts/weight_manager.py convert path/to/tf_model.ckpt path/to/pytorch_model.pth --model-type pretraining

# For fine-tuning model
./scripts/weight_manager.py convert path/to/tf_finetune.ckpt path/to/pytorch_finetune.pth --model-type finetuning
```

### 2. Load Directly into PyTorch Model
```python
from lightning_transfer.epibert_lightning import EpiBERTLightning

# Create model and load TensorFlow weights
model = EpiBERTLightning(model_type="pretraining")
stats = model.load_tensorflow_weights("path/to/tf_model.ckpt")

# Validate weights match original
validation = model.validate_weights_against_tensorflow("path/to/tf_model.ckpt")
print(f"Validation passed: {validation['validation_passed']}")
```

### 3. Using the Weight Manager Utility
```bash
# Load and validate TensorFlow weights
./scripts/weight_manager.py load path/to/tf_model.ckpt --model-type pretraining --save validated_model.pth

# Create demo model for testing
./scripts/weight_manager.py demo --model-type pretraining --save demo_model.pth

# List available checkpoints
./scripts/weight_manager.py list /path/to/checkpoint/directory

# Generate weight mapping guide
./scripts/weight_manager.py guide --model-type pretraining --output mapping_guide.txt
```

## Model Types and Parameters

EpiBERT has two main configurations that exactly match the original TensorFlow implementations:

### Pretraining Model (`model_type="pretraining"`)
Matches `epibert_atac_pretrain.py`:
- **Attention heads**: 8
- **Transformer layers**: 8
- **Model dimension (d_model)**: 1024
- **Dropout rate**: 0.20
- **Pointwise dropout**: 0.10
- **Filter list**: [512, 640, 640, 768, 896, 1024]

### Fine-tuning Model (`model_type="finetuning"`)
Matches `epibert_rampage_finetune.py`:
- **Attention heads**: 4
- **Transformer layers**: 7
- **Model dimension (d_model)**: 1280
- **Dropout rate**: 0.2
- **Pointwise dropout**: 0.2
- **Filter list**: [768, 896, 1024, 1024, 1152, 1280]

## Weight Conversion Details

### Layer Name Mapping
The converter automatically maps TensorFlow layer names to PyTorch equivalents:

```
TensorFlow                          ->  PyTorch
transformer/layer_0/attention/      ->  transformer.layers.0.attention.
/query/kernel:0                     ->  .query_dense_layer.weight
/LayerNorm/gamma:0                  ->  .layer_norm.weight
stem_conv/kernel:0                  ->  stem_conv.weight
```

### Weight Tensor Transformations
- **Convolutional layers**: (H, W, in_ch, out_ch) → (out_ch, in_ch, H, W)
- **Dense/Linear layers**: (in_feat, out_feat) → (out_feat, in_feat)
- **LayerNorm parameters**: gamma → weight, beta → bias
- **Bias tensors**: No transformation needed

### Supported Checkpoint Formats
- TensorFlow checkpoint files (`.ckpt`)
- Keras HDF5 models (`.h5`)
- SavedModel directories
- TensorFlow variables

## Advanced Usage

### Custom Weight Loading
```python
from scripts.tf_to_pytorch_converter import TensorFlowToPyTorchConverter

# Create converter
converter = TensorFlowToPyTorchConverter(model_type="pretraining")

# Convert checkpoint
stats = converter.convert_checkpoint(
    tf_checkpoint_path="model.ckpt",
    output_path="converted_model.pth",
    validate=True
)

# Load into existing model
model = EpiBERTLightning(model_type="pretraining")
checkpoint = torch.load("converted_model.pth")
model.load_state_dict(checkpoint['state_dict'])
```

### Weight Validation
```python
# Validate converted weights against original
validation_results = model.validate_weights_against_tensorflow(
    tf_checkpoint_path="original_model.ckpt",
    tolerance=1e-5
)

print(f"Total parameters matched: {validation_results['matched_params']}")
print(f"Validation passed: {validation_results['validation_passed']}")

# Check weight differences
for name, diff in validation_results['weight_differences']:
    if diff > 1e-5:
        print(f"Large difference in {name}: {diff}")
```

### Model Output Comparison
```python
from scripts.weight_manager import compare_model_outputs

# Create original and converted models
original_model = EpiBERTLightning(model_type="pretraining")
converted_model = EpiBERTLightning(model_type="pretraining")

# Load weights
original_model.load_tensorflow_weights("tf_model.ckpt")
converted_model = EpiBERTLightning.load_from_pytorch_checkpoint("pytorch_model.pth")

# Compare outputs
comparison = compare_model_outputs(original_model, converted_model, tolerance=1e-5)
print(f"Outputs match: {comparison['outputs_match']}")
```

## Data Format Requirements

Since TFRecord datasets are not used, you'll need to prepare data in PyTorch-compatible formats:

### Supported Data Formats
- **HDF5** (`.h5`) - Recommended for large datasets
- **NumPy** (`.npz`) - Good for medium datasets
- **Pickle** (`.pkl`) - For small datasets or complex structures

### Required Data Fields
```python
# HDF5 file structure
with h5py.File('data.h5', 'r') as f:
    sequences = f['sequences']      # Shape: (N, 4, seq_length) or (N, seq_length)
    atac_profiles = f['atac_profiles']  # Shape: (N, profile_length)
    motif_activities = f['motif_activities']  # Optional: (N, 693)
    peaks_centers = f['peaks_centers']        # Optional: (N, n_peaks)
```

### Creating Data Files
```python
from lightning_transfer.data_module import create_hdf5_from_arrays

# Convert your data to HDF5
create_hdf5_from_arrays(
    sequences=your_sequences,          # numpy array
    atac_profiles=your_atac_profiles,  # numpy array
    output_path="training_data.h5",
    motif_activities=your_motifs,      # optional
    peaks_centers=your_peaks          # optional
)
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the EpiBERT project directory
cd /path/to/EpiBERT
python -c "from scripts.tf_to_pytorch_converter import *"
```

**2. TensorFlow Not Available**
```bash
# Install TensorFlow for checkpoint reading
pip install tensorflow>=2.8.0
```

**3. Shape Mismatches**
The converter handles most shape transformations automatically. If you see shape errors:
- Check that the model_type matches your checkpoint
- Verify the TensorFlow checkpoint is valid
- Use the validation function to identify specific issues

**4. Weight Loading Failures**
```python
# Load with less strict requirements
stats = model.load_tensorflow_weights(tf_path, strict=False)

# Check what was loaded vs missing
print("Missing keys:", stats.get('missing_keys', []))
print("Unexpected keys:", stats.get('unexpected_keys', []))
```

### Validation Failures
If weight validation fails:
1. Check model_type matches the checkpoint
2. Verify TensorFlow checkpoint integrity
3. Use looser tolerance for validation
4. Check for version differences in the models

### Performance Considerations
- HDF5 format is fastest for large datasets
- Use `num_workers > 0` in DataLoader for parallel loading
- Consider data preprocessing and caching for repeated use

## Examples

### Complete Workflow Example
```python
#!/usr/bin/env python3
"""Complete example: Convert weights and train model"""

from lightning_transfer.epibert_lightning import EpiBERTLightning
from lightning_transfer.data_module import EpiBERTDataModule
import pytorch_lightning as pl

# 1. Create model with TensorFlow weights
model = EpiBERTLightning(model_type="pretraining")
model.load_tensorflow_weights("path/to/tf_pretrained.ckpt")

# 2. Validate weights
validation = model.validate_weights_against_tensorflow("path/to/tf_pretrained.ckpt")
assert validation['validation_passed'], "Weight validation failed!"

# 3. Set up data
data_module = EpiBERTDataModule(
    data_dir="path/to/data",
    batch_size=4,
    num_workers=4
)

# 4. Create trainer
trainer = pl.Trainer(
    max_epochs=10,
    devices=1,
    accelerator='gpu'
)

# 5. Train
trainer.fit(model, data_module)

# 6. Save final model
model.save_pytorch_checkpoint("final_model.pth")
```

### Batch Conversion Script
```bash
#!/bin/bash
# Convert multiple TensorFlow checkpoints

# Pretraining models
for ckpt in pretrain_*.ckpt; do
    output="pytorch_${ckpt%.ckpt}.pth"
    ./scripts/weight_manager.py convert "$ckpt" "$output" --model-type pretraining
done

# Fine-tuning models  
for ckpt in finetune_*.ckpt; do
    output="pytorch_${ckpt%.ckpt}.pth"
    ./scripts/weight_manager.py convert "$ckpt" "$output" --model-type finetuning
done
```

## API Reference

### TensorFlowToPyTorchConverter
Main conversion class with methods:
- `convert_checkpoint()`: Convert full checkpoint
- `_load_tf_checkpoint()`: Load TensorFlow weights
- `_convert_weights()`: Convert weight format
- `_validate_weights()`: Validate conversion

### EpiBERTLightning Weight Methods
- `load_tensorflow_weights()`: Load TF weights into model
- `save_pytorch_checkpoint()`: Save model weights
- `load_from_pytorch_checkpoint()`: Class method to load model
- `validate_weights_against_tensorflow()`: Compare with original

### Weight Manager Utility
Command-line tool with subcommands:
- `convert`: Convert TF to PyTorch
- `load`: Load and validate weights
- `demo`: Create demo model
- `list`: List available checkpoints
- `guide`: Generate mapping guide

This comprehensive system ensures seamless transition from TensorFlow to PyTorch while maintaining exact parameter compatibility.