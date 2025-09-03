# EpiBERT PyTorch Lightning Transfer Guide

This directory contains a PyTorch Lightning implementation of the EpiBERT model, demonstrating how the original TensorFlow-based architecture can be transferred to the PyTorch Lightning framework.

## Files Overview

- **`epibert_lightning.py`**: Main model implementation using PyTorch Lightning
- **`data_module.py`**: Data loading and preprocessing module
- **`train_lightning.py`**: Training script with command-line interface
- **`requirements_lightning.txt`**: PyTorch Lightning ecosystem dependencies

## Key Differences from Original TensorFlow Implementation

### 1. Framework Migration
- **TensorFlow → PyTorch**: Complete migration from TensorFlow/Keras to PyTorch
- **Custom training loops → Lightning Module**: Replaces manual `@tf.function` decorated training steps
- **tf.distribute.Strategy → Lightning distributed training**: Built-in distributed training support

### 2. Model Architecture
- **tf.keras.Model → pl.LightningModule**: Inherits from Lightning base class
- **Custom layers**: Reimplemented TensorFlow layers in PyTorch (SoftmaxPooling, ConvBlock, etc.)
- **Attention mechanism**: Custom multi-head attention with rotary embeddings
- **Same architectural principles**: Maintains convolutional stems, transformer layers, and output heads

### 3. Training Infrastructure
- **Automatic optimization**: Lightning handles optimizer steps, gradient clipping, etc.
- **Built-in callbacks**: ModelCheckpoint, EarlyStopping, LearningRateMonitor
- **Integrated logging**: Seamless WandB integration
- **Distributed training**: Multi-GPU support with minimal code changes

### 4. Data Pipeline
- **TFRecord → PyTorch Dataset**: Conversion from TensorFlow data format
- **Custom augmentations**: Equivalent data augmentation pipeline
- **DataModule pattern**: Organized data loading with Lightning DataModule

## Effort Assessment

### Complete Transfer Requirements:

1. **Model Architecture (HIGH EFFORT)**
   - Convert all TensorFlow layers to PyTorch equivalents
   - Reimplement custom layers (attention, pooling, etc.)
   - Ensure mathematical equivalence between frameworks
   - Estimated: **3-4 weeks** for experienced developer

2. **Data Pipeline (MEDIUM-HIGH EFFORT)**
   - Convert TFRecord files to PyTorch-compatible format (HDF5/custom)
   - Reimplement data augmentation pipeline
   - Ensure identical preprocessing steps
   - Estimated: **2-3 weeks**

3. **Training Infrastructure (LOW-MEDIUM EFFORT)**
   - Lightning provides most training infrastructure
   - Configure callbacks, logging, distributed training
   - Adapt hyperparameters and schedules
   - Estimated: **1-2 weeks**

4. **Testing and Validation (MEDIUM EFFORT)**
   - Ensure model outputs match between frameworks
   - Validate training convergence
   - Test distributed training setup
   - Estimated: **2-3 weeks**

### **Total Estimated Effort: 8-12 weeks** for complete transfer

## Benefits of Lightning Transfer

### 1. **Simplified Training Code**
```python
# TensorFlow (manual training loop)
@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Lightning (automatic)
def training_step(self, batch, batch_idx):
    predictions = self(batch['inputs'])
    loss = F.mse_loss(predictions, batch['targets'])
    return loss
```

### 2. **Built-in Best Practices**
- Automatic gradient clipping
- Mixed precision training
- Learning rate scheduling
- Model checkpointing
- Early stopping

### 3. **Better Distributed Training**
- Multi-GPU training with 1-line change
- Automatic gradient synchronization
- Memory optimization

### 4. **Enhanced Monitoring**
- Integrated WandB logging
- Automatic metric tracking
- Progress bars and status monitoring

## Usage Example

```bash
# Install dependencies
pip install -r requirements_lightning.txt

# Train model
python train_lightning.py \
    --data_dir /path/to/converted/data \
    --batch_size 4 \
    --gpus 2 \
    --max_epochs 100 \
    --learning_rate 1e-4
```

## Data Conversion Required

The original EpiBERT uses TFRecord format. For PyTorch Lightning, you would need to:

1. **Convert TFRecord to HDF5/PyTorch format**
2. **Implement equivalent data augmentation**
3. **Ensure identical preprocessing pipeline**

See `data_module.py` for the structure and `convert_tfrecord_to_hdf5()` function for conversion approach.

## Key Architectural Components Preserved

1. **Convolutional stems** for sequence and ATAC processing
2. **Multi-head attention** with rotary positional embeddings
3. **Transformer layers** with residual connections
4. **Masking strategy** for self-supervised learning
5. **Multi-modal fusion** of sequence, ATAC, and motif data

## Performance Considerations

- **Memory usage**: PyTorch may have different memory patterns
- **Training speed**: Need to benchmark against TensorFlow version
- **Convergence**: Ensure identical hyperparameters and schedules
- **Distributed scaling**: Lightning provides better multi-GPU scaling

This implementation demonstrates the feasibility of transferring EpiBERT to PyTorch Lightning while maintaining the core architectural principles and training objectives.