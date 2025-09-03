# Enhanced EpiBERT Data Handling for Batched Multi-Sample Datasets

This document describes the enhanced data handling capabilities for EpiBERT that support batched data input with multiple paired ATAC-seq and RAMPAGE-seq samples.

## Overview

The enhanced data handling system provides:

- **Paired Sample Management**: Handle multiple ATAC-seq and RAMPAGE-seq samples from different conditions
- **Batch Composition Control**: Configure how batches are formed (balanced by condition, cell type, etc.)
- **Efficient Data Loading**: Memory-efficient loading with caching and parallel processing
- **Comprehensive Processing Pipeline**: End-to-end processing from raw data to training-ready datasets
- **Validation and Quality Control**: Robust validation of data integrity and format compatibility

## Key Components

### 1. PairedDataModule (`lightning_transfer/paired_data_module.py`)

Enhanced PyTorch Lightning DataModule for handling multiple paired samples.

**Features:**
- **Sample metadata management** with flexible manifest files
- **Condition-balanced sampling** for fair representation
- **Memory-efficient caching** with LRU eviction
- **Configurable batch composition** with custom samplers
- **Comprehensive data validation** and error handling

**Example Usage:**
```python
from lightning_transfer.paired_data_module import PairedDataModule

# Create data module from sample manifest
data_module = PairedDataModule(
    manifest_file="samples_manifest.yaml",
    batch_size=8,
    num_workers=4,
    balance_conditions=True,
    use_condition_sampler=True,
    max_samples_per_condition=100,
    cache_size=50
)

# Setup for training
data_module.setup("fit")

# Get dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
```

### 2. Batch Data Processor (`scripts/batch_data_processor.py`)

Parallel processing system for multiple samples.

**Features:**
- **Parallel sample processing** with configurable worker count
- **Progress tracking and error handling** with detailed logging
- **Configurable pipeline steps** (download, ATAC, RAMPAGE, peaks, motifs)
- **Automatic output organization** with structured directories
- **Training manifest generation** from processing results

**Example Usage:**
```bash
# Process multiple samples from manifest
python scripts/batch_data_processor.py \
    --config base_config.yaml \
    --manifest samples_manifest.yaml \
    --output /path/to/output \
    --workers 8 \
    --steps atac,rampage,peaks

# Create training manifest from results
python scripts/batch_data_processor.py \
    --create-training-manifest processing_results.json \
    --output training_manifest.yaml
```

### 3. Data Converter (`scripts/data_converter.py`)

Converts various genomic data formats to EpiBERT-compatible HDF5 format.

**Features:**
- **Multiple input format support** (BED, bedGraph, FASTA, peaks)
- **Configurable genomic regions** with automatic or custom region definition
- **Parallel batch conversion** for multiple samples
- **Data validation and quality control** with comprehensive error handling
- **Optimized HDF5 output** with compression and metadata

**Example Usage:**
```bash
# Convert single sample
python scripts/data_converter.py \
    --sample-id sample1 \
    --atac-bed sample1_atac.bedgraph \
    --rampage-bed sample1_rampage.bedgraph \
    --peaks-bed sample1_peaks.bed \
    --genome-fasta hg38.fa \
    --genome-sizes hg38.chrom.sizes \
    --output sample1_epibert.h5

# Batch convert from manifest
python scripts/data_converter.py \
    --batch-manifest raw_data_manifest.yaml \
    --output-dir converted_data \
    --genome-fasta hg38.fa \
    --genome-sizes hg38.chrom.sizes \
    --workers 4
```

### 4. Enhanced Original Data Module (`lightning_transfer/data_module.py`)

Updated to integrate seamlessly with the paired data module.

**New Features:**
- **Automatic module selection** based on input type
- **Backward compatibility** with existing single-sample workflows
- **Unified factory function** for both single and multi-sample datasets

**Example Usage:**
```python
from lightning_transfer.data_module import create_data_module

# Single-sample dataset (backward compatible)
data_module = create_data_module(
    data_dir="/path/to/single/sample/data",
    batch_size=8
)

# Multi-sample paired dataset
data_module = create_data_module(
    manifest_file="samples_manifest.yaml",
    batch_size=8,
    use_paired_dataset=True,
    balance_conditions=True
)
```

## Sample Manifest Format

The sample manifest defines metadata for multiple paired samples. Supports YAML, CSV, and JSON formats.

### YAML Format Example
```yaml
samples:
  - sample_id: "sample1_K562_ctrl"
    condition: "control"
    cell_type: "K562"
    atac_file: "/path/to/sample1_atac.h5"
    rampage_file: "/path/to/sample1_rampage.h5"
    batch: "train"
    replicate: 1
    
  - sample_id: "sample2_K562_treat"
    condition: "treatment"
    cell_type: "K562"
    atac_file: "/path/to/sample2_atac.h5"
    rampage_file: "/path/to/sample2_rampage.h5"
    batch: "train"
    replicate: 1
    
  - sample_id: "sample3_GM12878_ctrl"
    condition: "control"
    cell_type: "GM12878"
    atac_file: "/path/to/sample3_atac.h5"
    rampage_file: "/path/to/sample3_rampage.h5"
    batch: "val"
    replicate: 1
```

### CSV Format Example
```csv
sample_id,condition,cell_type,atac_file,rampage_file,batch,replicate
sample1_K562_ctrl,control,K562,/path/to/sample1_atac.h5,/path/to/sample1_rampage.h5,train,1
sample2_K562_treat,treatment,K562,/path/to/sample2_atac.h5,/path/to/sample2_rampage.h5,train,1
sample3_GM12878_ctrl,control,GM12878,/path/to/sample3_atac.h5,/path/to/sample3_rampage.h5,val,1
```

## Complete Workflow Examples

### Example 1: Processing Multiple Samples from Raw Data

```bash
# 1. Create sample manifest for raw data
cat > raw_samples_manifest.yaml << EOF
samples:
  - sample_id: "exp1_K562_ctrl"
    condition: "control"
    cell_type: "K562"
    atac_bam: "/data/exp1_K562_ctrl_atac.bam"
    rampage_bam: "/data/exp1_K562_ctrl_rampage.bam"
    
  - sample_id: "exp1_K562_treat"
    condition: "treatment"
    cell_type: "K562"
    atac_bam: "/data/exp1_K562_treat_atac.bam"
    rampage_bam: "/data/exp1_K562_treat_rampage.bam"
EOF

# 2. Create base configuration
cat > base_config.yaml << EOF
genome_fasta: "/ref/hg38.fa"
genome_sizes: "/ref/hg38.chrom.sizes"
blacklist: "/ref/hg38.blacklist.bed"
motif_database: "/ref/motifs.meme"
background_peaks: "/ref/background.bed"
threads: 4
rampage_scale_factor: 2.0
EOF

# 3. Process all samples in parallel
python scripts/batch_data_processor.py \
    --config base_config.yaml \
    --manifest raw_samples_manifest.yaml \
    --output processed_data \
    --workers 4 \
    --steps all

# 4. Convert processed data to EpiBERT format
python scripts/data_converter.py \
    --batch-manifest processed_data/processing_results.json \
    --output-dir epibert_data \
    --genome-fasta /ref/hg38.fa \
    --genome-sizes /ref/hg38.chrom.sizes \
    --workers 4

# 5. Train model with paired dataset
python lightning_transfer/train_lightning.py \
    --manifest_file epibert_data/converted_samples_manifest.yaml \
    --model_type pretraining \
    --batch_size 8 \
    --balance_conditions \
    --use_condition_sampler \
    --max_epochs 100
```

### Example 2: Working with Pre-processed Data

```python
from lightning_transfer.paired_data_module import PairedDataModule, create_sample_manifest
from lightning_transfer.epibert_lightning import EpiBERTLightning
import pytorch_lightning as pl

# Create sample manifest from existing processed files
samples = [
    {
        'sample_id': 'sample1',
        'condition': 'control',
        'cell_type': 'K562',
        'atac_file': '/data/sample1_atac.h5',
        'rampage_file': '/data/sample1_rampage.h5',
        'batch': 'train',
        'replicate': 1
    },
    {
        'sample_id': 'sample2',
        'condition': 'treatment',
        'cell_type': 'K562',
        'atac_file': '/data/sample2_atac.h5',
        'rampage_file': '/data/sample2_rampage.h5',
        'batch': 'train',
        'replicate': 1
    }
]

# Save manifest
create_sample_manifest(samples, "training_manifest.yaml")

# Create data module
data_module = PairedDataModule(
    manifest_file="training_manifest.yaml",
    batch_size=4,
    num_workers=2,
    balance_conditions=True,
    use_condition_sampler=True
)

# Create model
model = EpiBERTLightning(
    model_type='pretraining',
    learning_rate=1e-4
)

# Train
trainer = pl.Trainer(
    max_epochs=10,
    gpus=1 if torch.cuda.is_available() else 0,
    precision='bf16'
)

trainer.fit(model, data_module)
```

### Example 3: Condition-Balanced Training

```python
from lightning_transfer.paired_data_module import PairedDataModule, ConditionBalancedSampler

# Create data module with condition balancing
data_module = PairedDataModule(
    manifest_file="samples_manifest.yaml",
    batch_size=8,
    balance_conditions=True,  # Balance during dataset creation
    use_condition_sampler=True,  # Use custom sampler for batching
    max_samples_per_condition=50,  # Limit samples per condition
    cache_size=100  # Cache 100 samples in memory
)

# Setup and inspect
data_module.setup("fit")

# Get sample information
train_dataset = data_module.train_dataset
sample_info = train_dataset.get_sample_info()

print("Sample distribution:")
print(sample_info.groupby(['condition', 'cell_type']).size())

# Train with balanced batches
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, data_module)
```

## Configuration Options

### PairedDataModule Configuration

```python
PairedDataModule(
    manifest_file="samples.yaml",        # Sample manifest file
    batch_size=8,                        # Batch size
    num_workers=4,                       # Data loading workers
    input_length=524288,                 # Input sequence length
    output_length=4096,                  # Output profile length
    atac_mask_dropout=0.15,             # Masking fraction for training
    balance_conditions=True,             # Balance samples across conditions
    use_condition_sampler=True,          # Use condition-balanced sampler
    max_samples_per_condition=None,      # Max samples per condition
    cache_size=100                       # Number of samples to cache
)
```

### BatchDataProcessor Configuration

```yaml
# base_config.yaml
genome_fasta: "/path/to/genome.fa"
genome_sizes: "/path/to/genome.sizes"
blacklist: "/path/to/blacklist.bed"
motif_database: "/path/to/motifs.meme"
background_peaks: "/path/to/background.bed"
threads: 4
rampage_scale_factor: 2.0
```

### DataConverter Configuration

```python
DataConverter(
    genome_fasta="/path/to/genome.fa",   # Reference genome
    genome_sizes="/path/to/genome.sizes", # Chromosome sizes
    sequence_length=524288,              # Input sequence length
    output_length=4096,                  # Output profile length
    resolution=128                       # Profile resolution (bp per bin)
)
```

## Validation and Quality Control

### Manifest Validation

```python
from lightning_transfer.paired_data_module import validate_manifest

# Validate sample manifest
result = validate_manifest("samples_manifest.yaml")

print(f"Valid: {result['valid']}")
print(f"Number of samples: {result['num_samples']}")
print(f"Conditions: {result['conditions']}")
print(f"Missing files: {result['missing_files']}")
print(f"Errors: {result['errors']}")
```

### Data Quality Checks

```python
from lightning_transfer.data_module import validate_data_format

# Validate individual data files
for sample in samples:
    info = validate_data_format(sample['atac_file'])
    
    if info['valid']:
        print(f"✓ {sample['sample_id']}: {info['n_samples']} samples")
    else:
        print(f"✗ {sample['sample_id']}: {info['error']}")
```

## Testing and Validation

Run comprehensive tests to validate all functionality:

```bash
# Run all tests
python scripts/test_enhanced_data_handling.py

# Test individual components
python -c "
from scripts.test_enhanced_data_handling import test_paired_data_module
test_paired_data_module()
"
```

## Performance Optimization

### Memory Management
- **Caching**: Adjust `cache_size` based on available memory
- **Batch Size**: Optimize based on GPU memory and sample complexity
- **Workers**: Use 2-4x CPU cores for `num_workers`

### I/O Optimization
- **HDF5 Compression**: Uses gzip compression for storage efficiency
- **Parallel Loading**: Multiple workers for concurrent data loading
- **Memory Mapping**: Efficient access to large datasets

### Training Efficiency
- **Condition Balancing**: Ensures fair representation of all conditions
- **Data Augmentation**: Configurable augmentations for better generalization
- **Mixed Precision**: Support for bf16/fp16 training

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `batch_size`, `cache_size`, or `num_workers`
2. **File Not Found**: Check file paths in manifest and ensure all files exist
3. **Shape Mismatches**: Validate data formats and sequence/profile lengths
4. **Slow Loading**: Increase `num_workers` and check disk I/O performance

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed logging for debugging
data_module = PairedDataModule(..., cache_size=1)  # Disable caching for debugging
```

This enhanced data handling system provides a robust, scalable solution for training EpiBERT with multiple paired ATAC-seq and RAMPAGE-seq samples, with comprehensive support for batch processing, data validation, and performance optimization.