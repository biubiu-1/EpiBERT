# EpiBERT Data Preparation Tools

This document describes the available data preparation tools for building training and fine-tuning datasets for EpiBERT.

## Overview

The `data_processing/` directory contains all tools necessary for preprocessing genomic data into the format required by EpiBERT:

```
data_processing/
├── alignment_and_peak_call/     # ATAC-seq alignment and peak calling
├── create_signal_tracks/        # Convert BAM files to signal tracks  
├── downloading_utilities/       # Download utilities from GEO/ENCODE
├── motif_enrichment/           # Motif enrichment analysis
└── write_TF_records/           # Create training datasets
```

## 1. Data Download (`downloading_utilities/`)

Tools for downloading raw genomic data:
- Download FASTQ files from GEO
- Download processed BAM files from ENCODE
- Batch download utilities

## 2. Alignment and Peak Calling (`alignment_and_peak_call/`)

ATAC-seq processing pipeline:
- Align FASTQ files to reference genome
- Peak calling using MACS2
- Quality control and filtering
- Fragment file generation

## 3. Signal Track Creation (`create_signal_tracks/`)

Convert aligned data to signal tracks:

### ATAC-seq Processing
- **`bam_to_bed_ATAC.wdl`**: Convert ATAC-seq BAM to fragments file
- **`fragments_to_bed_scores.wdl`**: Convert fragments to scaled bedGraph signal
- **`liftover_fragments.wdl`**: Convert hg19 fragments to hg38

### RAMPAGE-seq Processing  
- **`bam_to_bed_RAMPAGE_5PRIME.wdl`**: Convert RAMPAGE BAM to 5' TSS positions
- Creates scaled bedGraph signal files for transcription start sites

## 4. Motif Enrichment (`motif_enrichment/`)

Transcription factor motif analysis:
- Compute motif enrichments using MEME SEA
- Uses consensus motifs from Vierstra et al. 2020
- Generates motif activity scores for model input

## 5. TFRecord Generation (`write_TF_records/`)

Core dataset creation tools:

### Main Processing Scripts
- **`data_processing_seq_to_atac_globalacc.py`**: Create pretraining dataset
- **`data_processing_seq_to_atac_rampage_globalacc.py`**: Create fine-tuning dataset  
- **`data_process_seq_to_atac_rampage_globalacc_testTSS.py`**: Create test datasets

### Data Splits
The `sequence_splits/` directory contains predefined train/validation/test splits:
- Training sequences
- Validation sequences (TSS-centered)
- Test sequences (TSS and non-TSS centered)

### Utilities
- **`utils.py`**: Common data processing functions
- **`cython_fxn.pyx`**: Optimized Cython functions for data processing

## Quick Start Guide

### 1. Download Raw Data
```bash
# Example: Download ATAC-seq data from ENCODE
cd data_processing/downloading_utilities/
python download_encode_atac.py --accession ENCSR356KRQ
```

### 2. Process ATAC-seq Data
```bash
# Align and call peaks
cd data_processing/alignment_and_peak_call/
./process_atac_sample.sh sample_R1.fastq.gz sample_R2.fastq.gz

# Create signal tracks
cd ../create_signal_tracks/
cromwell run bam_to_bed_ATAC.wdl --inputs atac_inputs.json
cromwell run fragments_to_bed_scores.wdl --inputs signal_inputs.json
```

### 3. Generate Motif Enrichments
```bash
cd data_processing/motif_enrichment/
python compute_motif_enrichment.py --peaks peaks.bed --genome hg38.fa
```

### 4. Create Training Dataset
```bash
cd data_processing/write_TF_records/
python data_processing_seq_to_atac_globalacc.py \
    --sequence_bed training_sequences.bed \
    --atac_signal_dir ./atac_signals/ \
    --motif_scores motif_enrichments.tsv \
    --output_dir ./training_tfrecords/
```

## Data Format Requirements

### Input Files Required:
1. **Genomic sequences**: BED format with sequence coordinates
2. **ATAC-seq signals**: BigWig or bedGraph format, normalized
3. **RAMPAGE-seq signals** (for fine-tuning): BigWig format, 5' TSS signals  
4. **Motif enrichments**: TSV with motif activity scores per region
5. **Reference genome**: FASTA format (hg38 recommended)

### Output Format:
- **TensorFlow**: TFRecord format (original implementation)
- **PyTorch Lightning**: HDF5 or PyTorch tensor format (for Lightning implementation)

## Data Conversion for Lightning

For the PyTorch Lightning implementation, you can convert TFRecord files to HDF5:

```python
from lightning_transfer.data_module import convert_tfrecord_to_hdf5

convert_tfrecord_to_hdf5(
    tfrecord_pattern="./training_tfrecords/*.tfr",
    output_file="./training_data.h5"
)
```

## Configuration Files

Example configuration files are provided in each processing directory:
- Input JSON files for WDL workflows
- Parameter files for data processing scripts
- Template configuration for different data types

## Troubleshooting

### Common Issues:
1. **Memory errors**: Reduce batch size in processing scripts
2. **File format errors**: Ensure all input files use correct genome assembly
3. **Missing dependencies**: Install required bioinformatics tools (samtools, bedtools, etc.)

### Performance Tips:
- Use SSD storage for temporary files
- Parallelize processing across multiple samples
- Use appropriate memory limits for large datasets

For specific questions about data processing, see the README files in each subdirectory or consult the original EpiBERT paper for methodology details.