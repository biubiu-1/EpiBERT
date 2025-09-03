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
├── write_TF_records/           # Create training datasets
├── run_pipeline.sh             # Master pipeline script
└── example_config.yaml         # Example configuration file
```

## Quick Start

For most users, the easiest way to process data is using the master pipeline script:

```bash
# 1. Create a configuration file (see example_config.yaml)
cp example_config.yaml my_config.yaml
# Edit my_config.yaml with your file paths

# 2. Run the complete pipeline
./run_pipeline.sh -c my_config.yaml

# 3. Or run specific steps
./run_pipeline.sh -c my_config.yaml -s atac    # ATAC-seq only
./run_pipeline.sh -c my_config.yaml -s peaks   # Peak calling only
```

## Individual Scripts (General Use)

All workflows have been converted from WDL to standalone bash scripts for broader accessibility:

## 1. Data Download (`downloading_utilities/`)

### ENCODE BAM Download
```bash
./downloading_utilities/encode_bam_download.sh \
  -a ENCFF123ABC,ENCFF456DEF \
  -s sample1 \
  -t ATAC-seq \
  -o output_dir
```

### SRA FASTQ Download
```bash
./downloading_utilities/sra_fastq_download.sh \
  -s SRR123456,SRR789012 \
  -o output_dir \
  -j 4
```

## 2. Signal Track Creation (`create_signal_tracks/`)

### ATAC-seq BAM to Fragments
```bash
./create_signal_tracks/bam_to_bed_ATAC.sh \
  -i input.bam \
  -s sample1 \
  -o output_dir
```

### Fragments to BedGraph Signal
```bash
./create_signal_tracks/fragments_to_bed_scores.sh \
  -i fragments.bed.gz \
  -s sample1 \
  -g genome.chrom.sizes \
  -o output_dir
```

### RAMPAGE-seq BAM to TSS Signal
```bash
./create_signal_tracks/bam_to_bed_RAMPAGE_5prime.sh \
  -i rampage.bam \
  -s sample1 \
  -g genome.chrom.sizes \
  -o output_dir
```

## 3. Alignment and Peak Calling (`alignment_and_peak_call/`)

### MACS2 Peak Calling with Pseudo-replicates
```bash
./alignment_and_peak_call/macs2_peak_call_from_fragment.sh \
  -i fragments.bed.gz \
  -s sample1 \
  -b blacklist.bed \
  -o output_dir
```

## 4. Motif Enrichment (`motif_enrichment/`)

### MEME SEA Motif Analysis
```bash
./motif_enrichment/meme_run_sea.sh \
  -p peaks.bed.gz \
  -m motifs.meme \
  -g genome.fa \
  -b background.bed \
  -n sample1 \
  -o output_dir
```

## Legacy WDL Workflows

The original WDL workflows are still available for cloud-based execution:

### ATAC-seq Processing
- **`bam_to_bed_ATAC.wdl`**: Convert ATAC-seq BAM to fragments file
- **`fragments_to_bed_scores.wdl`**: Convert fragments to scaled bedGraph signal
- **`liftover_fragments.wdl`**: Convert hg19 fragments to hg38

### RAMPAGE-seq Processing  
- **`bam_to_bed_RAMPAGE_5PRIME.wdl`**: Convert RAMPAGE BAM to 5' TSS positions
- Creates scaled bedGraph signal files for transcription start sites

### Motif Enrichment
- **`meme_run_sea.wdl`**: Compute motif enrichments using MEME SEA
- Uses consensus motifs from Vierstra et al. 2020

## Dependencies

The general-use scripts require the following tools to be installed:

### Core Dependencies
- **samtools** (≥1.10): BAM file processing
- **bedtools** (≥2.27): Genomic interval operations
- **awk, sort, gzip**: Standard Unix tools

### Specific Dependencies
- **MACS2**: Peak calling (`pip install MACS2`)
- **MEME Suite**: Motif analysis (`meme-suite.org`)
- **SRA Toolkit**: FASTQ downloads (`ncbi.nlm.nih.gov/sra/docs/toolkitsoft`)
- **pigz**: Fast compression (optional, for SRA downloads)
- **wget**: File downloads
- **python3**: Required for peak merging scripts

### Docker Alternative
For exact WDL environment compatibility, the scripts can be run using the same Docker images:
- `njaved/samtools_bedtools`: General processing
- `fooliu/macs2`: Peak calling
- `memesuite/memesuite:5.4.1`: Motif analysis
- `njaved/sra-tools-ubuntu:latest`: SRA downloads

## Output Formats

### For TensorFlow (TFRecord)
The processed data can be converted to TensorFlow TFRecord format using the `write_TF_records/` utilities for training the original EpiBERT models.

### For PyTorch Lightning
The processed data can be converted to HDF5 format for use with the Lightning implementation. See `lightning_transfer/` for PyTorch-compatible data loading.

## Reference Files

You will need the following reference files:
- **Genome FASTA**: Reference genome sequence (e.g., hg38.fa)
- **Chromosome sizes**: Tab-delimited chr\tsize file (e.g., hg38.chrom.sizes)
- **Blacklist regions**: ENCODE blacklist BED file for peak filtering
- **Motif database**: MEME format motif database (e.g., JASPAR, HOCOMOCO)
- **Background peaks**: Accessible genomic regions for motif enrichment

These can be downloaded from:
- ENCODE: `encode-public.s3.amazonaws.com`
- UCSC Genome Browser: `hgdownload.cse.ucsc.edu`
- JASPAR: `jaspar.genereg.net`
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