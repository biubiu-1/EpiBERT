#!/bin/bash

# EpiBERT Data Processing Pipeline
# Master script that runs the complete data processing pipeline
# Combines all individual scripts into a cohesive workflow

set -euo pipefail

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -c CONFIG_FILE [OPTIONS]

Run the complete EpiBERT data processing pipeline.

REQUIRED PARAMETERS:
  -c CONFIG_FILE      Configuration file with pipeline parameters

OPTIONAL PARAMETERS:
  -s STEP             Run specific step only (download|atac|rampage|peaks|motifs|all)
  -o OUTPUT_DIR       Output directory (default: ./epibert_data)
  -j THREADS          Number of threads (default: 4)
  -h                  Show this help message

EXAMPLES:
  $0 -c config.yaml
  $0 -c config.yaml -s atac -o /path/to/output
  $0 -c config.yaml -j 8

CONFIG FILE FORMAT (YAML):
  sample_id: "sample_name"
  
  # Data download
  encode_accessions: "ENCFF123ABC,ENCFF456DEF"  # Optional
  sra_accessions: "SRR123456,SRR789012"        # Optional
  
  # ATAC-seq processing
  atac_bam: "/path/to/atac.bam"                 # Required for ATAC processing
  
  # RAMPAGE-seq processing
  rampage_bam: "/path/to/rampage.bam"           # Required for RAMPAGE processing
  rampage_scale_factor: 2.0                    # Optional (default: 2.0)
  
  # Peak calling
  blacklist: "/path/to/blacklist.bed"          # Required for peak calling
  
  # Motif enrichment
  motif_database: "/path/to/motifs.meme"       # Required for motif analysis
  background_peaks: "/path/to/background.bed"  # Required for motif analysis
  
  # Reference files
  genome_fasta: "/path/to/genome.fa"           # Required
  genome_sizes: "/path/to/genome.sizes"        # Required

PIPELINE STEPS:
  1. download: Download data from ENCODE/SRA
  2. atac: Process ATAC-seq BAM to fragments and bedGraph
  3. rampage: Process RAMPAGE-seq BAM to TSS bedGraph
  4. peaks: Call peaks from ATAC-seq fragments
  5. motifs: Perform motif enrichment analysis
  6. all: Run complete pipeline

OUTPUT STRUCTURE:
  OUTPUT_DIR/
  ├── download/          # Downloaded files
  ├── atac/             # ATAC-seq processing results
  ├── rampage/          # RAMPAGE-seq processing results
  ├── peaks/            # Peak calling results
  ├── motifs/           # Motif enrichment results
  └── logs/             # Processing logs

DEPENDENCIES:
  - All individual pipeline scripts
  - yq (for YAML parsing) or manual config parsing
  - Standard Unix tools (awk, sed, grep)
EOF
}

# Default values
CONFIG_FILE=""
STEP="all"
OUTPUT_DIR="./epibert_data"
THREADS=4

# Parse command line arguments
while getopts "c:s:o:j:h" opt; do
    case $opt in
        c) CONFIG_FILE="$OPTARG" ;;
        s) STEP="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        j) THREADS="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: Missing required parameter -c CONFIG_FILE" >&2
    usage
    exit 1
fi

# Check config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# Validate step parameter
valid_steps="download atac rampage peaks motifs all"
if [[ ! " $valid_steps " =~ " $STEP " ]]; then
    echo "Error: Invalid step '$STEP'. Must be one of: $valid_steps" >&2
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for script dependencies
check_script() {
    local script_path="$1"
    if [[ ! -f "$script_path" ]]; then
        echo "Error: Required script not found: $script_path" >&2
        exit 1
    fi
    if [[ ! -x "$script_path" ]]; then
        echo "Error: Script not executable: $script_path" >&2
        exit 1
    fi
}

# Parse config file (simple key: value format)
parse_config() {
    local key="$1"
    local default="${2:-}"
    
    # Simple YAML/config parser
    local value
    value=$(grep "^$key:" "$CONFIG_FILE" | head -1 | sed "s/^$key://g" | sed 's/^[[:space:]]*//g' | sed 's/[[:space:]]*$//g' | sed 's/^["'\'']//' | sed 's/["'\'']$//')
    
    if [[ -z "$value" && -n "$default" ]]; then
        value="$default"
    fi
    
    echo "$value"
}

echo "=== EpiBERT Data Processing Pipeline ==="
echo "Config file: $CONFIG_FILE"
echo "Step: $STEP"
echo "Output directory: $OUTPUT_DIR"
echo "Threads: $THREADS"
echo "========================================="

# Create output directory structure
mkdir -p "$OUTPUT_DIR"/{download,atac,rampage,peaks,motifs,logs}

# Parse configuration
SAMPLE_ID=$(parse_config "sample_id")
ENCODE_ACCESSIONS=$(parse_config "encode_accessions")
SRA_ACCESSIONS=$(parse_config "sra_accessions")
ATAC_BAM=$(parse_config "atac_bam")
RAMPAGE_BAM=$(parse_config "rampage_bam")
RAMPAGE_SCALE_FACTOR=$(parse_config "rampage_scale_factor" "2.0")
BLACKLIST=$(parse_config "blacklist")
MOTIF_DATABASE=$(parse_config "motif_database")
BACKGROUND_PEAKS=$(parse_config "background_peaks")
GENOME_FASTA=$(parse_config "genome_fasta")
GENOME_SIZES=$(parse_config "genome_sizes")

# Validate required configuration
if [[ -z "$SAMPLE_ID" ]]; then
    echo "Error: sample_id must be specified in config file" >&2
    exit 1
fi

echo "Sample ID: $SAMPLE_ID"

# Function to run a step with logging
run_step() {
    local step_name="$1"
    local script_path="$2"
    shift 2
    local args=("$@")
    
    local log_file="$OUTPUT_DIR/logs/${step_name}.log"
    local start_time=$(date)
    
    echo "Running $step_name..."
    echo "Command: $script_path ${args[*]}"
    echo "Log file: $log_file"
    
    # Run script with logging
    if "$script_path" "${args[@]}" > "$log_file" 2>&1; then
        local end_time=$(date)
        echo "✓ $step_name completed successfully"
        echo "Started: $start_time" >> "$log_file"
        echo "Finished: $end_time" >> "$log_file"
    else
        local end_time=$(date)
        echo "✗ $step_name failed" >&2
        echo "Started: $start_time" >> "$log_file"
        echo "Failed: $end_time" >> "$log_file"
        echo "Check log file: $log_file" >&2
        exit 1
    fi
}

# Download step
run_download() {
    echo "=== Data Download ==="
    
    if [[ -n "$ENCODE_ACCESSIONS" ]]; then
        check_script "$SCRIPT_DIR/downloading_utilities/encode_bam_download.sh"
        run_step "encode_download" \
            "$SCRIPT_DIR/downloading_utilities/encode_bam_download.sh" \
            -a "$ENCODE_ACCESSIONS" \
            -s "$SAMPLE_ID" \
            -t "ATAC-seq" \
            -o "$OUTPUT_DIR/download" \
            -j "$THREADS"
    fi
    
    if [[ -n "$SRA_ACCESSIONS" ]]; then
        check_script "$SCRIPT_DIR/downloading_utilities/sra_fastq_download.sh"
        run_step "sra_download" \
            "$SCRIPT_DIR/downloading_utilities/sra_fastq_download.sh" \
            -s "$SRA_ACCESSIONS" \
            -o "$OUTPUT_DIR/download" \
            -j "$THREADS"
    fi
}

# ATAC-seq processing
run_atac() {
    echo "=== ATAC-seq Processing ==="
    
    if [[ -z "$ATAC_BAM" ]]; then
        echo "Error: atac_bam must be specified for ATAC processing" >&2
        exit 1
    fi
    
    # BAM to fragments
    check_script "$SCRIPT_DIR/create_signal_tracks/bam_to_bed_ATAC.sh"
    run_step "atac_fragments" \
        "$SCRIPT_DIR/create_signal_tracks/bam_to_bed_ATAC.sh" \
        -i "$ATAC_BAM" \
        -s "$SAMPLE_ID" \
        -o "$OUTPUT_DIR/atac" \
        -j "$THREADS"
    
    # Fragments to bedGraph
    if [[ -n "$GENOME_SIZES" ]]; then
        check_script "$SCRIPT_DIR/create_signal_tracks/fragments_to_bed_scores.sh"
        run_step "atac_bedgraph" \
            "$SCRIPT_DIR/create_signal_tracks/fragments_to_bed_scores.sh" \
            -i "$OUTPUT_DIR/atac/${SAMPLE_ID}.bed.gz" \
            -s "$SAMPLE_ID" \
            -g "$GENOME_SIZES" \
            -o "$OUTPUT_DIR/atac"
    fi
}

# RAMPAGE-seq processing
run_rampage() {
    echo "=== RAMPAGE-seq Processing ==="
    
    if [[ -z "$RAMPAGE_BAM" ]]; then
        echo "Error: rampage_bam must be specified for RAMPAGE processing" >&2
        exit 1
    fi
    
    if [[ -z "$GENOME_SIZES" ]]; then
        echo "Error: genome_sizes must be specified for RAMPAGE processing" >&2
        exit 1
    fi
    
    check_script "$SCRIPT_DIR/create_signal_tracks/bam_to_bed_RAMPAGE_5prime.sh"
    run_step "rampage_processing" \
        "$SCRIPT_DIR/create_signal_tracks/bam_to_bed_RAMPAGE_5prime.sh" \
        -i "$RAMPAGE_BAM" \
        -s "$SAMPLE_ID" \
        -g "$GENOME_SIZES" \
        -o "$OUTPUT_DIR/rampage" \
        -j "$THREADS" \
        -p "$RAMPAGE_SCALE_FACTOR"
}

# Peak calling
run_peaks() {
    echo "=== Peak Calling ==="
    
    local fragments_file="$OUTPUT_DIR/atac/${SAMPLE_ID}.bed.gz"
    if [[ ! -f "$fragments_file" ]]; then
        echo "Error: ATAC fragments file not found: $fragments_file" >&2
        echo "Run ATAC processing first" >&2
        exit 1
    fi
    
    if [[ -z "$BLACKLIST" ]]; then
        echo "Error: blacklist must be specified for peak calling" >&2
        exit 1
    fi
    
    check_script "$SCRIPT_DIR/alignment_and_peak_call/macs2_peak_call_from_fragment.sh"
    run_step "peak_calling" \
        "$SCRIPT_DIR/alignment_and_peak_call/macs2_peak_call_from_fragment.sh" \
        -i "$fragments_file" \
        -s "$SAMPLE_ID" \
        -b "$BLACKLIST" \
        -o "$OUTPUT_DIR/peaks" \
        -j "$THREADS"
}

# Motif enrichment
run_motifs() {
    echo "=== Motif Enrichment ==="
    
    local peaks_file="$OUTPUT_DIR/peaks/${SAMPLE_ID}.pr.peaks.bed.gz"
    if [[ ! -f "$peaks_file" ]]; then
        echo "Error: Peaks file not found: $peaks_file" >&2
        echo "Run peak calling first" >&2
        exit 1
    fi
    
    if [[ -z "$MOTIF_DATABASE" || -z "$BACKGROUND_PEAKS" || -z "$GENOME_FASTA" ]]; then
        echo "Error: motif_database, background_peaks, and genome_fasta must be specified for motif analysis" >&2
        exit 1
    fi
    
    check_script "$SCRIPT_DIR/motif_enrichment/meme_run_sea.sh"
    run_step "motif_enrichment" \
        "$SCRIPT_DIR/motif_enrichment/meme_run_sea.sh" \
        -p "$peaks_file" \
        -m "$MOTIF_DATABASE" \
        -g "$GENOME_FASTA" \
        -b "$BACKGROUND_PEAKS" \
        -o "$OUTPUT_DIR/motifs" \
        -n "$SAMPLE_ID" \
        -j "$THREADS"
}

# Run specified step(s)
case "$STEP" in
    "download")
        run_download
        ;;
    "atac")
        run_atac
        ;;
    "rampage")
        run_rampage
        ;;
    "peaks")
        run_peaks
        ;;
    "motifs")
        run_motifs
        ;;
    "all")
        [[ -n "$ENCODE_ACCESSIONS" || -n "$SRA_ACCESSIONS" ]] && run_download
        [[ -n "$ATAC_BAM" ]] && run_atac
        [[ -n "$RAMPAGE_BAM" ]] && run_rampage
        [[ -n "$ATAC_BAM" && -n "$BLACKLIST" ]] && run_peaks
        [[ -n "$MOTIF_DATABASE" && -n "$BACKGROUND_PEAKS" && -n "$GENOME_FASTA" ]] && run_motifs
        ;;
esac

echo "=== Pipeline Summary ==="
echo "Output directory: $OUTPUT_DIR"
echo "Generated files:"
find "$OUTPUT_DIR" -type f -name "${SAMPLE_ID}*" | sort | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  $file ($size)"
done

echo "=== EpiBERT Data Processing Pipeline Complete ==="