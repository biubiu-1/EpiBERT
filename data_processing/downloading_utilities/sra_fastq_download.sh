#!/bin/bash

# SRA FASTQ Download Script
# Converts the sra_fetch_fasterqdump.wdl workflow to a standalone bash script
# Downloads FASTQ files from SRA using fasterq-dump

set -euo pipefail

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -s SRR_IDS [OPTIONS]

Download FASTQ files from SRA using fasterq-dump.

REQUIRED PARAMETERS:
  -s SRR_IDS       Comma-separated list of SRR accession IDs

OPTIONAL PARAMETERS:
  -o OUTPUT_DIR    Output directory (default: current directory)
  -j THREADS       Number of threads for fasterq-dump (default: 1)
  -t TEMP_DIR      Temporary directory for SRA tools (default: /tmp)
  -c COMPRESS      Compress output files with pigz (default: true)
  -h               Show this help message

EXAMPLES:
  $0 -s SRR123456,SRR789012
  $0 -s SRR345678 -o /path/to/output -j 4 -t /scratch

DEPENDENCIES:
  - SRA Toolkit (prefetch, fasterq-dump)
  - pigz (for compression)
  - Docker (optional, for using exact WDL environment)

OUTPUT:
  - [SRR_ID]_1.fastq.gz: Read 1 (forward reads)
  - [SRR_ID]_2.fastq.gz: Read 2 (reverse reads, if paired-end)

NOTES:
  - For single-end data, only _1.fastq.gz will be created
  - The script automatically detects paired vs single-end data
EOF
}

# Default values
OUTPUT_DIR="."
THREADS=1
TEMP_DIR="/tmp"
COMPRESS=true
SRR_IDS=""

# Parse command line arguments
while getopts "s:o:j:t:c:h" opt; do
    case $opt in
        s) SRR_IDS="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        j) THREADS="$OPTARG" ;;
        t) TEMP_DIR="$OPTARG" ;;
        c) COMPRESS="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$SRR_IDS" ]]; then
    echo "Error: Missing required parameter -s SRR_IDS" >&2
    usage
    exit 1
fi

# Check dependencies
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed or not in PATH" >&2
        echo "Please install SRA Toolkit and pigz" >&2
        exit 1
    fi
}

check_dependency "prefetch"
check_dependency "fasterq-dump"
if [[ "$COMPRESS" == "true" ]]; then
    check_dependency "pigz"
fi

# Create output and temp directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# Split SRR IDs into array
IFS=',' read -ra SRR_ARRAY <<< "$SRR_IDS"

echo "=== SRA FASTQ Download Started ==="
echo "SRR IDs: $SRR_IDS"
echo "Output directory: $OUTPUT_DIR"
echo "Temp directory: $TEMP_DIR"
echo "Threads: $THREADS"
echo "Compress: $COMPRESS"
echo "=================================="

# Process each SRR ID
for srr_id in "${SRR_ARRAY[@]}"; do
    echo "Processing: $srr_id"
    
    # Change to output directory for this sample
    cd "$OUTPUT_DIR"
    
    # Prefetch the SRA file
    echo "  Prefetching $srr_id..."
    prefetch "$srr_id" --output-directory "$TEMP_DIR"
    
    # Extract FASTQ files
    echo "  Extracting FASTQ files..."
    fasterq-dump --split-3 "$TEMP_DIR/$srr_id/$srr_id.sra" \
                 --outdir "$OUTPUT_DIR" \
                 --temp "$TEMP_DIR" \
                 --threads "$THREADS" \
                 --skip-technical
    
    # Handle single-end vs paired-end naming
    if [[ ! -f "${srr_id}_1.fastq" && -f "${srr_id}.fastq" ]]; then
        echo "  Single-end data detected, renaming..."
        mv "${srr_id}.fastq" "${srr_id}_1.fastq"
    fi
    
    # Compress FASTQ files
    if [[ "$COMPRESS" == "true" ]]; then
        echo "  Compressing FASTQ files..."
        for fastq_file in "${srr_id}"*.fastq; do
            if [[ -f "$fastq_file" ]]; then
                pigz "$fastq_file"
            fi
        done
    fi
    
    # Clean up prefetched SRA file
    rm -rf "$TEMP_DIR/$srr_id"
    
    # Verify output files
    if [[ -f "${srr_id}_1.fastq.gz" || -f "${srr_id}_1.fastq" ]]; then
        echo "  ✓ Successfully processed $srr_id"
        
        # Report file info
        if [[ -f "${srr_id}_2.fastq.gz" || -f "${srr_id}_2.fastq" ]]; then
            echo "    Paired-end data: ${srr_id}_1.fastq.gz, ${srr_id}_2.fastq.gz"
        else
            echo "    Single-end data: ${srr_id}_1.fastq.gz"
        fi
    else
        echo "  ✗ Error processing $srr_id" >&2
        exit 1
    fi
done

echo "=== SRA FASTQ Download Complete ==="
echo "Downloaded files:"
ls -lh "$OUTPUT_DIR"/*_1.fastq* "$OUTPUT_DIR"/*_2.fastq* 2>/dev/null || true

echo "=== SRA FASTQ Download Finished ==="