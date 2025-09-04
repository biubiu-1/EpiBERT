#!/bin/bash

# ENCODE BAM Download Script
# Converts the encode_bam_download.wdl workflow to a standalone bash script
# Downloads and merges BAM files from ENCODE project

set -euo pipefail

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -a ACCESSIONS -s SAMPLE_ID -t ASSAY [OPTIONS]

Download and merge BAM files from ENCODE project.

REQUIRED PARAMETERS:
  -a ACCESSIONS    Comma-separated list of ENCODE accession IDs
  -s SAMPLE_ID     Sample identifier for output file naming
  -t ASSAY         Assay type (e.g., ATAC-seq, ChIP-seq)

OPTIONAL PARAMETERS:
  -o OUTPUT_DIR    Output directory (default: current directory)
  -j THREADS       Number of threads for samtools (default: 1)
  -h               Show this help message

EXAMPLES:
  $0 -a ENCFF123ABC,ENCFF456DEF -s sample1 -t ATAC-seq
  $0 -a ENCFF789GHI -s sample2 -t ChIP-seq -o /path/to/output -j 4

DEPENDENCIES:
  - wget
  - samtools
  - Docker (optional, for using exact WDL environment)

OUTPUT:
  - [SAMPLE_ID].[ASSAY].merged.bam: Merged and sorted BAM file
EOF
}

# Default values
OUTPUT_DIR="."
THREADS=1
ACCESSIONS=""
SAMPLE_ID=""
ASSAY=""

# Parse command line arguments
while getopts "a:s:t:o:j:h" opt; do
    case $opt in
        a) ACCESSIONS="$OPTARG" ;;
        s) SAMPLE_ID="$OPTARG" ;;
        t) ASSAY="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        j) THREADS="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$ACCESSIONS" || -z "$SAMPLE_ID" || -z "$ASSAY" ]]; then
    echo "Error: Missing required parameters" >&2
    usage
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Split accessions into array
IFS=',' read -ra ACC_ARRAY <<< "$ACCESSIONS"

echo "=== ENCODE BAM Download Started ==="
echo "Accessions: $ACCESSIONS"
echo "Sample ID: $SAMPLE_ID"
echo "Assay: $ASSAY"
echo "Output directory: $OUTPUT_DIR"
echo "Threads: $THREADS"
echo "=================================="

# Download and sort each BAM file
downloaded_bams=()
for accession in "${ACC_ARRAY[@]}"; do
    echo "Downloading and sorting: $accession"
    
    # Download BAM file
    wget -O "${accession}.bam" "https://www.encodeproject.org/files/${accession}/@@download/${accession}.bam"
    
    # Sort BAM file
    samtools sort -@ "$THREADS" "${accession}.bam" -o "${accession}.sort.bam"
    
    # Clean up unsorted BAM
    rm "${accession}.bam"
    
    downloaded_bams+=("${accession}.sort.bam")
    echo "Completed: $accession"
done

# Merge BAM files if multiple, otherwise rename single file
output_bam="${SAMPLE_ID}.${ASSAY}.merged.bam"

if [[ ${#downloaded_bams[@]} -gt 1 ]]; then
    echo "Merging ${#downloaded_bams[@]} BAM files..."
    samtools merge -@ "$THREADS" "$output_bam" "${downloaded_bams[@]}"
    
    # Clean up individual sorted BAMs
    for bam in "${downloaded_bams[@]}"; do
        rm "$bam"
    done
else
    echo "Single BAM file, renaming..."
    mv "${downloaded_bams[0]}" "$output_bam"
fi

echo "=== Download Complete ==="
echo "Output file: $output_bam"
echo "File size: $(du -h "$output_bam" | cut -f1)"

# Validate output
if [[ -f "$output_bam" ]]; then
    echo "✓ Successfully created merged BAM file"
    samtools quickcheck "$output_bam" && echo "✓ BAM file integrity check passed"
else
    echo "✗ Error: Output BAM file not created" >&2
    exit 1
fi

echo "=== ENCODE BAM Download Finished ==="