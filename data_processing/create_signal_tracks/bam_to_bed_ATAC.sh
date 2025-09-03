#!/bin/bash

# ATAC-seq BAM to BED Conversion Script
# Converts the bam_to_bed_ATAC.wdl workflow to a standalone bash script
# Processes ATAC-seq BAM files to create fragment files

set -euo pipefail

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -i INPUT_BAM -s SAMPLE_ID [OPTIONS]

Convert ATAC-seq BAM file to BED fragments file with quality filtering.

REQUIRED PARAMETERS:
  -i INPUT_BAM     Input BAM file (coordinate-sorted)
  -s SAMPLE_ID     Sample identifier for output file naming

OPTIONAL PARAMETERS:
  -o OUTPUT_DIR    Output directory (default: current directory)
  -j THREADS       Number of threads for samtools (default: 1)
  -m MEMORY        Memory limit for samtools sort (default: 2G)
  -q MIN_MAPQ      Minimum mapping quality (default: 20)
  -f MIN_FRAGLEN   Minimum fragment length (default: 1)
  -h               Show this help message

EXAMPLES:
  $0 -i sample.bam -s sample1
  $0 -i /path/to/sample.bam -s sample2 -o /path/to/output -j 4

DEPENDENCIES:
  - samtools
  - bedtools
  - awk, gzip

OUTPUT:
  - [SAMPLE_ID].bed.gz: Filtered ATAC-seq fragments
  - [SAMPLE_ID].fragment_count.txt: Number of fragments (in millions)

PROCESSING STEPS:
  1. Name-sort BAM file
  2. Convert to BEDPE format
  3. Filter for proper pairs (same chromosome)
  4. Filter by mapping quality (≥20)
  5. Extract fragment coordinates with Tn5 offset correction
  6. Filter by fragment length
  7. Compress output

NOTES:
  - Applies Tn5 transposase insertion site correction (+4/-5 bp)
  - Filters out fragments with invalid coordinates
  - Reports fragment count for normalization
EOF
}

# Default values
OUTPUT_DIR="."
THREADS=1
MEMORY="2G"
MIN_MAPQ=20
MIN_FRAGLEN=1
INPUT_BAM=""
SAMPLE_ID=""

# Parse command line arguments
while getopts "i:s:o:j:m:q:f:h" opt; do
    case $opt in
        i) INPUT_BAM="$OPTARG" ;;
        s) SAMPLE_ID="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        j) THREADS="$OPTARG" ;;
        m) MEMORY="$OPTARG" ;;
        q) MIN_MAPQ="$OPTARG" ;;
        f) MIN_FRAGLEN="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$INPUT_BAM" || -z "$SAMPLE_ID" ]]; then
    echo "Error: Missing required parameters" >&2
    usage
    exit 1
fi

# Check input file exists
if [[ ! -f "$INPUT_BAM" ]]; then
    echo "Error: Input BAM file not found: $INPUT_BAM" >&2
    exit 1
fi

# Check dependencies
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed or not in PATH" >&2
        exit 1
    fi
}

check_dependency "samtools"
check_dependency "bedtools"
check_dependency "awk"
check_dependency "gzip"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== ATAC-seq BAM to BED Conversion Started ==="
echo "Input BAM: $INPUT_BAM"
echo "Sample ID: $SAMPLE_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Threads: $THREADS"
echo "Memory: $MEMORY"
echo "Min MAPQ: $MIN_MAPQ"
echo "Min fragment length: $MIN_FRAGLEN"
echo "================================================"

# Change to output directory
cd "$OUTPUT_DIR"

# Create temporary files
TEMP_SORTED_BAM="temp_name_sorted.bam"
TEMP_BEDPE="temp.bedpe"
OUTPUT_BED="${SAMPLE_ID}.bed.gz"
FRAGMENT_COUNT_FILE="${SAMPLE_ID}.fragment_count.txt"

echo "Step 1: Name-sorting BAM file..."
samtools sort -n "$INPUT_BAM" -@ "$THREADS" -m "$MEMORY" -o "$TEMP_SORTED_BAM"

echo "Step 2: Converting to BEDPE and filtering..."
# Convert to BEDPE, filter for proper pairs and mapping quality
bedtools bamtobed -i "$TEMP_SORTED_BAM" -bedpe | \
awk -v mapq="$MIN_MAPQ" '$1 == $4 && $8 >= mapq' | \
awk -v OFS="\t" '{
    if($9 == "+") {
        print $1, $2+4, $6-5
    } else if($9 == "-") {
        print $1, $5+4, $3-5
    }
}' | \
awk -v OFS="\t" '{
    if($3 < $2) {
        print $1, $3, $2
    } else if($3 > $2) {
        print $1, $2, $3
    }
}' | \
awk -v min_len="$MIN_FRAGLEN" '$3-$2 > min_len' | \
gzip > "$OUTPUT_BED"

echo "Step 3: Calculating fragment count..."
zcat "$OUTPUT_BED" | wc -l | awk '{print $1 / 1000000.0}' > "$FRAGMENT_COUNT_FILE"

# Clean up temporary files
rm -f "$TEMP_SORTED_BAM" "$TEMP_BEDPE"

echo "Step 4: Validating output..."
if [[ -f "$OUTPUT_BED" && -f "$FRAGMENT_COUNT_FILE" ]]; then
    fragment_count=$(cat "$FRAGMENT_COUNT_FILE")
    file_size=$(du -h "$OUTPUT_BED" | cut -f1)
    
    echo "✓ Successfully created fragment file"
    echo "  Output file: $OUTPUT_BED"
    echo "  File size: $file_size"
    echo "  Fragment count: ${fragment_count}M fragments"
    
    # Quick validation of file format
    echo "  Sample fragments (first 3 lines):"
    zcat "$OUTPUT_BED" | head -3 | while read line; do
        echo "    $line"
    done
else
    echo "✗ Error: Output files not created" >&2
    exit 1
fi

echo "=== ATAC-seq BAM to BED Conversion Complete ==="