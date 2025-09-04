#!/bin/bash

# RAMPAGE-seq BAM to BedGraph Script
# Converts the bam_to_bed_RAMPAGE_5prime.wdl workflow to a standalone bash script
# Processes RAMPAGE-seq BAM files to create 5' TSS signal tracks

set -euo pipefail

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -i INPUT_BAM -s SAMPLE_ID -g GENOME_FILE [OPTIONS]

Convert RAMPAGE-seq BAM file to 5' TSS bedGraph signal track.

REQUIRED PARAMETERS:
  -i INPUT_BAM     Input BAM file URL or local file path
  -s SAMPLE_ID     Sample identifier for output file naming
  -g GENOME_FILE   Genome sizes file (chromosome sizes)

OPTIONAL PARAMETERS:
  -o OUTPUT_DIR    Output directory (default: current directory)
  -j THREADS       Number of threads for samtools (default: 1)
  -p SCALE_FACTOR  Paired-end scaling factor (default: 2.0)
  -w WINDOW        TSS window size (default: 10bp, ±5bp)
  -h               Show this help message

EXAMPLES:
  $0 -i sample.bam -s sample1 -g hg38.chrom.sizes
  $0 -i "https://url/to/sample.bam" -s sample2 -g mm10.chrom.sizes -o /path/to/output

DEPENDENCIES:
  - wget (for URL downloads)
  - samtools
  - bedtools
  - awk, gzip

OUTPUT:
  - [SAMPLE_ID].bed.gz: 5' TSS positions
  - [SAMPLE_ID].bedgraph.gz: Scaled bedGraph signal track
  - [SAMPLE_ID].scale_factor.txt: Final scaling factor used

PROCESSING STEPS:
  1. Download BAM if URL provided
  2. Extract 5' TSS positions from properly paired reads
  3. Create ±5bp windows around TSS
  4. Generate normalized bedGraph signal
  5. Filter out non-standard chromosomes

NOTES:
  - Uses mate 1 TSS positions for paired-end RAMPAGE data
  - Applies scaling normalization based on total read count
  - Filters out decoy/alt chromosomes (KI, GL, EBV, chrM, etc.)
EOF
}

# Default values
OUTPUT_DIR="."
THREADS=1
SCALE_FACTOR=2.0
WINDOW=5
INPUT_BAM=""
SAMPLE_ID=""
GENOME_FILE=""

# Parse command line arguments
while getopts "i:s:g:o:j:p:w:h" opt; do
    case $opt in
        i) INPUT_BAM="$OPTARG" ;;
        s) SAMPLE_ID="$OPTARG" ;;
        g) GENOME_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        j) THREADS="$OPTARG" ;;
        p) SCALE_FACTOR="$OPTARG" ;;
        w) WINDOW="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$INPUT_BAM" || -z "$SAMPLE_ID" || -z "$GENOME_FILE" ]]; then
    echo "Error: Missing required parameters" >&2
    usage
    exit 1
fi

# Check genome file exists
if [[ ! -f "$GENOME_FILE" ]]; then
    echo "Error: Genome sizes file not found: $GENOME_FILE" >&2
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

echo "=== RAMPAGE-seq BAM to BedGraph Conversion Started ==="
echo "Input BAM: $INPUT_BAM"
echo "Sample ID: $SAMPLE_ID"
echo "Genome file: $GENOME_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Threads: $THREADS"
echo "Scale factor: $SCALE_FACTOR"
echo "TSS window: ±${WINDOW}bp"
echo "========================================================="

# Change to output directory
cd "$OUTPUT_DIR"

# Download BAM if URL provided, otherwise use local file
LOCAL_BAM="downloaded.bam"
if [[ "$INPUT_BAM" =~ ^https?:// ]]; then
    echo "Step 1: Downloading BAM file..."
    wget -O "$LOCAL_BAM" "$INPUT_BAM"
else
    if [[ ! -f "$INPUT_BAM" ]]; then
        echo "Error: Local BAM file not found: $INPUT_BAM" >&2
        exit 1
    fi
    LOCAL_BAM="$INPUT_BAM"
fi

# Create temporary files
TEMP_BED="${SAMPLE_ID}.temp.bed"
OUTPUT_BED="${SAMPLE_ID}.bed.gz"
OUTPUT_BEDGRAPH="${SAMPLE_ID}.bedgraph.gz"
SCALE_FACTOR_FILE="${SAMPLE_ID}.scale_factor.txt"

echo "Step 2: Extracting 5' TSS positions..."
# Extract 5' TSS positions from properly paired reads
bedtools bamtobed -i "$LOCAL_BAM" | \
awk '$5 > 0' | \
gzip > "$OUTPUT_BED"

echo "Step 3: Calculating scaling factor..."
# Calculate scaling factor based on fragment count
fragment_count=$(zcat "$OUTPUT_BED" | wc -l)
scale_factor_value=$(echo "$fragment_count $SCALE_FACTOR" | awk '{print $1 / ($2 * 1000000.0)}')
scale_factor_inv=$(echo "$scale_factor_value" | awk '{print 1.0 / $1}')

echo "  Fragment count: $fragment_count"
echo "  Scale factor: $scale_factor_value"
echo "  Inverse scale factor: $scale_factor_inv"

# Save scale factor
echo "$scale_factor_value" > "$SCALE_FACTOR_FILE"

echo "Step 4: Creating TSS bedGraph signal..."
# Create temporary TSS positions file
TEMP_TSS="temp_tss.bed"

# Extract properly paired reads and create TSS windows
samtools view -bf 0x2 "$LOCAL_BAM" | \
samtools sort -n | \
bedtools bamtobed -i stdin -bedpe -mate1 | \
awk -v window="$WINDOW" 'BEGIN {OFS="\t"} {
    if ($9 == "+") {
        print $1, $2-window, $2+window
    } else if ($9 == "-") {
        print $1, $3-window, $3+window
    }
}' | \
grep -v 'KI\|GL\|EBV\|chrM\|chrMT\|K\|J\|phi\|ERCC' | \
sort -k1,1 -k2,2n > "$TEMP_TSS"

# Generate normalized bedGraph
bedtools genomecov -i "$TEMP_TSS" -g "$GENOME_FILE" -bg -scale "$scale_factor_inv" | \
sort -k1,1 -k2,2n | \
gzip > "$OUTPUT_BEDGRAPH"

# Clean up temporary files
rm -f "$TEMP_BED" "$TEMP_TSS"
if [[ "$INPUT_BAM" =~ ^https?:// ]]; then
    rm -f "$LOCAL_BAM"
fi

echo "Step 5: Validating output..."
if [[ -f "$OUTPUT_BED" && -f "$OUTPUT_BEDGRAPH" && -f "$SCALE_FACTOR_FILE" ]]; then
    bed_size=$(du -h "$OUTPUT_BED" | cut -f1)
    bedgraph_size=$(du -h "$OUTPUT_BEDGRAPH" | cut -f1)
    final_scale_factor=$(cat "$SCALE_FACTOR_FILE")
    
    echo "✓ Successfully created RAMPAGE signal tracks"
    echo "  TSS positions: $OUTPUT_BED ($bed_size)"
    echo "  BedGraph signal: $OUTPUT_BEDGRAPH ($bedgraph_size)"
    echo "  Final scale factor: $final_scale_factor"
    
    # Show sample of bedGraph
    echo "  Sample bedGraph entries (first 3 lines):"
    zcat "$OUTPUT_BEDGRAPH" | head -3 | while read line; do
        echo "    $line"
    done
else
    echo "✗ Error: Output files not created" >&2
    exit 1
fi

echo "=== RAMPAGE-seq BAM to BedGraph Conversion Complete ==="