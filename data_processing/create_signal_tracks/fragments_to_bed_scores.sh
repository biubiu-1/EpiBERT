#!/bin/bash

# Fragments to BedGraph Signal Script
# Converts the fragments_to_bed_scores.wdl workflow to a standalone bash script
# Creates scaled bedGraph signal tracks from ATAC-seq fragment files

set -euo pipefail

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -i INPUT_FRAGMENTS -s SAMPLE_ID -g GENOME_FILE [OPTIONS]

Convert ATAC-seq fragments to normalized bedGraph signal track.

REQUIRED PARAMETERS:
  -i INPUT_FRAGMENTS  Input fragments file (BED format, can be gzipped)
  -s SAMPLE_ID        Sample identifier for output file naming
  -g GENOME_FILE      Genome sizes file (chromosome sizes)

OPTIONAL PARAMETERS:
  -o OUTPUT_DIR       Output directory (default: current directory)
  -n NORM_FACTOR      Normalization factor for scaling (default: auto-calculate)
  -f SCALE_TYPE       Scaling type: 'cpm' (counts per million) or 'fpkm' (default: cpm)
  -h                  Show this help message

EXAMPLES:
  $0 -i fragments.bed.gz -s sample1 -g hg38.chrom.sizes
  $0 -i fragments.bed -s sample2 -g mm10.chrom.sizes -o /path/to/output -n 1.5

DEPENDENCIES:
  - bedtools
  - awk, gzip

OUTPUT:
  - [SAMPLE_ID].bedgraph.gz: Normalized bedGraph signal track
  - [SAMPLE_ID].scaling_info.txt: Scaling factor and fragment count info

PROCESSING STEPS:
  1. Count total fragments for normalization
  2. Calculate scaling factor (CPM or FPKM)
  3. Generate coverage using bedtools genomecov
  4. Apply scaling normalization
  5. Sort and compress output

NOTES:
  - Uses counts per million (CPM) normalization by default
  - FPKM normalization requires effective genome size estimation
  - Filters out non-standard chromosomes
EOF
}

# Default values
OUTPUT_DIR="."
NORM_FACTOR=""
SCALE_TYPE="cpm"
INPUT_FRAGMENTS=""
SAMPLE_ID=""
GENOME_FILE=""

# Parse command line arguments
while getopts "i:s:g:o:n:f:h" opt; do
    case $opt in
        i) INPUT_FRAGMENTS="$OPTARG" ;;
        s) SAMPLE_ID="$OPTARG" ;;
        g) GENOME_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        n) NORM_FACTOR="$OPTARG" ;;
        f) SCALE_TYPE="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$INPUT_FRAGMENTS" || -z "$SAMPLE_ID" || -z "$GENOME_FILE" ]]; then
    echo "Error: Missing required parameters" >&2
    usage
    exit 1
fi

# Check input files exist
if [[ ! -f "$INPUT_FRAGMENTS" ]]; then
    echo "Error: Input fragments file not found: $INPUT_FRAGMENTS" >&2
    exit 1
fi

if [[ ! -f "$GENOME_FILE" ]]; then
    echo "Error: Genome sizes file not found: $GENOME_FILE" >&2
    exit 1
fi

# Validate scale type
if [[ "$SCALE_TYPE" != "cpm" && "$SCALE_TYPE" != "fpkm" ]]; then
    echo "Error: Scale type must be 'cpm' or 'fpkm'" >&2
    exit 1
fi

# Check dependencies
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed or not in PATH" >&2
        exit 1
    fi
}

check_dependency "bedtools"
check_dependency "awk"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== Fragments to BedGraph Conversion Started ==="
echo "Input fragments: $INPUT_FRAGMENTS"
echo "Sample ID: $SAMPLE_ID"
echo "Genome file: $GENOME_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Scale type: $SCALE_TYPE"
if [[ -n "$NORM_FACTOR" ]]; then
    echo "Custom normalization factor: $NORM_FACTOR"
fi
echo "=================================================="

# Change to output directory
cd "$OUTPUT_DIR"

# Output files
OUTPUT_BEDGRAPH="${SAMPLE_ID}.bedgraph.gz"
SCALING_INFO="${SAMPLE_ID}.scaling_info.txt"
TEMP_FRAGMENTS="temp_fragments.bed"

echo "Step 1: Preparing fragments file..."
# Determine if input is gzipped and extract if needed
if [[ "$INPUT_FRAGMENTS" =~ \.gz$ ]]; then
    echo "  Extracting gzipped fragments..."
    zcat "$INPUT_FRAGMENTS" > "$TEMP_FRAGMENTS"
else
    echo "  Using uncompressed fragments..."
    cp "$INPUT_FRAGMENTS" "$TEMP_FRAGMENTS"
fi

echo "Step 2: Counting fragments for normalization..."
fragment_count=$(wc -l < "$TEMP_FRAGMENTS")
echo "  Total fragments: $fragment_count"

echo "Step 3: Calculating scaling factor..."
if [[ -n "$NORM_FACTOR" ]]; then
    # Use custom normalization factor
    scale_factor="$NORM_FACTOR"
    echo "  Using custom scaling factor: $scale_factor"
elif [[ "$SCALE_TYPE" == "cpm" ]]; then
    # Counts per million normalization
    scale_factor=$(echo "$fragment_count" | awk '{print 1000000.0 / $1}')
    echo "  CPM scaling factor: $scale_factor"
elif [[ "$SCALE_TYPE" == "fpkm" ]]; then
    # FPKM normalization (simplified, assumes average fragment length)
    avg_fragment_length=200  # Typical ATAC-seq fragment length
    scale_factor=$(echo "$fragment_count $avg_fragment_length" | awk '{print 1000000000.0 / ($1 * $2)}')
    echo "  FPKM scaling factor: $scale_factor (assuming ${avg_fragment_length}bp avg length)"
fi

# Save scaling information
cat > "$SCALING_INFO" << EOF
Sample ID: $SAMPLE_ID
Fragment count: $fragment_count
Scale type: $SCALE_TYPE
Scale factor: $scale_factor
EOF

if [[ -n "$NORM_FACTOR" ]]; then
    echo "Custom normalization: $NORM_FACTOR" >> "$SCALING_INFO"
fi

echo "Step 4: Generating bedGraph coverage..."
# Filter out non-standard chromosomes and generate coverage
grep -v 'KI\|GL\|EBV\|chrM\|chrMT\|_KI\|_GL' "$TEMP_FRAGMENTS" | \
sort -k1,1 -k2,2n | \
bedtools genomecov -i stdin -g "$GENOME_FILE" -bg -scale "$scale_factor" | \
sort -k1,1 -k2,2n | \
gzip > "$OUTPUT_BEDGRAPH"

# Clean up temporary files
rm -f "$TEMP_FRAGMENTS"

echo "Step 5: Validating output..."
if [[ -f "$OUTPUT_BEDGRAPH" && -f "$SCALING_INFO" ]]; then
    bedgraph_size=$(du -h "$OUTPUT_BEDGRAPH" | cut -f1)
    total_regions=$(zcat "$OUTPUT_BEDGRAPH" | wc -l)
    
    echo "✓ Successfully created bedGraph signal track"
    echo "  Output file: $OUTPUT_BEDGRAPH ($bedgraph_size)"
    echo "  Scaling info: $SCALING_INFO"
    echo "  Total regions: $total_regions"
    
    # Show sample of bedGraph and scaling info
    echo "  Sample bedGraph entries (first 3 lines):"
    zcat "$OUTPUT_BEDGRAPH" | head -3 | while read line; do
        echo "    $line"
    done
    
    echo "  Scaling information:"
    while read line; do
        echo "    $line"
    done < "$SCALING_INFO"
    
    # Calculate some basic statistics
    echo "  Signal statistics:"
    zcat "$OUTPUT_BEDGRAPH" | awk '{
        sum += $4 * ($3 - $2)
        count++
        if (NR == 1 || $4 < min) min = $4
        if (NR == 1 || $4 > max) max = $4
    } END {
        print "    Mean signal: " (sum / count)
        print "    Min signal: " min
        print "    Max signal: " max
    }'
else
    echo "✗ Error: Output files not created" >&2
    exit 1
fi

echo "=== Fragments to BedGraph Conversion Complete ==="