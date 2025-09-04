#!/bin/bash

# Genome Liftover Script
# Converts the liftover_fragments.wdl workflow to a standalone bash script
# Lifts genomic coordinates between genome assemblies (e.g., hg19 to hg38)

set -euo pipefail

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -i INPUT_BED -s SAMPLE_ID -c CHAIN_FILE [OPTIONS]

Lift genomic coordinates between genome assemblies using UCSC liftOver.

REQUIRED PARAMETERS:
  -i INPUT_BED     Input BED file (can be gzipped)
  -s SAMPLE_ID     Sample identifier for output file naming
  -c CHAIN_FILE    LiftOver chain file (e.g., hg19ToHg38.over.chain.gz)

OPTIONAL PARAMETERS:
  -o OUTPUT_DIR    Output directory (default: current directory)
  -l LIFTOVER_BIN  Path to liftOver binary (default: auto-download)
  -m MIN_MATCH     Minimum fraction of bases that must map (default: 0.95)
  -h               Show this help message

EXAMPLES:
  $0 -i fragments_hg19.bed.gz -s sample1 -c hg19ToHg38.over.chain.gz
  $0 -i peaks.bed -s sample2 -c mm9ToMm10.over.chain.gz -o /path/to/output

DEPENDENCIES:
  - wget (for downloading liftOver if needed)
  - gzip, gunzip

OUTPUT:
  - [SAMPLE_ID].lifted.bed.gz: Successfully lifted coordinates
  - [SAMPLE_ID].unmapped.bed: Coordinates that failed to lift
  - [SAMPLE_ID].liftover_stats.txt: Liftover statistics

PROCESSING STEPS:
  1. Download liftOver binary if not provided
  2. Decompress input file if gzipped
  3. Run liftOver with specified chain file
  4. Compress output and generate statistics

NOTES:
  - Chain files can be downloaded from UCSC Genome Browser
  - Common chain files: hg19ToHg38, hg38ToHg19, mm9ToMm10, etc.
  - Unmapped regions are saved for quality control
  - Reports success rate and statistics
EOF
}

# Default values
OUTPUT_DIR="."
LIFTOVER_BIN=""
MIN_MATCH=0.95
INPUT_BED=""
SAMPLE_ID=""
CHAIN_FILE=""

# Parse command line arguments
while getopts "i:s:c:o:l:m:h" opt; do
    case $opt in
        i) INPUT_BED="$OPTARG" ;;
        s) SAMPLE_ID="$OPTARG" ;;
        c) CHAIN_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        l) LIFTOVER_BIN="$OPTARG" ;;
        m) MIN_MATCH="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$INPUT_BED" || -z "$SAMPLE_ID" || -z "$CHAIN_FILE" ]]; then
    echo "Error: Missing required parameters" >&2
    usage
    exit 1
fi

# Check input files exist
if [[ ! -f "$INPUT_BED" ]]; then
    echo "Error: Input BED file not found: $INPUT_BED" >&2
    exit 1
fi

if [[ ! -f "$CHAIN_FILE" ]]; then
    echo "Error: Chain file not found: $CHAIN_FILE" >&2
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== Genome Liftover Started ==="
echo "Input BED: $INPUT_BED"
echo "Sample ID: $SAMPLE_ID"
echo "Chain file: $CHAIN_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Minimum match: $MIN_MATCH"
echo "=============================="

# Change to output directory
cd "$OUTPUT_DIR"

# Handle liftOver binary
if [[ -z "$LIFTOVER_BIN" ]]; then
    LIFTOVER_BIN="./liftOver"
    if [[ ! -f "$LIFTOVER_BIN" ]]; then
        echo "Step 1: Downloading liftOver binary..."
        # Detect architecture
        if [[ $(uname -m) == "x86_64" ]]; then
            ARCH="x86_64"
        elif [[ $(uname -m) == "aarch64" ]]; then
            ARCH="aarch64"
        else
            echo "Error: Unsupported architecture: $(uname -m)" >&2
            echo "Please provide liftOver binary with -l option" >&2
            exit 1
        fi
        
        # Download from UCSC
        LIFTOVER_URL="http://hgdownload.cse.ucsc.edu/admin/exe/linux.${ARCH}/liftOver"
        wget -O "$LIFTOVER_BIN" "$LIFTOVER_URL"
        chmod +x "$LIFTOVER_BIN"
        echo "  Downloaded liftOver binary"
    fi
elif [[ ! -f "$LIFTOVER_BIN" ]]; then
    echo "Error: liftOver binary not found: $LIFTOVER_BIN" >&2
    exit 1
else
    # Copy to local directory if not already there
    if [[ "$(basename $LIFTOVER_BIN)" != "liftOver" || "$(dirname $LIFTOVER_BIN)" != "." ]]; then
        cp "$LIFTOVER_BIN" "./liftOver"
        LIFTOVER_BIN="./liftOver"
        chmod +x "$LIFTOVER_BIN"
    fi
fi

# Output files
TEMP_INPUT="temp_input.bed"
OUTPUT_BED="${SAMPLE_ID}.lifted.bed"
OUTPUT_BED_GZ="${OUTPUT_BED}.gz"
UNMAPPED_BED="${SAMPLE_ID}.unmapped.bed"
STATS_FILE="${SAMPLE_ID}.liftover_stats.txt"

echo "Step 2: Preparing input file..."
# Decompress input if gzipped
if [[ "$INPUT_BED" =~ \.gz$ ]]; then
    echo "  Decompressing input file..."
    zcat "$INPUT_BED" > "$TEMP_INPUT"
else
    echo "  Using uncompressed input file..."
    cp "$INPUT_BED" "$TEMP_INPUT"
fi

# Count input regions
input_count=$(wc -l < "$TEMP_INPUT")
echo "  Input regions: $input_count"

echo "Step 3: Running liftOver..."
# Run liftOver with minimum match requirement
"$LIFTOVER_BIN" -minMatch="$MIN_MATCH" "$TEMP_INPUT" "$CHAIN_FILE" "$OUTPUT_BED" "$UNMAPPED_BED"

echo "Step 4: Processing results..."
# Count results
if [[ -f "$OUTPUT_BED" ]]; then
    lifted_count=$(wc -l < "$OUTPUT_BED")
    gzip "$OUTPUT_BED"
else
    lifted_count=0
    touch "$OUTPUT_BED_GZ"
fi

if [[ -f "$UNMAPPED_BED" ]]; then
    unmapped_count=$(wc -l < "$UNMAPPED_BED")
else
    unmapped_count=0
    touch "$UNMAPPED_BED"
fi

# Calculate success rate
success_rate=$(echo "$lifted_count $input_count" | awk '{printf "%.2f", ($1 / $2) * 100}')

echo "Step 5: Generating statistics..."
# Create statistics file
cat > "$STATS_FILE" << EOF
Sample ID: $SAMPLE_ID
Input file: $INPUT_BED
Chain file: $CHAIN_FILE
Minimum match: $MIN_MATCH

Liftover Results:
- Input regions: $input_count
- Successfully lifted: $lifted_count
- Failed to lift: $unmapped_count
- Success rate: $success_rate%

Output files:
- Lifted coordinates: $OUTPUT_BED_GZ
- Unmapped coordinates: $UNMAPPED_BED
EOF

# Clean up temporary files
rm -f "$TEMP_INPUT"

echo "Step 6: Validating output..."
if [[ -f "$OUTPUT_BED_GZ" && -f "$UNMAPPED_BED" && -f "$STATS_FILE" ]]; then
    lifted_size=$(du -h "$OUTPUT_BED_GZ" | cut -f1)
    unmapped_size=$(du -h "$UNMAPPED_BED" | cut -f1)
    
    echo "✓ Successfully completed liftover"
    echo "  Lifted coordinates: $OUTPUT_BED_GZ ($lifted_size)"
    echo "  Unmapped coordinates: $UNMAPPED_BED ($unmapped_size)"
    echo "  Statistics: $STATS_FILE"
    echo "  Success rate: $success_rate% ($lifted_count/$input_count)"
    
    # Show sample of lifted coordinates
    if [[ $lifted_count -gt 0 ]]; then
        echo "  Sample lifted coordinates (first 3 lines):"
        zcat "$OUTPUT_BED_GZ" | head -3 | while read line; do
            echo "    $line"
        done
    fi
    
    # Warning for low success rate
    if [[ $(echo "$success_rate" | awk '{print ($1 < 90 ? 1 : 0)}') -eq 1 ]]; then
        echo "  ⚠️  Warning: Low success rate ($success_rate%). Check chain file and input format."
    fi
else
    echo "✗ Error: Output files not created" >&2
    exit 1
fi

echo "=== Genome Liftover Complete ==="