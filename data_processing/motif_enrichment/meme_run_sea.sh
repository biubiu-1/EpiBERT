#!/bin/bash

# MEME SEA Motif Enrichment Script
# Converts the meme_run_sea.wdl workflow to a standalone bash script
# Performs transcription factor motif enrichment analysis using MEME SEA

set -euo pipefail

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -p PEAK_FILE -m MOTIF_FILE -g GENOME_FASTA -b BACKGROUND_PEAKS [OPTIONS]

Perform motif enrichment analysis using MEME SEA (Simple Enrichment Analysis).

REQUIRED PARAMETERS:
  -p PEAK_FILE        Input peak file (BED format, can be gzipped)
  -m MOTIF_FILE       Motif database file (MEME format)
  -g GENOME_FASTA     Reference genome FASTA file
  -b BACKGROUND_PEAKS Background peaks file (BED format)

OPTIONAL PARAMETERS:
  -o OUTPUT_DIR       Output directory (default: current directory)
  -n INPUT_NAME       Output file prefix (default: motif_enrichment)
  -t THRESH           P-value threshold for SEA (default: 0.05)
  -k TOP_PEAKS        Number of top peaks to analyze (default: 10000)
  -w HALF_WIDTH       Half-width around peak summit (default: 150bp)
  -j THREADS          Number of threads (default: 1)
  -h                  Show this help message

EXAMPLES:
  $0 -p peaks.bed.gz -m motifs.meme -g hg38.fa -b background.bed
  $0 -p peaks.bed -m TF_motifs.meme -g genome.fa -b bg.bed -o results -n sample1 -t 0.01

DEPENDENCIES:
  - MEME Suite (sea command)
  - bedtools
  - awk, sort, head

OUTPUT:
  - [INPUT_NAME].tsv: MEME SEA results with motif enrichment statistics
  - [INPUT_NAME]_peak_sequences.fa: Peak sequences used for analysis
  - [INPUT_NAME]_background_sequences.fa: Background sequences

PROCESSING STEPS:
  1. Select top N peaks by score/significance
  2. Center peaks around summits with specified half-width
  3. Extract peak sequences from genome FASTA
  4. Extract background sequences
  5. Run MEME SEA motif enrichment analysis

NOTES:
  - Peaks are ranked by their score (column 5) for top-k selection
  - Background peaks should represent accessible genomic regions
  - MEME motif database format required for motif file
  - Results include E-values, p-values, and motif match statistics
EOF
}

# Default values
OUTPUT_DIR="."
INPUT_NAME="motif_enrichment"
THRESH=0.05
TOP_PEAKS=10000
HALF_WIDTH=150
THREADS=1
PEAK_FILE=""
MOTIF_FILE=""
GENOME_FASTA=""
BACKGROUND_PEAKS=""

# Parse command line arguments
while getopts "p:m:g:b:o:n:t:k:w:j:h" opt; do
    case $opt in
        p) PEAK_FILE="$OPTARG" ;;
        m) MOTIF_FILE="$OPTARG" ;;
        g) GENOME_FASTA="$OPTARG" ;;
        b) BACKGROUND_PEAKS="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        n) INPUT_NAME="$OPTARG" ;;
        t) THRESH="$OPTARG" ;;
        k) TOP_PEAKS="$OPTARG" ;;
        w) HALF_WIDTH="$OPTARG" ;;
        j) THREADS="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$PEAK_FILE" || -z "$MOTIF_FILE" || -z "$GENOME_FASTA" || -z "$BACKGROUND_PEAKS" ]]; then
    echo "Error: Missing required parameters" >&2
    usage
    exit 1
fi

# Check input files exist
for file in "$PEAK_FILE" "$MOTIF_FILE" "$GENOME_FASTA" "$BACKGROUND_PEAKS"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Input file not found: $file" >&2
        exit 1
    fi
done

# Check dependencies
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed or not in PATH" >&2
        echo "Please install MEME Suite and bedtools" >&2
        exit 1
    fi
}

check_dependency "sea"
check_dependency "bedtools"
check_dependency "awk"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== MEME SEA Motif Enrichment Analysis Started ==="
echo "Peak file: $PEAK_FILE"
echo "Motif file: $MOTIF_FILE"
echo "Genome FASTA: $GENOME_FASTA"
echo "Background peaks: $BACKGROUND_PEAKS"
echo "Output directory: $OUTPUT_DIR"
echo "Output prefix: $INPUT_NAME"
echo "P-value threshold: $THRESH"
echo "Top peaks: $TOP_PEAKS"
echo "Peak half-width: $HALF_WIDTH bp"
echo "Threads: $THREADS"
echo "========================================================="

# Change to output directory
cd "$OUTPUT_DIR"

# Output files
PEAK_SEQUENCES="${INPUT_NAME}_peak_sequences.fa"
BACKGROUND_SEQUENCES="${INPUT_NAME}_background_sequences.fa"
TEMP_PEAKS="temp_sorted_peaks.bed"
OUTPUT_TSV="${INPUT_NAME}.tsv"

echo "Step 1: Preparing peak file..."
# Extract and sort peaks by score (column 5)
if [[ "$PEAK_FILE" =~ \.gz$ ]]; then
    echo "  Extracting gzipped peak file..."
    zcat "$PEAK_FILE" | sort -k5,5nr | head -"$TOP_PEAKS" > "$TEMP_PEAKS"
else
    echo "  Processing uncompressed peak file..."
    sort -k5,5nr "$PEAK_FILE" | head -"$TOP_PEAKS" > "$TEMP_PEAKS"
fi

total_input_peaks=$(wc -l < "$TEMP_PEAKS")
echo "  Selected top $total_input_peaks peaks for analysis"

echo "Step 2: Centering peaks around summits..."
# Center peaks around summit with specified half-width
# Assuming peak format: chr start end name score strand summit_offset
awk -v hw="$HALF_WIDTH" 'BEGIN{OFS="\t"} {
    summit_pos = $2 + $7  # Assuming column 7 contains summit offset
    if (NF < 7) summit_pos = ($2 + $3) / 2  # Use peak center if no summit info
    print $1, summit_pos - hw, summit_pos + hw
}' "$TEMP_PEAKS" | \
awk 'BEGIN{OFS="\t"} $2 >= 0 && $3 > $2' | \
sort -k1,1 -k2,2n > sorted_peaks.bed

centered_peaks=$(wc -l < sorted_peaks.bed)
echo "  Created $centered_peaks centered peak regions"

echo "Step 3: Extracting peak sequences..."
# Extract sequences from genome FASTA
bedtools getfasta -fi "$GENOME_FASTA" -bed sorted_peaks.bed > "$PEAK_SEQUENCES"

# Count sequences
peak_seq_count=$(grep -c "^>" "$PEAK_SEQUENCES")
echo "  Extracted $peak_seq_count peak sequences"

echo "Step 4: Extracting background sequences..."
# Extract background sequences
bedtools getfasta -fi "$GENOME_FASTA" -bed "$BACKGROUND_PEAKS" > "$BACKGROUND_SEQUENCES"

# Count background sequences
bg_seq_count=$(grep -c "^>" "$BACKGROUND_SEQUENCES")
echo "  Extracted $bg_seq_count background sequences"

echo "Step 5: Running MEME SEA analysis..."
# Create SEA output directory
mkdir -p sea_output

# Run MEME SEA
echo "  Analyzing motif enrichments..."
sea --p "$PEAK_SEQUENCES" \
    --m "$MOTIF_FILE" \
    --n "$BACKGROUND_SEQUENCES" \
    --thresh "$THRESH" \
    --verbosity 1 \
    --o sea_output

# Move and rename output
if [[ -f "sea_output/sea.tsv" ]]; then
    mv "sea_output/sea.tsv" "$OUTPUT_TSV"
    echo "  ✓ SEA analysis completed successfully"
else
    echo "  ✗ Error: SEA analysis failed" >&2
    exit 1
fi

echo "Step 6: Processing results..."
# Count significant motifs
if [[ -f "$OUTPUT_TSV" ]]; then
    total_motifs=$(tail -n +2 "$OUTPUT_TSV" | wc -l)
    significant_motifs=$(tail -n +2 "$OUTPUT_TSV" | awk -v t="$THRESH" '$3 <= t' | wc -l)
    
    echo "  Total motifs tested: $total_motifs"
    echo "  Significant motifs (p ≤ $THRESH): $significant_motifs"
    
    # Show top 5 significant motifs
    if [[ $significant_motifs -gt 0 ]]; then
        echo "  Top 5 significant motifs:"
        echo "    Motif_ID	E-value	p-value"
        tail -n +2 "$OUTPUT_TSV" | awk -v t="$THRESH" '$3 <= t' | \
        sort -k3,3g | head -5 | awk 'BEGIN{OFS="\t"} {print $1, $2, $3}' | \
        while read line; do
            echo "    $line"
        done
    fi
fi

# Clean up temporary files
rm -f "$TEMP_PEAKS" sorted_peaks.bed
rm -rf sea_output

echo "Step 7: Validating output..."
if [[ -f "$OUTPUT_TSV" && -f "$PEAK_SEQUENCES" && -f "$BACKGROUND_SEQUENCES" ]]; then
    output_size=$(du -h "$OUTPUT_TSV" | cut -f1)
    peak_seq_size=$(du -h "$PEAK_SEQUENCES" | cut -f1)
    bg_seq_size=$(du -h "$BACKGROUND_SEQUENCES" | cut -f1)
    
    echo "✓ Successfully completed motif enrichment analysis"
    echo "  Results: $OUTPUT_TSV ($output_size)"
    echo "  Peak sequences: $PEAK_SEQUENCES ($peak_seq_size)"
    echo "  Background sequences: $BACKGROUND_SEQUENCES ($bg_seq_size)"
    
    # Show file contents summary
    echo "  Analysis summary:"
    echo "    Peaks analyzed: $peak_seq_count"
    echo "    Background sequences: $bg_seq_count"
    echo "    Motifs tested: $total_motifs"
    echo "    Significant enrichments: $significant_motifs"
else
    echo "✗ Error: Output files not created" >&2
    exit 1
fi

echo "=== MEME SEA Motif Enrichment Analysis Complete ==="