#!/bin/bash

# MACS2 Peak Calling from Fragments Script
# Converts the macs2_peak_call_from_fragment.wdl workflow to a standalone bash script
# Performs ATAC-seq peak calling with pseudo-replicates and IDR filtering

set -euo pipefail

# Function to display usage
usage() {
    cat << EOF
Usage: $0 -i INPUT_FRAGMENTS -s SAMPLE_ID -b BLACKLIST [OPTIONS]

Perform MACS2 peak calling from ATAC-seq fragments with pseudo-replicate analysis.

REQUIRED PARAMETERS:
  -i INPUT_FRAGMENTS  Input fragments file (BED format, can be gzipped)
  -s SAMPLE_ID        Sample identifier for output file naming
  -b BLACKLIST        Blacklist regions file (BED format)

OPTIONAL PARAMETERS:
  -o OUTPUT_DIR       Output directory (default: current directory)
  -q Q_VALUE          Q-value cutoff (default: auto-calculated based on data)
  -w HALF_WIDTH       Half-width for peak centering (default: 250bp)
  -p POS_SHIFT        Positive strand shift (default: 4)
  -n NEG_SHIFT        Negative strand shift (default: -5)
  -j THREADS          Number of threads (default: 1)
  -h                  Show this help message

EXAMPLES:
  $0 -i fragments.bed.gz -s sample1 -b hg38.blacklist.bed
  $0 -i fragments.bed -s sample2 -b blacklist.bed -o /path/to/output -q 0.01

DEPENDENCIES:
  - MACS2
  - bedtools
  - python3 (for peak merging script)
  - awk, sort, gzip

OUTPUT:
  - [SAMPLE_ID].pr.peaks.bed.gz: Final merged and filtered peaks
  - [SAMPLE_ID].narrow_peaks.*: Raw MACS2 narrow peaks (full, PR1, PR2)
  - [SAMPLE_ID].peak_stats.txt: Peak calling statistics

PROCESSING STEPS:
  1. Split fragments into pseudo-replicates
  2. Convert fragments to Tn5 insertion sites
  3. Run MACS2 on full dataset and pseudo-replicates
  4. Filter peaks by blacklist regions
  5. Center peaks and apply half-width
  6. Merge overlapping peaks within pseudo-replicates
  7. Apply IDR-like filtering for reproducible peaks

NOTES:
  - Auto-calculates q-value based on fragment count
  - Uses ATAC-seq specific Tn5 shifting parameters
  - Requires peaks to be present in both pseudo-replicates
EOF
}

# Default values
OUTPUT_DIR="."
Q_VALUE=""
HALF_WIDTH=250
POS_SHIFT=4
NEG_SHIFT=-5
THREADS=1
INPUT_FRAGMENTS=""
SAMPLE_ID=""
BLACKLIST=""

# Parse command line arguments
while getopts "i:s:b:o:q:w:p:n:j:h" opt; do
    case $opt in
        i) INPUT_FRAGMENTS="$OPTARG" ;;
        s) SAMPLE_ID="$OPTARG" ;;
        b) BLACKLIST="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        q) Q_VALUE="$OPTARG" ;;
        w) HALF_WIDTH="$OPTARG" ;;
        p) POS_SHIFT="$OPTARG" ;;
        n) NEG_SHIFT="$OPTARG" ;;
        j) THREADS="$OPTARG" ;;
        h) usage; exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    esac
done

# Validate required parameters
if [[ -z "$INPUT_FRAGMENTS" || -z "$SAMPLE_ID" || -z "$BLACKLIST" ]]; then
    echo "Error: Missing required parameters" >&2
    usage
    exit 1
fi

# Check input files exist
if [[ ! -f "$INPUT_FRAGMENTS" ]]; then
    echo "Error: Input fragments file not found: $INPUT_FRAGMENTS" >&2
    exit 1
fi

if [[ ! -f "$BLACKLIST" ]]; then
    echo "Error: Blacklist file not found: $BLACKLIST" >&2
    exit 1
fi

# Check dependencies
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed or not in PATH" >&2
        exit 1
    fi
}

check_dependency "macs2"
check_dependency "bedtools"
check_dependency "python3"
check_dependency "awk"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=== MACS2 Peak Calling Started ==="
echo "Input fragments: $INPUT_FRAGMENTS"
echo "Sample ID: $SAMPLE_ID"
echo "Blacklist: $BLACKLIST"
echo "Output directory: $OUTPUT_DIR"
echo "Half-width: $HALF_WIDTH"
echo "Tn5 shifts: +$POS_SHIFT / $NEG_SHIFT"
echo "Threads: $THREADS"
echo "=================================="

# Change to output directory
cd "$OUTPUT_DIR"

# Create temporary files
TEMP_FRAGMENTS="${SAMPLE_ID}.fragments.bed"
PEAK_STATS="${SAMPLE_ID}.peak_stats.txt"

echo "Step 1: Preparing fragments file..."
# Extract fragments if gzipped
if [[ "$INPUT_FRAGMENTS" =~ \.gz$ ]]; then
    zcat "$INPUT_FRAGMENTS" > "$TEMP_FRAGMENTS"
else
    cp "$INPUT_FRAGMENTS" "$TEMP_FRAGMENTS"
fi

# Count fragments for normalization and q-value calculation
fragment_count=$(wc -l < "$TEMP_FRAGMENTS")
echo "  Total fragments: $fragment_count"

# Auto-calculate q-value if not provided
if [[ -z "$Q_VALUE" ]]; then
    if [[ $fragment_count -lt 5000000 ]]; then
        Q_VALUE=0.01
    elif [[ $fragment_count -lt 25000000 ]]; then
        Q_VALUE=0.005
    elif [[ $fragment_count -lt 50000000 ]]; then
        Q_VALUE=0.0025
    elif [[ $fragment_count -lt 100000000 ]]; then
        Q_VALUE=0.001
    else
        Q_VALUE=0.0005
    fi
fi

echo "  Using q-value: $Q_VALUE"

echo "Step 2: Converting fragments to Tn5 insertion sites..."
# Create forward and reverse strand insertion sites
awk -v pos_shift="$POS_SHIFT" 'BEGIN{OFS="\t"} {print $1, $2+pos_shift, $2+pos_shift+1}' "$TEMP_FRAGMENTS" > fwd.bed
awk -v neg_shift="$NEG_SHIFT" 'BEGIN{OFS="\t"} {print $1, $3-neg_shift, $3-neg_shift+1}' "$TEMP_FRAGMENTS" > rev.bed

# Combine and shuffle for pseudo-replicate splitting
cat fwd.bed rev.bed | sort -k1,1 -k2,2n | shuf > "${SAMPLE_ID}.bed"

echo "Step 3: Creating pseudo-replicates..."
total_lines=$(wc -l < "${SAMPLE_ID}.bed")
split_point=$((total_lines / 2))

head -n "$split_point" "${SAMPLE_ID}.bed" > "${SAMPLE_ID}_subset1.bed"
tail -n +"$((split_point + 1))" "${SAMPLE_ID}.bed" > "${SAMPLE_ID}_subset2.bed"

echo "  Pseudo-replicate 1: $split_point sites"
echo "  Pseudo-replicate 2: $((total_lines - split_point)) sites"

echo "Step 4: Running MACS2 peak calling..."
# Run MACS2 on full dataset and pseudo-replicates
echo "  Full dataset..."
macs2 callpeak -t "${SAMPLE_ID}.bed" -f BED -g hs \
    --shift -75 --extsize 150 --nomodel --call-summits --nolambda \
    --keep-dup all -q "$Q_VALUE" -n "$SAMPLE_ID" 2>/dev/null

echo "  Pseudo-replicate 1..."
macs2 callpeak -t "${SAMPLE_ID}_subset1.bed" -f BED -g hs \
    --shift -75 --extsize 150 --nomodel --call-summits --nolambda \
    --keep-dup all -q "$Q_VALUE" -n "${SAMPLE_ID}_subset1" 2>/dev/null

echo "  Pseudo-replicate 2..."
macs2 callpeak -t "${SAMPLE_ID}_subset2.bed" -f BED -g hs \
    --shift -75 --extsize 150 --nomodel --call-summits --nolambda \
    --keep-dup all -q "$Q_VALUE" -n "${SAMPLE_ID}_subset2" 2>/dev/null

echo "Step 5: Processing and filtering peaks..."
# Process each peak file: filter chromosomes, blacklist, and center peaks
process_peaks() {
    local input_peaks="$1"
    local output_peaks="$2"
    
    cat "$input_peaks" | \
    grep 'chr' | \
    grep -v 'chrKI\|chrGL\|chrEBV\|chrM\|chrMT\|_KI\|_GL' | \
    bedtools intersect -a - -b "$BLACKLIST" -v -wa | \
    awk -v hw="$HALF_WIDTH" 'BEGIN{OFS="\t"} {print $1, $2+$10-hw, $2+$10+hw, $4, $9, $10}' | \
    sort -k1,1 -k2,2n > "$output_peaks"
}

process_peaks "${SAMPLE_ID}_peaks.narrowPeak" "peak_file_overall.bed"
process_peaks "${SAMPLE_ID}_subset1_peaks.narrowPeak" "peak_pr1.bed"
process_peaks "${SAMPLE_ID}_subset2_peaks.narrowPeak" "peak_pr2.bed"

echo "Step 6: Finding reproducible peaks..."
# Find peaks that intersect between full dataset and both pseudo-replicates
bedtools intersect -a peak_file_overall.bed -b peak_pr1.bed -wa -f 0.50 -u > peak_intersect_pr1.bed
bedtools intersect -a peak_intersect_pr1.bed -b peak_pr2.bed -wa -f 0.50 -u | gzip > "${SAMPLE_ID}.pr.temp.peaks.bed.gz"

echo "Step 7: Merging overlapping peaks..."
# Create peak merging script
cat > peaks_merge_script.py << 'EOF'
import subprocess
import os
import sys

peak_extend_command = "zcat " + sys.argv[1] + ''' | sort -k1,1 -k2,2n | bedtools cluster -i - > extended.bed''' 
subprocess.call(peak_extend_command, shell=True)

command = '''awk '{count[$NF]++} END {for (id in count) print id,count[id]}' extended.bed > cluster_counts.txt'''
subprocess.call(command, shell=True)

command = '''awk -v OFS="\t" 'NR==FNR {counts[$1]=$2; next} {$NF = $NF OFS counts[$NF]; print}' cluster_counts.txt extended.bed | awk '{OFS="\t"}{print $1,$2,$3,$4,$5,$6,$7,$8}' > extended_with_counts.bed'''
subprocess.call(command, shell=True)

subprocess.call('''awk '$8 == 1' extended_with_counts.bed > singletons.bed''', shell=True)
subprocess.call('''awk '$8 > 1' extended_with_counts.bed > multiples.bed''', shell=True)
subprocess.call('''awk '$2 > 1' cluster_counts.txt | awk '{print $1}' > multiple_clusters.txt''', shell=True)

clusters = []
with open('multiple_clusters.txt', 'r') as file:
    clusters = [line.strip() for line in file]
    
multiple_out_file = open('multiples_collapsed.bed', 'a')

for k, cluster in enumerate(clusters):
    subprocess.call("awk '$7 ==" + str(cluster) + "' multiples.bed > temp_cluster.bed", shell=True)
    subprocess.call("sort -k5,5nr temp_cluster.bed > temp_cluster.sorted.bed", shell=True)
    
    with open('temp_cluster.sorted.bed', 'r') as peaks_file:
        peaks = peaks_file.readlines()
    
    kept_peaks = []
    while len(peaks) > 0:
        peak = peaks.pop(0)
        kept_peaks.append(peak)
        
        with open('temp_peak_file', 'w') as temp_peak_file:
            temp_peak_file.write(peak)

        with open('temp_file', 'w') as temp_file:
            for temp_peak in peaks:
                temp_file.write(temp_peak)
        
        intersect_command = "bedtools intersect -a temp_file -b temp_peak_file -wa > temp_overlap"
        subprocess.call(intersect_command, shell=True)
        
        with open('temp_overlap', 'r') as overlapping_peaks_file:
            overlapping_peaks = overlapping_peaks_file.readlines()

        peaks = [p for p in peaks if p not in overlapping_peaks]

    for line in kept_peaks:
        multiple_out_file.write(line)
    
    if k % 100 == 0:
        print('on cluster ' + str(k) + ' of ' + str(len(clusters)))

multiple_out_file.close()
subprocess.call('cat multiples_collapsed.bed singletons.bed > all_peaks_collapsed.bed', shell=True)
EOF

# Run peak merging
python3 peaks_merge_script.py "${SAMPLE_ID}.pr.temp.peaks.bed.gz"
gzip all_peaks_collapsed.bed
mv all_peaks_collapsed.bed.gz "${SAMPLE_ID}.pr.peaks.bed.gz"

# Count final peaks
final_peak_count=$(zcat "${SAMPLE_ID}.pr.peaks.bed.gz" | wc -l)

echo "Step 8: Generating statistics..."
# Create peak statistics file
cat > "$PEAK_STATS" << EOF
Sample ID: $SAMPLE_ID
Input fragments: $fragment_count
Q-value used: $Q_VALUE
Half-width: $HALF_WIDTH
Tn5 shifts: +$POS_SHIFT / $NEG_SHIFT

Peak counts:
- Full dataset: $(wc -l < peak_file_overall.bed)
- Pseudo-replicate 1: $(wc -l < peak_pr1.bed)
- Pseudo-replicate 2: $(wc -l < peak_pr2.bed)
- Intersect with PR1: $(wc -l < peak_intersect_pr1.bed)
- Final reproducible peaks: $final_peak_count
EOF

# Clean up temporary files
rm -f "$TEMP_FRAGMENTS" fwd.bed rev.bed "${SAMPLE_ID}.bed" "${SAMPLE_ID}_subset"*.bed
rm -f peak_file_overall.bed peak_pr1.bed peak_pr2.bed peak_intersect_pr1.bed
rm -f "${SAMPLE_ID}.pr.temp.peaks.bed.gz"
rm -f extended.bed cluster_counts.txt extended_with_counts.bed singletons.bed multiples.bed
rm -f multiples_collapsed.bed multiple_clusters.txt temp_cluster* temp_peak_file temp_file temp_overlap
rm -f peaks_merge_script.py

echo "Step 9: Validating output..."
if [[ -f "${SAMPLE_ID}.pr.peaks.bed.gz" && -f "$PEAK_STATS" ]]; then
    peak_file_size=$(du -h "${SAMPLE_ID}.pr.peaks.bed.gz" | cut -f1)
    
    echo "✓ Successfully completed peak calling"
    echo "  Final peaks: ${SAMPLE_ID}.pr.peaks.bed.gz ($peak_file_size)"
    echo "  Final peak count: $final_peak_count"
    echo "  Statistics: $PEAK_STATS"
    
    # Show sample peaks
    echo "  Sample peaks (first 3 lines):"
    zcat "${SAMPLE_ID}.pr.peaks.bed.gz" | head -3 | while read line; do
        echo "    $line"
    done
else
    echo "✗ Error: Output files not created" >&2
    exit 1
fi

echo "=== MACS2 Peak Calling Complete ==="