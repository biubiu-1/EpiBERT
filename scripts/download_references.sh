#!/bin/bash

# EpiBERT Reference Data Download Script
# Downloads essential reference files needed for EpiBERT data processing

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Default values
OUTPUT_DIR="reference"
GENOME="hg38"
FORCE_DOWNLOAD=false
DOWNLOAD_ALL=true
DOWNLOAD_GENOME=false
DOWNLOAD_BLACKLIST=false
DOWNLOAD_MOTIFS=false
DOWNLOAD_CHROM_SIZES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --genome|-g)
            GENOME="$2"
            shift 2
            ;;
        --force|-f)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --genome-only)
            DOWNLOAD_ALL=false
            DOWNLOAD_GENOME=true
            shift
            ;;
        --blacklist-only)
            DOWNLOAD_ALL=false
            DOWNLOAD_BLACKLIST=true
            shift
            ;;
        --motifs-only)
            DOWNLOAD_ALL=false
            DOWNLOAD_MOTIFS=true
            shift
            ;;
        --chrom-sizes-only)
            DOWNLOAD_ALL=false
            DOWNLOAD_CHROM_SIZES=true
            shift
            ;;
        --help|-h)
            echo "EpiBERT Reference Data Download Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir, -o DIR    Output directory for reference files (default: reference)"
            echo "  --genome, -g GENOME     Genome assembly (default: hg38, also supports hg19)"
            echo "  --force, -f             Force re-download even if files exist"
            echo "  --genome-only           Download only genome FASTA file"
            echo "  --blacklist-only        Download only blacklist regions"
            echo "  --motifs-only           Download only motif databases"
            echo "  --chrom-sizes-only      Download only chromosome sizes file"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "By default, downloads all reference files needed for EpiBERT."
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "EpiBERT Reference Data Download"
print_status "Genome: $GENOME"
print_status "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for required tools
if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
    print_error "Neither wget nor curl is available. Please install one of them."
    exit 1
fi

# Determine download command
if command -v wget >/dev/null 2>&1; then
    DOWNLOAD_CMD="wget -O"
    DOWNLOAD_SIMPLE="wget"
else
    DOWNLOAD_CMD="curl -L -o"
    DOWNLOAD_SIMPLE="curl -L"
fi

# Function to download file with retry
download_file() {
    local url="$1"
    local output="$2"
    local description="$3"
    
    print_status "Downloading $description..."
    print_status "URL: $url"
    print_status "Output: $output"
    
    if [ -f "$output" ] && [ "$FORCE_DOWNLOAD" = false ]; then
        print_warning "File already exists: $output (use --force to re-download)"
        return 0
    fi
    
    # Create directory if needed
    mkdir -p "$(dirname "$output")"
    
    # Download with retry
    local retries=3
    local count=0
    
    while [ $count -lt $retries ]; do
        if command -v wget >/dev/null 2>&1; then
            if wget -O "$output" "$url"; then
                print_status "✓ Downloaded $description"
                return 0
            fi
        else
            if curl -L -o "$output" "$url"; then
                print_status "✓ Downloaded $description"
                return 0
            fi
        fi
        
        count=$((count + 1))
        if [ $count -lt $retries ]; then
            print_warning "Download failed, retrying in 5 seconds... (attempt $((count + 1))/$retries)"
            sleep 5
        fi
    done
    
    print_error "Failed to download $description after $retries attempts"
    return 1
}

# Function to download and decompress if needed
download_and_decompress() {
    local url="$1"
    local output="$2"
    local description="$3"
    
    if [[ "$url" == *.gz ]]; then
        local temp_file="${output}.gz"
        if download_file "$url" "$temp_file" "$description (compressed)"; then
            print_status "Decompressing $description..."
            if command -v gunzip >/dev/null 2>&1; then
                gunzip -f "$temp_file"
            else
                print_error "gunzip not available for decompression"
                return 1
            fi
        else
            return 1
        fi
    else
        download_file "$url" "$output" "$description"
    fi
}

# Download chromosome sizes
if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_CHROM_SIZES" = true ]; then
    print_header "Downloading Chromosome Sizes"
    
    if [ "$GENOME" = "hg38" ]; then
        CHROM_SIZES_URL="https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes"
    elif [ "$GENOME" = "hg19" ]; then
        CHROM_SIZES_URL="https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.chrom.sizes"
    else
        print_error "Unsupported genome: $GENOME"
        exit 1
    fi
    
    download_file "$CHROM_SIZES_URL" "$OUTPUT_DIR/${GENOME}.chrom.sizes" "chromosome sizes"
fi

# Download blacklist regions
if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_BLACKLIST" = true ]; then
    print_header "Downloading Blacklist Regions"
    
    if [ "$GENOME" = "hg38" ]; then
        BLACKLIST_URL="https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg38-blacklist.v2.bed.gz"
    elif [ "$GENOME" = "hg19" ]; then
        BLACKLIST_URL="https://github.com/Boyle-Lab/Blacklist/raw/master/lists/hg19-blacklist.v2.bed.gz"
    else
        print_error "Unsupported genome: $GENOME"
        exit 1
    fi
    
    download_and_decompress "$BLACKLIST_URL" "$OUTPUT_DIR/${GENOME}-blacklist.v2.bed" "ENCODE blacklist regions"
fi

# Download genome FASTA
if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_GENOME" = true ]; then
    print_header "Downloading Genome FASTA"
    
    if [ "$GENOME" = "hg38" ]; then
        GENOME_URL="https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.fa.gz"
    elif [ "$GENOME" = "hg19" ]; then
        GENOME_URL="https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz"
    else
        print_error "Unsupported genome: $GENOME"
        exit 1
    fi
    
    print_warning "Genome FASTA is very large (~3GB compressed, ~3GB uncompressed)"
    read -p "Do you want to download the genome FASTA? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_and_decompress "$GENOME_URL" "$OUTPUT_DIR/${GENOME}.fa" "genome FASTA"
        
        # Index with samtools if available
        if command -v samtools >/dev/null 2>&1; then
            print_status "Indexing genome FASTA with samtools..."
            samtools faidx "$OUTPUT_DIR/${GENOME}.fa"
            print_status "✓ Genome FASTA indexed"
        else
            print_warning "samtools not available - genome FASTA not indexed"
        fi
    else
        print_status "Skipping genome FASTA download"
    fi
fi

# Download motif databases
if [ "$DOWNLOAD_ALL" = true ] || [ "$DOWNLOAD_MOTIFS" = true ]; then
    print_header "Downloading Motif Databases"
    
    # JASPAR CORE vertebrates database
    JASPAR_URL="https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt"
    download_file "$JASPAR_URL" "$OUTPUT_DIR/JASPAR2022_CORE_vertebrates.meme" "JASPAR 2022 CORE vertebrates motifs"
    
    # Vierstra et al. 2020 consensus motifs (if available)
    VIERSTRA_URL="https://www.vierstra.org/resources/motif_clustering/motifs.tar.gz"
    if download_file "$VIERSTRA_URL" "$OUTPUT_DIR/vierstra_motifs.tar.gz" "Vierstra et al. consensus motifs"; then
        print_status "Extracting Vierstra motifs..."
        cd "$OUTPUT_DIR"
        tar -xzf vierstra_motifs.tar.gz
        cd - >/dev/null
        print_status "✓ Vierstra motifs extracted"
    fi
fi

# Download additional useful files
if [ "$DOWNLOAD_ALL" = true ]; then
    print_header "Downloading Additional Reference Files"
    
    # Download gene annotations
    if [ "$GENOME" = "hg38" ]; then
        GTF_URL="https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/genes/hg38.refGene.gtf.gz"
    elif [ "$GENOME" = "hg19" ]; then
        GTF_URL="https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/genes/hg19.refGene.gtf.gz"
    fi
    
    download_and_decompress "$GTF_URL" "$OUTPUT_DIR/${GENOME}.refGene.gtf" "RefSeq gene annotations"
    
    # Download CpG islands
    if [ "$GENOME" = "hg38" ]; then
        CPGI_URL="https://hgdownload.cse.ucsc.edu/goldenpath/hg38/database/cpgIslandExt.txt.gz"
    elif [ "$GENOME" = "hg19" ]; then
        CPGI_URL="https://hgdownload.cse.ucsc.edu/goldenpath/hg19/database/cpgIslandExt.txt.gz"
    fi
    
    download_and_decompress "$CPGI_URL" "$OUTPUT_DIR/${GENOME}.cpgIslandExt.txt" "CpG islands"
fi

# Create summary file
print_header "Creating Reference Summary"

SUMMARY_FILE="$OUTPUT_DIR/reference_summary.txt"
cat > "$SUMMARY_FILE" << EOF
EpiBERT Reference Files Summary
===============================
Generated on: $(date)
Genome assembly: $GENOME
Download directory: $OUTPUT_DIR

Files downloaded:
EOF

# List downloaded files
find "$OUTPUT_DIR" -type f -not -name "reference_summary.txt" | sort | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo "  $(basename "$file") ($size)" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" << EOF

Usage Notes:
-----------
1. Genome FASTA: Use for sequence extraction and motif analysis
2. Chromosome sizes: Required for bedtools and signal track generation
3. Blacklist regions: Filter out problematic genomic regions
4. GTF annotations: Gene/transcript coordinates for analysis
5. Motif databases: For motif enrichment analysis

Example usage in data processing:
  ./data_processing/run_pipeline.sh \\
    --genome-fasta $OUTPUT_DIR/${GENOME}.fa \\
    --chrom-sizes $OUTPUT_DIR/${GENOME}.chrom.sizes \\
    --blacklist $OUTPUT_DIR/${GENOME}-blacklist.v2.bed \\
    --motif-db $OUTPUT_DIR/JASPAR2022_CORE_vertebrates.meme

For more information, see the data processing documentation:
  data_processing/README.md
EOF

print_status "Reference summary written to: $SUMMARY_FILE"

# Validate downloads
print_header "Validating Downloads"

ERROR_COUNT=0

# Check essential files
ESSENTIAL_FILES=(
    "${GENOME}.chrom.sizes"
    "${GENOME}-blacklist.v2.bed"
)

for file in "${ESSENTIAL_FILES[@]}"; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        size=$(stat -c%s "$OUTPUT_DIR/$file" 2>/dev/null || stat -f%z "$OUTPUT_DIR/$file" 2>/dev/null || echo "0")
        if [ "$size" -gt 100 ]; then
            print_status "✓ $file ($(du -h "$OUTPUT_DIR/$file" | cut -f1))"
        else
            print_error "✗ $file is too small or empty"
            ERROR_COUNT=$((ERROR_COUNT + 1))
        fi
    else
        print_warning "✗ $file not found"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    fi
done

# Check optional files
OPTIONAL_FILES=(
    "${GENOME}.fa"
    "JASPAR2022_CORE_vertebrates.meme"
    "${GENOME}.refGene.gtf"
)

for file in "${OPTIONAL_FILES[@]}"; do
    if [ -f "$OUTPUT_DIR/$file" ]; then
        print_status "✓ $file (optional)"
    else
        print_status "- $file (optional, not downloaded)"
    fi
done

print_header "Download Summary"

if [ $ERROR_COUNT -eq 0 ]; then
    print_status "✅ All essential reference files downloaded successfully!"
    print_status "Files are ready for use with EpiBERT data processing pipeline."
else
    print_warning "⚠️  $ERROR_COUNT essential files missing or incomplete"
    print_warning "Some data processing steps may fail without these files."
fi

print_status ""
print_status "Reference files location: $OUTPUT_DIR"
print_status "Summary file: $SUMMARY_FILE"
print_status ""
print_status "Next steps:"
print_status "1. Update your data processing configuration to use these reference files"
print_status "2. See data_processing/README.md for usage instructions"
print_status "3. Run data processing pipeline: ./data_processing/run_pipeline.sh"

print_header "Download Complete"