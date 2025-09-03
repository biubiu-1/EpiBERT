#!/bin/bash

# EpiBERT Environment Setup Script
# This script sets up the complete environment for EpiBERT including dependencies,
# validation of tools, and initial configuration.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python package
check_python_package() {
    python3 -c "import $1" 2>/dev/null
}

# Parse command line arguments
IMPLEMENTATION="tensorflow"  # Default to TensorFlow
INSTALL_DEPS=false
SETUP_CONDA=false
VALIDATE_ONLY=false
SETUP_LIGHTNING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --lightning)
            IMPLEMENTATION="lightning"
            SETUP_LIGHTNING=true
            shift
            ;;
        --tensorflow)
            IMPLEMENTATION="tensorflow"
            shift
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        --setup-conda)
            SETUP_CONDA=true
            shift
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --help|-h)
            echo "EpiBERT Environment Setup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --lightning        Setup for PyTorch Lightning implementation"
            echo "  --tensorflow       Setup for TensorFlow implementation (default)"
            echo "  --install-deps     Install missing dependencies automatically"
            echo "  --setup-conda      Create and activate conda environment"
            echo "  --validate-only    Only validate environment, don't install anything"
            echo "  --help, -h         Show this help message"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "EpiBERT Environment Setup"
print_status "Setting up environment for $IMPLEMENTATION implementation"

# Check Python version
print_status "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    print_status "Found Python $PYTHON_VERSION"
    
    # Check if Python version is >= 3.7
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)"; then
        print_status "Python version is compatible"
    else
        print_error "Python 3.7+ is required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    exit 1
fi

# Setup conda environment if requested
if [ "$SETUP_CONDA" = true ]; then
    print_status "Setting up conda environment..."
    
    if ! command_exists conda; then
        print_error "Conda is not installed. Please install Miniconda or Anaconda first."
        exit 1
    fi
    
    ENV_NAME="epibert"
    if [ "$IMPLEMENTATION" = "lightning" ]; then
        ENV_NAME="epibert-lightning"
    fi
    
    print_status "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.9 -y
    
    print_status "Activating conda environment: $ENV_NAME"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    
    print_status "Conda environment $ENV_NAME created and activated"
fi

# Install dependencies
if [ "$INSTALL_DEPS" = true ] && [ "$VALIDATE_ONLY" = false ]; then
    print_status "Installing Python dependencies..."
    
    if [ "$IMPLEMENTATION" = "lightning" ]; then
        print_status "Installing PyTorch Lightning dependencies..."
        pip install -r lightning_transfer/requirements_lightning.txt
    else
        print_status "Installing TensorFlow dependencies..."
        pip install -r requirements.txt
    fi
    
    print_status "Installing EpiBERT package..."
    pip install -e .
fi

# Validate core Python packages
print_status "Validating Python packages..."

REQUIRED_PACKAGES_TF=("tensorflow" "numpy" "pandas" "matplotlib" "seaborn" "scikit-learn" "scipy" "h5py" "pysam" "pybedtools" "wandb")
REQUIRED_PACKAGES_LIGHTNING=("torch" "pytorch_lightning" "torchmetrics" "numpy" "pandas" "matplotlib" "seaborn" "scikit-learn" "scipy" "h5py" "pysam" "pybedtools" "wandb")

if [ "$IMPLEMENTATION" = "lightning" ]; then
    REQUIRED_PACKAGES=("${REQUIRED_PACKAGES_LIGHTNING[@]}")
else
    REQUIRED_PACKAGES=("${REQUIRED_PACKAGES_TF[@]}")
fi

MISSING_PACKAGES=()
for package in "${REQUIRED_PACKAGES[@]}"; do
    if check_python_package "$package"; then
        print_status "✓ $package"
    else
        print_warning "✗ $package (missing)"
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_warning "Missing Python packages: ${MISSING_PACKAGES[*]}"
    if [ "$INSTALL_DEPS" = false ]; then
        print_warning "Use --install-deps to install missing packages automatically"
    fi
fi

# Check bioinformatics tools
print_status "Checking bioinformatics tools..."

BIOINF_TOOLS=("samtools" "bedtools" "bgzip" "tabix")
OPTIONAL_TOOLS=("macs2" "sra-toolkit" "pigz" "wget" "curl")

MISSING_TOOLS=()
for tool in "${BIOINF_TOOLS[@]}"; do
    if command_exists "$tool"; then
        VERSION=$(eval "${tool} --version 2>&1 | head -n1" || echo "unknown")
        print_status "✓ $tool ($VERSION)"
    else
        print_warning "✗ $tool (missing - required for data processing)"
        MISSING_TOOLS+=("$tool")
    fi
done

MISSING_OPTIONAL=()
for tool in "${OPTIONAL_TOOLS[@]}"; do
    if command_exists "$tool"; then
        print_status "✓ $tool (optional)"
    else
        print_warning "✗ $tool (optional - recommended for data processing)"
        MISSING_OPTIONAL+=("$tool")
    fi
done

# GPU/TPU detection
print_status "Checking compute resources..."

if [ "$IMPLEMENTATION" = "lightning" ]; then
    # Check for CUDA/GPU
    if command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        print_status "✓ Found $GPU_COUNT GPU(s)"
        
        # Check PyTorch CUDA support
        if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
            print_status "✓ PyTorch CUDA support available"
        else
            print_warning "PyTorch CUDA support not available"
        fi
    else
        print_warning "No GPU detected (will use CPU)"
    fi
else
    # Check for TPU
    if [ -n "$TPU_NAME" ]; then
        print_status "✓ TPU environment detected: $TPU_NAME"
    elif command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        print_status "✓ Found $GPU_COUNT GPU(s)"
    else
        print_warning "No GPU/TPU detected (will use CPU)"
    fi
fi

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p results/evaluation
mkdir -p logs
mkdir -p tmp

print_status "✓ Directory structure created"

# Create example config files
print_status "Creating example configuration files..."

# Create example training config
cat > example_training_config.yaml << 'EOF'
# EpiBERT Training Configuration Example
# Copy and modify this file for your training runs

# Data paths
data:
  train_data: "data/processed/train"
  valid_data: "data/processed/valid"
  test_data: "data/processed/test"
  
# Model configuration
model:
  type: "pretraining"  # or "finetuning"
  input_length: 524288
  output_length: 4096
  final_output_length: 4092
  
# Training parameters
training:
  batch_size: 4
  learning_rate: 0.0001
  max_epochs: 100
  patience: 10
  
# Logging
logging:
  wandb_project: "epibert"
  wandb_entity: "your_username"
  log_dir: "logs"
  
# Hardware
hardware:
  num_gpus: 1
  num_workers: 4
  precision: "bf16"
EOF

print_status "✓ Created example_training_config.yaml"

# Create example data processing config
cat > example_data_config.yaml << 'EOF'
# EpiBERT Data Processing Configuration Example
# Copy and modify this file for your data processing

# Input data
input:
  sample_name: "example_sample"
  atac_bam: "data/raw/sample.atac.bam"
  rampage_bam: "data/raw/sample.rampage.bam"  # optional, for fine-tuning
  
# Reference files
reference:
  genome_fasta: "reference/hg38.fa"
  chrom_sizes: "reference/hg38.chrom.sizes"
  blacklist: "reference/blacklist.bed"
  motif_database: "reference/motifs.meme"
  
# Output directories
output:
  base_dir: "data/processed"
  fragments_dir: "fragments"
  signals_dir: "signals"
  peaks_dir: "peaks"
  motifs_dir: "motifs"
  
# Processing parameters
processing:
  peak_calling:
    qvalue: 0.01
    shift: -75
    extsize: 150
  
  signal_tracks:
    bin_size: 128
    normalize: true
    
  motif_analysis:
    pvalue_threshold: 0.001
EOF

print_status "✓ Created example_data_config.yaml"

# Validation summary
print_header "Environment Validation Summary"

if [ ${#MISSING_PACKAGES[@]} -eq 0 ] && [ ${#MISSING_TOOLS[@]} -eq 0 ]; then
    print_status "✅ Environment is ready for EpiBERT!"
    print_status "All required dependencies are installed."
    
    if [ ${#MISSING_OPTIONAL[@]} -gt 0 ]; then
        print_warning "Optional tools missing: ${MISSING_OPTIONAL[*]}"
        print_warning "These are recommended but not required for basic functionality."
    fi
else
    print_warning "⚠️  Environment setup incomplete"
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        print_error "Missing Python packages: ${MISSING_PACKAGES[*]}"
        print_status "Install with: pip install ${MISSING_PACKAGES[*]}"
    fi
    
    if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
        print_error "Missing bioinformatics tools: ${MISSING_TOOLS[*]}"
        print_status "Install via package manager (apt, conda, homebrew, etc.)"
    fi
fi

# Print next steps
print_header "Next Steps"
print_status "1. Review and customize configuration files:"
print_status "   - example_training_config.yaml"
print_status "   - example_data_config.yaml"
print_status ""
print_status "2. Download reference data:"
print_status "   ./scripts/download_references.sh"
print_status ""
print_status "3. Process your data:"
print_status "   ./data_processing/run_pipeline.sh -c example_data_config.yaml"
print_status ""
print_status "4. Train a model:"
if [ "$IMPLEMENTATION" = "lightning" ]; then
    print_status "   ./scripts/train_model.sh --config example_training_config.yaml --lightning"
else
    print_status "   ./scripts/train_model.sh --config example_training_config.yaml"
fi
print_status ""
print_status "5. Evaluate performance:"
print_status "   ./scripts/evaluate_model.sh --model models/checkpoints/best_model --test-data data/processed/test"
print_status ""
print_status "📖 See README.md for detailed documentation"

print_header "Setup Complete"