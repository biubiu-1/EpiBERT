#!/bin/bash

# EpiBERT Master Workflow Script
# Complete end-to-end pipeline from environment setup to model evaluation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Default values
IMPLEMENTATION="lightning"
SKIP_SETUP=false
SKIP_REFERENCES=false
SKIP_DATA_PROCESSING=false
SKIP_TRAINING=false
SKIP_EVALUATION=false
CONFIG_FILE=""
DATA_CONFIG=""
TRAINING_CONFIG=""
DRY_RUN=false
INTERACTIVE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data-config)
            DATA_CONFIG="$2"
            shift 2
            ;;
        --training-config)
            TRAINING_CONFIG="$2"
            shift 2
            ;;
        --lightning)
            IMPLEMENTATION="lightning"
            shift
            ;;
        --tensorflow)
            IMPLEMENTATION="tensorflow"
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-references)
            SKIP_REFERENCES=true
            shift
            ;;
        --skip-data-processing)
            SKIP_DATA_PROCESSING=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --non-interactive)
            INTERACTIVE=false
            shift
            ;;
        --help|-h)
            echo "EpiBERT Master Workflow Script"
            echo "Complete end-to-end pipeline from setup to evaluation"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config, -c FILE       Master configuration file"
            echo "  --data-config FILE      Data processing configuration"
            echo "  --training-config FILE  Training configuration"
            echo "  --lightning             Use PyTorch Lightning (default)"
            echo "  --tensorflow            Use TensorFlow implementation"
            echo "  --skip-setup            Skip environment setup"
            echo "  --skip-references       Skip reference data download"
            echo "  --skip-data-processing  Skip data processing"
            echo "  --skip-training         Skip model training"
            echo "  --skip-evaluation       Skip model evaluation"
            echo "  --dry-run               Show what would be done without executing"
            echo "  --non-interactive       Run without user prompts"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Example workflows:"
            echo "  # Complete workflow with Lightning"
            echo "  $0 --config master_config.yaml --lightning"
            echo ""
            echo "  # Data processing only"
            echo "  $0 --data-config data_config.yaml --skip-training --skip-evaluation"
            echo ""
            echo "  # Training only (data already processed)"
            echo "  $0 --training-config train_config.yaml --skip-data-processing"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ASCII Art Header
echo -e "${BLUE}"
cat << 'EOF'
 ______      _ ____  ______ _____ _______ 
|  ____|    (_)  _ \|  ____|  __ \__   __|
| |__   _ __  _| |_) | |__  | |__) | | |   
|  __| | '_ \| |  _ <|  __| |  _  /  | |   
| |____| |_) | | |_) | |____| | \ \  | |   
|______| .__/|_|____/|______|_|  \_\ |_|   
       | |                                
       |_|                                
EOF
echo -e "${NC}"

print_header "EpiBERT Master Workflow"
echo "Complete pipeline for EpiBERT training and evaluation"
echo "Implementation: $IMPLEMENTATION"
echo ""

# Function to ask user confirmation
ask_confirmation() {
    if [ "$INTERACTIVE" = false ]; then
        return 0
    fi
    
    local message="$1"
    echo -e "${YELLOW}$message${NC}"
    read -p "Continue? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_status "Workflow stopped by user."
        exit 0
    fi
}

# Function to create default configs if not provided
create_default_configs() {
    if [ -z "$DATA_CONFIG" ]; then
        DATA_CONFIG="workflow_data_config.yaml"
        if [ ! -f "$DATA_CONFIG" ]; then
            print_status "Creating default data processing configuration..."
            cat > "$DATA_CONFIG" << 'EOF'
# EpiBERT Data Processing Configuration
input:
  sample_name: "example_sample"
  atac_bam: "data/raw/sample.atac.bam"
  rampage_bam: "data/raw/sample.rampage.bam"  # optional

reference:
  genome_fasta: "reference/hg38.fa"
  chrom_sizes: "reference/hg38.chrom.sizes"
  blacklist: "reference/hg38-blacklist.v2.bed"
  motif_database: "reference/JASPAR2022_CORE_vertebrates.meme"

output:
  base_dir: "data/processed"
  
processing:
  peak_calling:
    qvalue: 0.01
  signal_tracks:
    bin_size: 128
    normalize: true
EOF
            print_status "Created: $DATA_CONFIG"
        fi
    fi
    
    if [ -z "$TRAINING_CONFIG" ]; then
        TRAINING_CONFIG="workflow_training_config.yaml"
        if [ ! -f "$TRAINING_CONFIG" ]; then
            print_status "Creating default training configuration..."
            cat > "$TRAINING_CONFIG" << 'EOF'
# EpiBERT Training Configuration
data:
  train_data: "data/processed/train"
  valid_data: "data/processed/valid"
  test_data: "data/processed/test"

model:
  type: "pretraining"
  input_length: 524288
  output_length: 4096

training:
  batch_size: 4
  learning_rate: 0.0001
  max_epochs: 100
  patience: 10

logging:
  wandb_project: "epibert"
  log_dir: "logs"

hardware:
  num_gpus: 1
  num_workers: 4
  precision: "bf16"
EOF
            print_status "Created: $TRAINING_CONFIG"
        fi
    fi
}

# Workflow steps
workflow_steps=(
    "Environment Setup"
    "Reference Data Download"
    "Data Processing"
    "Model Training"
    "Model Evaluation"
)

# Print workflow overview
print_header "Workflow Overview"
step_num=1
for step in "${workflow_steps[@]}"; do
    status="SCHEDULED"
    
    case $step_num in
        1) [ "$SKIP_SETUP" = true ] && status="SKIPPED" ;;
        2) [ "$SKIP_REFERENCES" = true ] && status="SKIPPED" ;;
        3) [ "$SKIP_DATA_PROCESSING" = true ] && status="SKIPPED" ;;
        4) [ "$SKIP_TRAINING" = true ] && status="SKIPPED" ;;
        5) [ "$SKIP_EVALUATION" = true ] && status="SKIPPED" ;;
    esac
    
    if [ "$status" = "SCHEDULED" ]; then
        echo -e "$step_num. ${GREEN}$step${NC} - $status"
    else
        echo -e "$step_num. ${YELLOW}$step${NC} - $status"
    fi
    step_num=$((step_num + 1))
done

echo ""

if [ "$DRY_RUN" = true ]; then
    print_header "Dry Run Mode"
    print_status "This is a dry run. No commands will be executed."
    echo ""
fi

# Create default configs
create_default_configs

# Step 1: Environment Setup
if [ "$SKIP_SETUP" = false ]; then
    print_step "Step 1: Environment Setup"
    
    ask_confirmation "Set up EpiBERT environment with dependencies?"
    
    if [ "$DRY_RUN" = false ]; then
        print_status "Running environment setup..."
        if [ "$IMPLEMENTATION" = "lightning" ]; then
            ./setup_environment.sh --lightning --install-deps
        else
            ./setup_environment.sh --tensorflow --install-deps
        fi
        print_status "✓ Environment setup complete"
    else
        print_status "Would run: ./setup_environment.sh --${IMPLEMENTATION} --install-deps"
    fi
    echo ""
fi

# Step 2: Reference Data Download
if [ "$SKIP_REFERENCES" = false ]; then
    print_step "Step 2: Reference Data Download"
    
    ask_confirmation "Download reference genome and annotation files?"
    
    if [ "$DRY_RUN" = false ]; then
        print_status "Downloading reference data..."
        ./scripts/download_references.sh --output-dir reference
        print_status "✓ Reference data download complete"
    else
        print_status "Would run: ./scripts/download_references.sh --output-dir reference"
    fi
    echo ""
fi

# Step 3: Data Processing
if [ "$SKIP_DATA_PROCESSING" = false ]; then
    print_step "Step 3: Data Processing"
    
    print_status "Using data configuration: $DATA_CONFIG"
    ask_confirmation "Process raw genomic data into training format?"
    
    if [ "$DRY_RUN" = false ]; then
        print_status "Running data processing pipeline..."
        ./data_processing/run_pipeline.sh -c "$DATA_CONFIG"
        print_status "✓ Data processing complete"
    else
        print_status "Would run: ./data_processing/run_pipeline.sh -c $DATA_CONFIG"
    fi
    echo ""
fi

# Step 4: Model Training
if [ "$SKIP_TRAINING" = false ]; then
    print_step "Step 4: Model Training"
    
    print_status "Using training configuration: $TRAINING_CONFIG"
    ask_confirmation "Start model training?"
    
    if [ "$DRY_RUN" = false ]; then
        print_status "Starting model training..."
        if [ "$IMPLEMENTATION" = "lightning" ]; then
            ./scripts/train_model.sh --config "$TRAINING_CONFIG" --lightning
        else
            ./scripts/train_model.sh --config "$TRAINING_CONFIG" --tensorflow
        fi
        print_status "✓ Model training complete"
    else
        print_status "Would run: ./scripts/train_model.sh --config $TRAINING_CONFIG --$IMPLEMENTATION"
    fi
    echo ""
fi

# Step 5: Model Evaluation
if [ "$SKIP_EVALUATION" = false ]; then
    print_step "Step 5: Model Evaluation"
    
    ask_confirmation "Evaluate trained model performance?"
    
    if [ "$DRY_RUN" = false ]; then
        # Find the most recent model checkpoint
        MODEL_DIR="models/checkpoints"
        if [ "$IMPLEMENTATION" = "lightning" ]; then
            LATEST_MODEL=$(find "$MODEL_DIR" -name "*.ckpt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        else
            LATEST_MODEL=$(find "$MODEL_DIR" -name "*.h5" -o -name "checkpoint*" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        fi
        
        if [ -n "$LATEST_MODEL" ] && [ -f "$LATEST_MODEL" ]; then
            print_status "Found trained model: $LATEST_MODEL"
            
            # Get test data path from training config
            TEST_DATA=$(grep "test_data:" "$TRAINING_CONFIG" | sed 's/.*:[[:space:]]*//' | tr -d '"' | tr -d "'")
            
            if [ -n "$TEST_DATA" ] && [ -e "$TEST_DATA" ]; then
                print_status "Running model evaluation..."
                python3 scripts/evaluate_model.py \
                    --model_path "$LATEST_MODEL" \
                    --test_data "$TEST_DATA" \
                    --implementation "$IMPLEMENTATION" \
                    --output_dir "results/evaluation"
                print_status "✓ Model evaluation complete"
            else
                print_warning "Test data not found: $TEST_DATA"
                print_warning "Skipping evaluation"
            fi
        else
            print_warning "No trained model found in $MODEL_DIR"
            print_warning "Skipping evaluation"
        fi
    else
        print_status "Would run: python3 scripts/evaluate_model.py --model_path <latest_model> --test_data <test_data> --implementation $IMPLEMENTATION"
    fi
    echo ""
fi

# Workflow Summary
print_header "Workflow Summary"

if [ "$DRY_RUN" = false ]; then
    print_status "✅ EpiBERT workflow completed successfully!"
    
    # Print results summary
    echo ""
    print_status "Results Location:"
    print_status "  - Processed data: data/processed/"
    print_status "  - Trained models: models/checkpoints/"
    print_status "  - Training logs: logs/"
    print_status "  - Evaluation results: results/evaluation/"
    print_status "  - Reference data: reference/"
    
    echo ""
    print_status "Next Steps:"
    print_status "  1. Review evaluation metrics in results/evaluation/"
    print_status "  2. Use trained model for predictions"
    print_status "  3. Fine-tune on your specific data"
    print_status "  4. Analyze model attributions and interpretability"
    
    # Check for Jupyter notebooks
    if [ -d "example_usage" ]; then
        echo ""
        print_status "📓 Explore example notebooks:"
        find example_usage -name "*.ipynb" | while read notebook; do
            print_status "     - $notebook"
        done
    fi
    
else
    print_status "Dry run completed. Use without --dry-run to execute the workflow."
fi

# Configuration files summary
echo ""
print_status "Configuration Files Used:"
print_status "  - Data processing: $DATA_CONFIG"
print_status "  - Training: $TRAINING_CONFIG"

if [ "$INTERACTIVE" = true ] && [ "$DRY_RUN" = false ]; then
    echo ""
    ask_confirmation "Open results directory?"
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        if command -v open >/dev/null 2>&1; then
            open results/
        elif command -v xdg-open >/dev/null 2>&1; then
            xdg-open results/
        else
            print_status "Results directory: $(pwd)/results/"
        fi
    fi
fi

print_header "Workflow Complete"