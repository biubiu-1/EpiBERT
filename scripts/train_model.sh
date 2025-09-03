#!/bin/bash

# EpiBERT Master Training Script
# Orchestrates the complete training workflow with proper configuration management

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

# Default values
IMPLEMENTATION="tensorflow"
CONFIG_FILE=""
MODEL_TYPE="pretraining"
RESUME_FROM=""
DRY_RUN=false
VALIDATE_DATA=true
OUTPUT_DIR="models/checkpoints"
LOG_DIR="logs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
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
        --model-type|-t)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --resume-from|-r)
            RESUME_FROM="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log-dir|-l)
            LOG_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-validate)
            VALIDATE_DATA=false
            shift
            ;;
        --help|-h)
            echo "EpiBERT Master Training Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config, -c FILE       Configuration file (YAML)"
            echo "  --lightning             Use PyTorch Lightning implementation"
            echo "  --tensorflow            Use TensorFlow implementation (default)"
            echo "  --model-type, -t TYPE   Model type: pretraining or finetuning (default: pretraining)"
            echo "  --resume-from, -r PATH  Resume training from checkpoint"
            echo "  --output-dir, -o DIR    Output directory for checkpoints (default: models/checkpoints)"
            echo "  --log-dir, -l DIR       Log directory (default: logs)"
            echo "  --dry-run               Validate setup without starting training"
            echo "  --no-validate           Skip data validation"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --config example_training_config.yaml --lightning"
            echo "  $0 --config my_config.yaml --model-type finetuning"
            echo "  $0 --config my_config.yaml --resume-from models/checkpoints/epoch_10.ckpt"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "EpiBERT Training Orchestration"
print_status "Implementation: $IMPLEMENTATION"
print_status "Model type: $MODEL_TYPE"

# Check required arguments
if [ -z "$CONFIG_FILE" ]; then
    print_error "Configuration file is required. Use --config to specify."
    print_status "Example: $0 --config example_training_config.yaml"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

print_status "Configuration file: $CONFIG_FILE"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Read configuration file (basic YAML parsing)
function get_config_value() {
    local key="$1"
    local default="$2"
    local value=$(grep "^[[:space:]]*${key}:" "$CONFIG_FILE" | sed 's/.*:[[:space:]]*//' | tr -d '"' | tr -d "'")
    if [ -z "$value" ]; then
        echo "$default"
    else
        echo "$value"
    fi
}

# Extract configuration values
TRAIN_DATA=$(get_config_value "train_data" "")
VALID_DATA=$(get_config_value "valid_data" "")
TEST_DATA=$(get_config_value "test_data" "")
BATCH_SIZE=$(get_config_value "batch_size" "4")
LEARNING_RATE=$(get_config_value "learning_rate" "0.0001")
MAX_EPOCHS=$(get_config_value "max_epochs" "100")
PATIENCE=$(get_config_value "patience" "10")
WANDB_PROJECT=$(get_config_value "wandb_project" "epibert")
WANDB_ENTITY=$(get_config_value "wandb_entity" "")
NUM_GPUS=$(get_config_value "num_gpus" "1")
NUM_WORKERS=$(get_config_value "num_workers" "4")
PRECISION=$(get_config_value "precision" "bf16")

print_status "Parsed configuration:"
print_status "  Train data: $TRAIN_DATA"
print_status "  Valid data: $VALID_DATA"
print_status "  Batch size: $BATCH_SIZE"
print_status "  Learning rate: $LEARNING_RATE"
print_status "  Max epochs: $MAX_EPOCHS"

# Validate data paths
if [ "$VALIDATE_DATA" = true ]; then
    print_status "Validating data paths..."
    
    if [ -n "$TRAIN_DATA" ] && [ ! -e "$TRAIN_DATA" ]; then
        print_error "Training data not found: $TRAIN_DATA"
        exit 1
    fi
    
    if [ -n "$VALID_DATA" ] && [ ! -e "$VALID_DATA" ]; then
        print_error "Validation data not found: $VALID_DATA"
        exit 1
    fi
    
    print_status "✓ Data paths validated"
fi

# Check dependencies
print_status "Checking dependencies..."

if [ "$IMPLEMENTATION" = "lightning" ]; then
    # Check PyTorch Lightning dependencies
    python3 -c "import torch, pytorch_lightning" 2>/dev/null || {
        print_error "PyTorch Lightning dependencies not available"
        print_status "Install with: pip install -r lightning_transfer/requirements_lightning.txt"
        exit 1
    }
    
    # Check GPU availability if requested
    if [ "$NUM_GPUS" != "0" ]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            print_warning "nvidia-smi not found, but GPUs requested"
        else
            GPU_COUNT=$(nvidia-smi -L | wc -l)
            print_status "Found $GPU_COUNT GPU(s)"
        fi
    fi
else
    # Check TensorFlow dependencies
    python3 -c "import tensorflow" 2>/dev/null || {
        print_error "TensorFlow dependencies not available"
        print_status "Install with: pip install -r requirements.txt"
        exit 1
    }
fi

print_status "✓ Dependencies validated"

# Generate training command
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${MODEL_TYPE}_${IMPLEMENTATION}_${TIMESTAMP}"
OUTPUT_PATH="${OUTPUT_DIR}/${RUN_NAME}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}"

mkdir -p "$OUTPUT_PATH"
mkdir -p "$LOG_PATH"

if [ "$IMPLEMENTATION" = "lightning" ]; then
    # PyTorch Lightning training command
    TRAIN_CMD="python3 lightning_transfer/train_lightning.py"
    TRAIN_CMD="$TRAIN_CMD --data_dir $TRAIN_DATA"
    TRAIN_CMD="$TRAIN_CMD --model_type $MODEL_TYPE"
    TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
    TRAIN_CMD="$TRAIN_CMD --learning_rate $LEARNING_RATE"
    TRAIN_CMD="$TRAIN_CMD --max_epochs $MAX_EPOCHS"
    TRAIN_CMD="$TRAIN_CMD --patience $PATIENCE"
    TRAIN_CMD="$TRAIN_CMD --output_dir $OUTPUT_PATH"
    TRAIN_CMD="$TRAIN_CMD --num_workers $NUM_WORKERS"
    
    if [ -n "$WANDB_PROJECT" ]; then
        TRAIN_CMD="$TRAIN_CMD --wandb_project $WANDB_PROJECT"
    fi
    
    if [ -n "$WANDB_ENTITY" ]; then
        TRAIN_CMD="$TRAIN_CMD --wandb_entity $WANDB_ENTITY"
    fi
    
    if [ -n "$RESUME_FROM" ]; then
        TRAIN_CMD="$TRAIN_CMD --resume_from_checkpoint $RESUME_FROM"
    fi
    
    # Add precision
    TRAIN_CMD="$TRAIN_CMD --precision $PRECISION"
    
    # Add GPU configuration
    if [ "$NUM_GPUS" != "0" ]; then
        TRAIN_CMD="$TRAIN_CMD --gpus $NUM_GPUS"
    fi
    
else
    # TensorFlow training command
    if [ "$MODEL_TYPE" = "pretraining" ]; then
        TRAIN_CMD="bash execute_pretraining.sh"
    else
        TRAIN_CMD="bash execute_finetuning.sh"
    fi
    
    # Export environment variables for TF training
    export TRAIN_DATA="$TRAIN_DATA"
    export VALID_DATA="$VALID_DATA"
    export BATCH_SIZE="$BATCH_SIZE"
    export LEARNING_RATE="$LEARNING_RATE"
    export MAX_EPOCHS="$MAX_EPOCHS"
    export OUTPUT_DIR="$OUTPUT_PATH"
    export WANDB_PROJECT="$WANDB_PROJECT"
    export WANDB_ENTITY="$WANDB_ENTITY"
fi

# Save configuration and command
echo "# Training Configuration" > "$LOG_PATH/training_config.txt"
echo "Timestamp: $TIMESTAMP" >> "$LOG_PATH/training_config.txt"
echo "Implementation: $IMPLEMENTATION" >> "$LOG_PATH/training_config.txt"
echo "Model type: $MODEL_TYPE" >> "$LOG_PATH/training_config.txt"
echo "Config file: $CONFIG_FILE" >> "$LOG_PATH/training_config.txt"
echo "Output path: $OUTPUT_PATH" >> "$LOG_PATH/training_config.txt"
echo "Log path: $LOG_PATH" >> "$LOG_PATH/training_config.txt"
echo "" >> "$LOG_PATH/training_config.txt"
echo "# Training Command" >> "$LOG_PATH/training_config.txt"
echo "$TRAIN_CMD" >> "$LOG_PATH/training_config.txt"

# Copy configuration file
cp "$CONFIG_FILE" "$LOG_PATH/config.yaml"

print_status "Training configuration saved to: $LOG_PATH/training_config.txt"

# Print command
print_header "Training Command"
echo "$TRAIN_CMD"

if [ "$DRY_RUN" = true ]; then
    print_header "Dry Run Complete"
    print_status "Setup validated successfully. Training command prepared."
    print_status "Run without --dry-run to start training."
    exit 0
fi

# Ask for confirmation
print_header "Ready to Start Training"
print_status "Run name: $RUN_NAME"
print_status "Output directory: $OUTPUT_PATH"
print_status "Log directory: $LOG_PATH"

read -p "Start training? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Training cancelled."
    exit 0
fi

# Start training
print_header "Starting Training"
print_status "Training logs will be saved to: $LOG_PATH/training.log"

# Create training script
TRAINING_SCRIPT="$LOG_PATH/run_training.sh"
cat > "$TRAINING_SCRIPT" << EOF
#!/bin/bash
# Generated training script for $RUN_NAME

set -e

# Change to repository root
cd "\$(dirname "\$0")/../.."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="\$PWD:\$PYTHONPATH"

# Training command
$TRAIN_CMD
EOF

chmod +x "$TRAINING_SCRIPT"

# Execute training with logging
{
    echo "Starting training at $(date)"
    echo "Command: $TRAIN_CMD"
    echo "Working directory: $(pwd)"
    echo "Environment:"
    env | grep -E "(CUDA|PYTHON|WANDB)" || true
    echo "----------------------------------------"
    
    # Run the training command
    eval "$TRAIN_CMD"
    
    echo "----------------------------------------"
    echo "Training completed at $(date)"
} 2>&1 | tee "$LOG_PATH/training.log"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    print_header "Training Completed Successfully"
    print_status "Model saved to: $OUTPUT_PATH"
    print_status "Logs saved to: $LOG_PATH"
    
    # Find the best checkpoint
    if [ "$IMPLEMENTATION" = "lightning" ]; then
        BEST_CHECKPOINT=$(find "$OUTPUT_PATH" -name "*.ckpt" | head -1)
        if [ -n "$BEST_CHECKPOINT" ]; then
            print_status "Best checkpoint: $BEST_CHECKPOINT"
            
            # Suggest evaluation command
            print_status ""
            print_status "To evaluate the model, run:"
            print_status "  python3 scripts/evaluate_model.py \\"
            print_status "    --model_path $BEST_CHECKPOINT \\"
            print_status "    --test_data $TEST_DATA \\"
            print_status "    --implementation lightning \\"
            print_status "    --model_type $MODEL_TYPE"
        fi
    fi
    
else
    print_error "Training failed with exit code $TRAINING_EXIT_CODE"
    print_status "Check logs at: $LOG_PATH/training.log"
    exit $TRAINING_EXIT_CODE
fi

print_header "Training Complete"