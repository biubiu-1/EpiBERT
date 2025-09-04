#!/bin/bash -l

# EpiBERT PyTorch Lightning Pretraining Script
# Usage: ./execute_lightning_pretraining.sh <data_dir> <output_dir> [gpus] [project_name]

DATA_DIR=${1:-"./data/atac_pretraining"}
OUTPUT_DIR=${2:-"./checkpoints/pretraining"}
GPUS=${3:-1}
PROJECT_NAME=${4:-"epibert-lightning-pretraining"}

echo "Starting EpiBERT Lightning Pretraining..."
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Project Name: $PROJECT_NAME"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

python train_lightning.py \
    --data_dir="$DATA_DIR" \
    --checkpoint_dir="$OUTPUT_DIR" \
    --project_name="$PROJECT_NAME" \
    --run_name="pretraining-$(date +%Y%m%d-%H%M%S)" \
    --model_type="pretraining" \
    --input_length=524288 \
    --output_length=4096 \
    --batch_size=1 \
    --gpus=$GPUS \
    --precision="16-mixed" \
    --learning_rate=1e-4 \
    --warmup_steps=5000 \
    --total_steps=500000 \
    --max_epochs=100 \
    --accumulate_grad_batches=1 \
    --log_every_n_steps=50 \
    --num_workers=4

echo "Pretraining completed!"