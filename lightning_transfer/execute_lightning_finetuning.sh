#!/bin/bash -l

# EpiBERT PyTorch Lightning Fine-tuning Script
# Usage: ./execute_lightning_finetuning.sh <data_dir> <pretrained_checkpoint> <output_dir> [gpus] [project_name]

DATA_DIR=${1:-"./data/rampage_fine_tuning"}
PRETRAINED_CHECKPOINT=${2:-"./checkpoints/pretraining/best.ckpt"}
OUTPUT_DIR=${3:-"./checkpoints/finetuning"}
GPUS=${4:-1}
PROJECT_NAME=${5:-"epibert-lightning-finetuning"}

echo "Starting EpiBERT Lightning Fine-tuning..."
echo "Data Directory: $DATA_DIR"
echo "Pretrained Checkpoint: $PRETRAINED_CHECKPOINT"
echo "Output Directory: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Project Name: $PROJECT_NAME"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

python train_lightning.py \
    --data_dir="$DATA_DIR" \
    --checkpoint_dir="$OUTPUT_DIR" \
    --project_name="$PROJECT_NAME" \
    --run_name="finetuning-$(date +%Y%m%d-%H%M%S)" \
    --resume_from_checkpoint="$PRETRAINED_CHECKPOINT" \
    --input_length=524288 \
    --output_length=4096 \
    --batch_size=1 \
    --gpus=$GPUS \
    --precision="16-mixed" \
    --num_transformer_layers=8 \
    --num_heads=8 \
    --d_model=1024 \
    --dropout_rate=0.20 \
    --learning_rate=5e-4 \
    --warmup_steps=5000 \
    --total_steps=500000 \
    --max_epochs=100 \
    --accumulate_grad_batches=1 \
    --log_every_n_steps=50 \
    --num_workers=4

echo "Fine-tuning completed!"