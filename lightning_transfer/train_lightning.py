"""
Example training script for EpiBERT with PyTorch Lightning

This demonstrates how to train the Lightning version of EpiBERT,
showing the simplified training loop compared to the original TensorFlow version.
"""

import os
import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    DeviceStatsMonitor
)

from epibert_lightning import EpiBERTLightning, create_trainer
from data_module import EpiBERTDataModule


def main():
    parser = argparse.ArgumentParser(description='Train EpiBERT with PyTorch Lightning')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='pretraining',
                       choices=['pretraining', 'finetuning'],
                       help='Model type: pretraining or finetuning (auto-sets parameters)')
    parser.add_argument('--input_length', type=int, default=524288,
                       help='Length of input sequence')
    parser.add_argument('--output_length', type=int, default=4096,
                       help='Length of output profile')
    parser.add_argument('--final_output_length', type=int, default=4092,
                       help='Length of final output after cropping')
    parser.add_argument('--num_heads', type=int, default=None,
                       help='Number of attention heads (auto-set if not specified)')
    parser.add_argument('--num_transformer_layers', type=int, default=None,
                       help='Number of transformer layers (auto-set if not specified)')
    parser.add_argument('--d_model', type=int, default=None,
                       help='Model dimension (auto-set if not specified)')
    parser.add_argument('--dropout_rate', type=float, default=None,
                       help='Dropout rate (auto-set if not specified)')
    parser.add_argument('--pointwise_dropout_rate', type=float, default=None,
                       help='Pointwise dropout rate (auto-set if not specified)')
    parser.add_argument('--motif_dropout_rate', type=float, default=0.25,
                       help='Motif dropout rate')
    parser.add_argument('--motif_units_fc', type=int, default=32,
                       help='Motif FC units')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Number of warmup steps')
    parser.add_argument('--total_steps', type=int, default=100000,
                       help='Total training steps')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    
    # Hardware/training setup
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--precision', type=str, default='16-mixed',
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Logging and checkpointing
    parser.add_argument('--project_name', type=str, default='epibert-lightning',
                       help='Wandb project name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_every_n_steps', type=int, default=50,
                       help='Log metrics every N steps')
    
    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set up data module
    data_module = EpiBERTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_length=args.input_length,
        output_length=args.output_length
    )
    
    # Set up model - only pass non-None parameters to allow auto-setting
    model_kwargs = {
        'model_type': args.model_type,
        'input_length': args.input_length,
        'output_length': args.output_length,
        'final_output_length': args.final_output_length,
        'motif_dropout_rate': args.motif_dropout_rate,
        'motif_units_fc': args.motif_units_fc,
        'learning_rate': args.learning_rate,
        'warmup_steps': args.warmup_steps,
        'total_steps': args.total_steps
    }
    
    # Only add parameters if they were explicitly specified
    if args.num_heads is not None:
        model_kwargs['num_heads'] = args.num_heads
    if args.num_transformer_layers is not None:
        model_kwargs['num_transformer_layers'] = args.num_transformer_layers
    if args.d_model is not None:
        model_kwargs['d_model'] = args.d_model
    if args.dropout_rate is not None:
        model_kwargs['dropout_rate'] = args.dropout_rate
    if args.pointwise_dropout_rate is not None:
        model_kwargs['pointwise_dropout_rate'] = args.pointwise_dropout_rate
    
    model = EpiBERTLightning(**model_kwargs)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='epibert-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    device_stats = DeviceStatsMonitor()
    
    # Set up logger
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.run_name,
        log_model='all'
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[
            checkpoint_callback, 
            early_stopping_callback, 
            lr_monitor,
            device_stats
        ],
        logger=wandb_logger,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
        strategy='ddp' if args.gpus > 1 else 'auto',
        sync_batchnorm=True if args.gpus > 1 else False
    )
    
    # Print model info
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train the model
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=args.resume_from_checkpoint
    )
    
    # Test the model
    trainer.test(model=model, datamodule=data_module)
    
    print("Training completed!")


if __name__ == "__main__":
    main()