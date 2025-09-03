#!/usr/bin/env python3
"""
Enhanced test script for EpiBERT Lightning implementation.

This script provides comprehensive testing of the Lightning version of EpiBERT 
with various configuration options for minimal computation testing.
"""

import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional
import numpy as np
import argparse
from pathlib import Path
import json
import time

# Add the lightning_transfer directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lightning_transfer'))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import the original test components
from test_lightning_minimal import SimplifiedDataModule, SimplifiedEpiBERT


class EnhancedEpiBERTTest:
    """Enhanced test runner for EpiBERT Lightning with comprehensive options."""
    
    def __init__(self, args):
        self.args = args
        self.results = {}
        
    def log_info(self, message: str, level: str = "INFO"):
        """Log information with timestamps."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def test_gpu_availability(self):
        """Test GPU availability and setup."""
        self.log_info("Testing GPU availability...")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            self.log_info(f"GPU available: {device_count} device(s)")
            self.log_info(f"Primary GPU: {device_name}")
            return True
        else:
            self.log_info("No GPU available, using CPU")
            return False
    
    def create_data_module(self, stage: str = 'pretraining'):
        """Create data module with different configurations for different stages."""
        if stage == 'pretraining':
            return SimplifiedDataModule(
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                input_length=self.args.input_length,
                output_length=self.args.output_length,
                num_samples=self.args.num_samples
            )
        elif stage == 'finetuning':
            # Import from the test script
            from test_lightning_minimal import run_finetuning_test
            # This is a bit hacky but works for our testing purposes
            class FinetuningDataModule(SimplifiedDataModule):
                def _create_synthetic_data(self, split: str):
                    data = super()._create_synthetic_data(split)
                    for item in data:
                        rna_length = self.output_length // 4
                        rna_target = torch.abs(torch.randn(rna_length)) * 20
                        item['rna_target'] = rna_target
                    return data
            
            return FinetuningDataModule(
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                input_length=self.args.input_length,
                output_length=self.args.output_length,
                num_samples=self.args.num_samples
            )
    
    def create_model(self, stage: str = 'pretraining'):
        """Create model with different configurations for different stages."""
        if stage == 'pretraining':
            return SimplifiedEpiBERT(
                input_length=self.args.input_length,
                output_length=self.args.output_length,
                d_model=self.args.d_model,
                num_heads=self.args.num_heads,
                num_layers=self.args.num_layers,
                learning_rate=self.args.learning_rate
            )
        elif stage == 'finetuning':
            # Create the finetuning model from the test script
            from test_lightning_minimal import run_finetuning_test
            class FinetuningEpiBERT(SimplifiedEpiBERT):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.rna_head = nn.Sequential(
                        nn.Linear(self.d_model, self.d_model // 2),
                        nn.ReLU(),
                        nn.Linear(self.d_model // 2, self.output_length // 4)
                    )
                
                def forward(self, sequence, atac, motif_activity):
                    batch_size = sequence.shape[0]
                    
                    # Process sequence
                    seq_features = self.seq_conv(sequence)
                    if seq_features.shape[2] != self.output_length:
                        seq_pool_size = seq_features.shape[2] // self.output_length
                        if seq_pool_size > 1:
                            seq_features = nn.MaxPool1d(kernel_size=seq_pool_size)(seq_features)
                        else:
                            seq_features = nn.functional.interpolate(seq_features, size=self.output_length, mode='linear', align_corners=False)
                    seq_features = seq_features.transpose(1, 2)
                    
                    # Process ATAC
                    atac_input = atac.squeeze(1) if atac.dim() == 3 else atac
                    if atac_input.dim() == 1:
                        atac_input = atac_input.unsqueeze(0)
                    if atac_input.shape[1] != self.output_length:
                        atac_input = nn.functional.interpolate(atac_input.unsqueeze(1), size=self.output_length, mode='linear', align_corners=False).squeeze(1)
                    
                    atac_features = self.atac_conv(atac_input.unsqueeze(1))
                    atac_features = atac_features.transpose(1, 2)
                    
                    # Process motif activity
                    motif_features = self.motif_fc(motif_activity)
                    motif_features = motif_features.unsqueeze(1).expand(-1, self.output_length, -1)
                    
                    # Combine features
                    combined = torch.cat([seq_features, atac_features, motif_features], dim=-1)
                    if not hasattr(self, 'feature_projection'):
                        self.feature_projection = nn.Linear(combined.shape[-1], self.d_model).to(combined.device)
                    combined = self.feature_projection(combined)
                    
                    # Apply transformer
                    transformed = self.transformer(combined)
                    
                    # Predict ATAC
                    atac_pred = self.output_head(transformed).squeeze(-1)
                    
                    # Predict RNA
                    rna_features = transformed.mean(dim=1)
                    rna_pred = self.rna_head(rna_features)
                    
                    return atac_pred, rna_pred
                
                def training_step(self, batch, batch_idx):
                    sequence = batch['sequence']
                    atac = batch['atac']
                    motif_activity = batch['motif_activity']
                    atac_target = batch['target']
                    rna_target = batch['rna_target']
                    
                    atac_pred, rna_pred = self(sequence, atac, motif_activity)
                    
                    atac_loss = self.loss_fn(atac_pred + 1e-8, atac_target + 1e-8)
                    rna_loss = self.loss_fn(rna_pred + 1e-8, rna_target + 1e-8)
                    total_loss = 0.1 * atac_loss + 1.0 * rna_loss
                    
                    self.log('train_loss', total_loss, prog_bar=True)
                    self.log('train_atac_loss', atac_loss)
                    self.log('train_rna_loss', rna_loss)
                    
                    return total_loss
                
                def validation_step(self, batch, batch_idx):
                    sequence = batch['sequence']
                    atac = batch['atac']
                    motif_activity = batch['motif_activity']
                    atac_target = batch['target']
                    rna_target = batch['rna_target']
                    
                    atac_pred, rna_pred = self(sequence, atac, motif_activity)
                    
                    atac_loss = self.loss_fn(atac_pred + 1e-8, atac_target + 1e-8)
                    rna_loss = self.loss_fn(rna_pred + 1e-8, rna_target + 1e-8)
                    total_loss = 0.1 * atac_loss + 1.0 * rna_loss
                    
                    self.log('val_loss', total_loss, prog_bar=True, sync_dist=True)
                    self.log('val_atac_loss', atac_loss, sync_dist=True)
                    self.log('val_rna_loss', rna_loss, sync_dist=True)
                    
                    return total_loss
            
            return FinetuningEpiBERT(
                input_length=self.args.input_length,
                output_length=self.args.output_length,
                d_model=self.args.d_model,
                num_heads=self.args.num_heads,
                num_layers=self.args.num_layers,
                learning_rate=self.args.learning_rate
            )
    
    def create_trainer(self, stage: str = 'pretraining'):
        """Create trainer with appropriate configuration."""
        accelerator = 'gpu' if self.args.use_gpu and torch.cuda.is_available() else 'cpu'
        devices = 1 if accelerator == 'gpu' else 1
        
        callbacks = []
        if self.args.enable_checkpointing:
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=self.args.checkpoint_dir,
                filename=f'{stage}-{{epoch:02d}}-{{step}}',
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                save_last=False
            )
            callbacks.append(checkpoint_callback)
        
        if self.args.early_stopping:
            early_stop_callback = pl.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=self.args.patience,
                verbose=True
            )
            callbacks.append(early_stop_callback)
        
        trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=self.args.precision,
            log_every_n_steps=self.args.log_every_n_steps,
            enable_progress_bar=not self.args.quiet,
            enable_model_summary=True,
            fast_dev_run=self.args.fast_dev_run,
            limit_train_batches=self.args.limit_batches,
            limit_val_batches=self.args.limit_batches,
            callbacks=callbacks,
            logger=False,  # Disable default logger for simplicity
            enable_checkpointing=self.args.enable_checkpointing
        )
        
        return trainer
    
    def test_data_loading(self, data_module, stage: str):
        """Test data loading functionality."""
        self.log_info(f"Testing {stage} data loading...")
        
        data_module.setup("fit")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        # Test train loader
        train_batch = next(iter(train_loader))
        self.log_info(f"Train batch shapes:")
        for key, value in train_batch.items():
            self.log_info(f"  {key}: {value.shape}")
        
        # Test val loader
        val_batch = next(iter(val_loader))
        self.log_info(f"Validation batch shapes:")
        for key, value in val_batch.items():
            self.log_info(f"  {key}: {value.shape}")
        
        return train_batch, val_batch
    
    def test_model_forward(self, model, batch, stage: str):
        """Test model forward pass."""
        self.log_info(f"Testing {stage} model forward pass...")
        
        with torch.no_grad():
            if stage == 'pretraining':
                output = model(batch['sequence'], batch['atac'], batch['motif_activity'])
                self.log_info(f"Output shape: {output.shape}")
                self.log_info(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
                return {'output': output}
            elif stage == 'finetuning':
                atac_output, rna_output = model(batch['sequence'], batch['atac'], batch['motif_activity'])
                self.log_info(f"ATAC output shape: {atac_output.shape}")
                self.log_info(f"RNA output shape: {rna_output.shape}")
                self.log_info(f"ATAC range: [{atac_output.min():.3f}, {atac_output.max():.3f}]")
                self.log_info(f"RNA range: [{rna_output.min():.3f}, {rna_output.max():.3f}]")
                return {'atac_output': atac_output, 'rna_output': rna_output}
    
    def run_stage_test(self, stage: str):
        """Run comprehensive test for a specific stage."""
        self.log_info(f"=" * 60)
        self.log_info(f"TESTING {stage.upper()} STAGE")
        self.log_info(f"=" * 60)
        
        try:
            # Create data module
            data_module = self.create_data_module(stage)
            
            # Create model
            model = self.create_model(stage)
            param_count = sum(p.numel() for p in model.parameters())
            self.log_info(f"Model has {param_count:,} parameters")
            
            # Test data loading
            train_batch, val_batch = self.test_data_loading(data_module, stage)
            
            # Test forward pass
            outputs = self.test_model_forward(model, train_batch, stage)
            
            # Create trainer
            trainer = self.create_trainer(stage)
            
            # Run training
            self.log_info(f"Running {stage} training for {self.args.max_epochs} epochs...")
            start_time = time.time()
            trainer.fit(model, data_module)
            training_time = time.time() - start_time
            
            # Store results
            self.results[stage] = {
                'success': True,
                'parameters': param_count,
                'training_time': training_time,
                'batch_shapes': {k: v.shape for k, v in train_batch.items()},
                'output_shapes': {k: v.shape for k, v in outputs.items()}
            }
            
            self.log_info(f"✓ {stage} test completed successfully in {training_time:.2f}s!")
            return True
            
        except Exception as e:
            self.log_info(f"✗ {stage} test failed: {e}", level="ERROR")
            self.results[stage] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def run_benchmark(self):
        """Run performance benchmark."""
        if not self.args.benchmark:
            return
        
        self.log_info("=" * 60)
        self.log_info("PERFORMANCE BENCHMARK")
        self.log_info("=" * 60)
        
        # Test different model sizes
        model_configs = [
            {'d_model': 32, 'num_heads': 2, 'num_layers': 1, 'name': 'tiny'},
            {'d_model': 64, 'num_heads': 4, 'num_layers': 2, 'name': 'small'},
            {'d_model': 128, 'num_heads': 8, 'num_layers': 4, 'name': 'medium'},
        ]
        
        benchmark_results = {}
        
        for config in model_configs:
            self.log_info(f"Benchmarking {config['name']} model...")
            
            # Temporarily override args
            original_d_model = self.args.d_model
            original_num_heads = self.args.num_heads
            original_num_layers = self.args.num_layers
            
            self.args.d_model = config['d_model']
            self.args.num_heads = config['num_heads']
            self.args.num_layers = config['num_layers']
            
            try:
                model = self.create_model('pretraining')
                data_module = self.create_data_module('pretraining')
                
                # Count parameters
                param_count = sum(p.numel() for p in model.parameters())
                
                # Measure forward pass time
                data_module.setup("fit")
                batch = next(iter(data_module.train_dataloader()))
                
                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        model(batch['sequence'], batch['atac'], batch['motif_activity'])
                
                # Timing
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        output = model(batch['sequence'], batch['atac'], batch['motif_activity'])
                forward_time = (time.time() - start_time) / 10
                
                benchmark_results[config['name']] = {
                    'parameters': param_count,
                    'forward_time_ms': forward_time * 1000,
                    'config': config
                }
                
                self.log_info(f"  Parameters: {param_count:,}")
                self.log_info(f"  Forward time: {forward_time*1000:.2f}ms")
                
            except Exception as e:
                self.log_info(f"  Benchmark failed: {e}", level="ERROR")
                benchmark_results[config['name']] = {'error': str(e)}
            
            # Restore original args
            self.args.d_model = original_d_model
            self.args.num_heads = original_num_heads
            self.args.num_layers = original_num_layers
        
        self.results['benchmark'] = benchmark_results
    
    def save_results(self):
        """Save test results to file."""
        if self.args.output_file:
            self.log_info(f"Saving results to {self.args.output_file}")
            with open(self.args.output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
    
    def print_summary(self):
        """Print test summary."""
        self.log_info("=" * 60)
        self.log_info("TEST RESULTS SUMMARY")
        self.log_info("=" * 60)
        
        overall_success = True
        
        for stage in ['pretraining', 'finetuning']:
            if stage in self.results:
                result = self.results[stage]
                if result['success']:
                    params = result.get('parameters', 'Unknown')
                    time_taken = result.get('training_time', 0)
                    self.log_info(f"{stage.capitalize()}: ✓ PASSED ({params:,} params, {time_taken:.2f}s)")
                else:
                    self.log_info(f"{stage.capitalize()}: ✗ FAILED ({result.get('error', 'Unknown error')})")
                    overall_success = False
        
        if 'benchmark' in self.results:
            self.log_info("\nBenchmark Results:")
            for model_name, result in self.results['benchmark'].items():
                if 'error' not in result:
                    params = result['parameters']
                    forward_time = result['forward_time_ms']
                    self.log_info(f"  {model_name}: {params:,} params, {forward_time:.2f}ms/forward")
        
        status = "✓ ALL TESTS PASSED" if overall_success else "✗ SOME TESTS FAILED"
        self.log_info(f"\nOverall: {status}")
        
        return overall_success


def main():
    parser = argparse.ArgumentParser(description='Enhanced Lightning EpiBERT testing')
    
    # Model parameters
    parser.add_argument('--input_length', type=int, default=8192, help='Input sequence length')
    parser.add_argument('--output_length', type=int, default=256, help='Output profile length')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=1, help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per split')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data workers')
    
    # Hardware/performance
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--precision', type=str, default='32', choices=['16', '32', 'bf16'], help='Training precision')
    
    # Testing options
    parser.add_argument('--fast_dev_run', action='store_true', help='Run single batch only')
    parser.add_argument('--limit_batches', type=float, default=1.0, help='Limit batches per epoch')
    parser.add_argument('--test_pretraining', action='store_true', help='Test pretraining stage')
    parser.add_argument('--test_finetuning', action='store_true', help='Test finetuning stage')
    parser.add_argument('--test_both', action='store_true', help='Test both stages')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    # Trainer options
    parser.add_argument('--enable_checkpointing', action='store_true', help='Enable model checkpointing')
    parser.add_argument('--checkpoint_dir', type=str, default='./test_checkpoints', help='Checkpoint directory')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--log_every_n_steps', type=int, default=1, help='Log frequency')
    
    # Output options
    parser.add_argument('--quiet', action='store_true', help='Suppress progress bars')
    parser.add_argument('--output_file', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Default to testing both if no specific test specified
    if not any([args.test_pretraining, args.test_finetuning, args.test_both]):
        args.test_both = True
    
    # Create test runner
    test_runner = EnhancedEpiBERTTest(args)
    
    # Print initial info
    test_runner.log_info("Enhanced EpiBERT Lightning Testing")
    test_runner.log_info(f"PyTorch version: {torch.__version__}")
    test_runner.log_info(f"PyTorch Lightning version: {pl.__version__}")
    
    # Test GPU
    gpu_available = test_runner.test_gpu_availability()
    if args.use_gpu and not gpu_available:
        test_runner.log_info("GPU requested but not available, falling back to CPU")
        args.use_gpu = False
    
    # Run tests
    overall_success = True
    
    if args.test_pretraining or args.test_both:
        success = test_runner.run_stage_test('pretraining')
        overall_success = overall_success and success
    
    if args.test_finetuning or args.test_both:
        success = test_runner.run_stage_test('finetuning')
        overall_success = overall_success and success
    
    # Run benchmark if requested
    test_runner.run_benchmark()
    
    # Save results
    test_runner.save_results()
    
    # Print summary
    test_runner.print_summary()
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    exit(main())