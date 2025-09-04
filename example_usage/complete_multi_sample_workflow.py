#!/usr/bin/env python3
"""
Complete Multi-Sample EpiBERT Workflow Example

Demonstrates the complete workflow for training EpiBERT with multiple paired
ATAC-seq and RAMPAGE-seq samples, including data processing, training, and evaluation.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lightning_transfer.paired_data_module import PairedDataModule, create_sample_manifest, validate_manifest
from lightning_transfer.epibert_lightning import EpiBERTLightning
from scripts.batch_data_processor import BatchDataProcessor
from scripts.data_converter import DataConverter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_dataset(output_dir: str, num_samples_per_condition: int = 5):
    """Create a demo dataset for testing the multi-sample workflow"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating demo dataset with {num_samples_per_condition} samples per condition")
    
    samples = []
    conditions = ['control', 'treatment']
    cell_types = ['K562', 'GM12878']
    
    for condition in conditions:
        for cell_type in cell_types:
            for rep in range(1, num_samples_per_condition + 1):
                sample_id = f"{cell_type}_{condition}_rep{rep}"
                
                # Create synthetic data files
                atac_file = output_path / f"{sample_id}_atac.h5"
                rampage_file = output_path / f"{sample_id}_rampage.h5"
                
                # Generate synthetic data
                create_synthetic_data_file(atac_file, data_type='atac')
                create_synthetic_data_file(rampage_file, data_type='rampage')
                
                # Determine split
                if rep <= num_samples_per_condition - 2:
                    split = 'train'
                elif rep == num_samples_per_condition - 1:
                    split = 'val'
                else:
                    split = 'test'
                
                samples.append({
                    'sample_id': sample_id,
                    'condition': condition,
                    'cell_type': cell_type,
                    'atac_file': str(atac_file),
                    'rampage_file': str(rampage_file),
                    'batch': split,
                    'replicate': rep
                })
    
    # Create manifest file
    manifest_file = output_path / "demo_manifest.yaml"
    create_sample_manifest(samples, str(manifest_file))
    
    logger.info(f"Created demo dataset with {len(samples)} samples")
    logger.info(f"Manifest file: {manifest_file}")
    
    return str(manifest_file), samples


def create_synthetic_data_file(file_path: Path, data_type: str = 'atac'):
    """Create synthetic HDF5 data file for testing"""
    
    import h5py
    
    # Parameters for synthetic data
    n_regions = 200
    seq_length = 2048  # Smaller for demo
    profile_length = 512
    n_motifs = 693
    
    with h5py.File(file_path, 'w') as f:
        # Create synthetic sequences (one-hot encoded)
        sequences = np.random.rand(n_regions, 4, seq_length).astype(np.float32)
        sequences = (sequences == sequences.max(axis=1, keepdims=True)).astype(np.float32)
        
        # Create synthetic profiles with realistic characteristics
        if data_type == 'atac':
            # ATAC-seq profiles: sparse with peaks
            profiles = np.random.exponential(1.0, (n_regions, profile_length)).astype(np.float32)
            # Add some peaks
            for i in range(n_regions):
                n_peaks = np.random.poisson(5)
                peak_positions = np.random.randint(0, profile_length, n_peaks)
                profiles[i, peak_positions] += np.random.exponential(10, n_peaks)
        else:
            # RAMPAGE-seq profiles: sparser, more localized
            profiles = np.random.exponential(0.5, (n_regions, profile_length)).astype(np.float32)
            for i in range(n_regions):
                n_tss = np.random.poisson(3)
                tss_positions = np.random.randint(0, profile_length, n_tss)
                profiles[i, tss_positions] += np.random.exponential(5, n_tss)
        
        # Create synthetic motif activities
        motif_activities = np.random.beta(2, 5, (n_regions, n_motifs)).astype(np.float32)
        
        # Create synthetic peak centers
        peaks_centers = np.random.randint(0, profile_length, (n_regions, 10)).astype(np.int32)
        
        # Save datasets
        f.create_dataset('sequences', data=sequences, compression='gzip')
        if data_type == 'atac':
            f.create_dataset('atac_profiles', data=profiles, compression='gzip')
        else:
            f.create_dataset('rampage_profiles', data=profiles, compression='gzip')
        f.create_dataset('motif_activities', data=motif_activities, compression='gzip')
        f.create_dataset('peaks_centers', data=peaks_centers, compression='gzip')
        
        # Add metadata
        f.attrs['n_regions'] = n_regions
        f.attrs['data_type'] = data_type
        f.attrs['created_for'] = 'demo'


def analyze_dataset(manifest_file: str):
    """Analyze the dataset composition and statistics"""
    
    logger.info("Analyzing dataset composition...")
    
    # Validate manifest
    validation_result = validate_manifest(manifest_file)
    
    logger.info(f"Manifest validation: {'✓ PASSED' if validation_result['valid'] else '✗ FAILED'}")
    logger.info(f"Total samples: {validation_result['num_samples']}")
    logger.info(f"Conditions: {validation_result['conditions']}")
    logger.info(f"Cell types: {validation_result['cell_types']}")
    
    if validation_result['missing_files']:
        logger.warning(f"Missing files: {validation_result['missing_files']}")
    
    # Load and analyze manifest
    with open(manifest_file, 'r') as f:
        manifest_data = yaml.safe_load(f)
    
    samples_df = pd.DataFrame(manifest_data['samples'])
    
    # Print sample distribution
    print("\nSample Distribution:")
    print("=" * 50)
    distribution = samples_df.groupby(['condition', 'cell_type', 'batch']).size().unstack(fill_value=0)
    print(distribution)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Samples by condition and cell type
    plt.subplot(2, 2, 1)
    condition_counts = samples_df.groupby(['condition', 'cell_type']).size().unstack()
    condition_counts.plot(kind='bar', stacked=True)
    plt.title('Samples by Condition and Cell Type')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    
    # Plot 2: Samples by split
    plt.subplot(2, 2, 2)
    split_counts = samples_df['batch'].value_counts()
    split_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Dataset Split Distribution')
    
    # Plot 3: Replicates distribution
    plt.subplot(2, 2, 3)
    replicate_counts = samples_df['replicate'].value_counts()
    replicate_counts.plot(kind='bar')
    plt.title('Replicate Distribution')
    plt.xlabel('Replicate Number')
    plt.ylabel('Number of Samples')
    
    # Plot 4: Condition balance across splits
    plt.subplot(2, 2, 4)
    split_condition = samples_df.groupby(['batch', 'condition']).size().unstack()
    split_condition.plot(kind='bar', stacked=True)
    plt.title('Condition Balance Across Splits')
    plt.xlabel('Dataset Split')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return samples_df


def setup_data_module(manifest_file: str, config: dict):
    """Setup the paired data module with configuration"""
    
    logger.info("Setting up paired data module...")
    
    data_module = PairedDataModule(
        manifest_file=manifest_file,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        input_length=config['input_length'],
        output_length=config['output_length'],
        atac_mask_dropout=config['atac_mask_dropout'],
        balance_conditions=config['balance_conditions'],
        use_condition_sampler=config['use_condition_sampler'],
        max_samples_per_condition=config.get('max_samples_per_condition'),
        cache_size=config['cache_size']
    )
    
    return data_module


def train_model(data_module: PairedDataModule, config: dict):
    """Train the EpiBERT model with paired samples"""
    
    logger.info("Setting up model and trainer...")
    
    # Create model
    model = EpiBERTLightning(
        model_type=config['model_type'],
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config['output_dir'],
        filename='{epoch}-{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    if config.get('enable_early_stopping', True):
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 10),
            mode='min',
            min_delta=config.get('min_delta', 0.001)
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        gpus=config['num_gpus'] if torch.cuda.is_available() else 0,
        precision=config.get('precision', 'bf16'),
        callbacks=callbacks,
        log_every_n_steps=config.get('log_every_n_steps', 100),
        val_check_interval=config.get('val_check_interval', 1.0),
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        deterministic=config.get('deterministic', False),
        enable_progress_bar=True
    )
    
    logger.info("Starting training...")
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)
    
    logger.info("Training completed!")
    
    return model, trainer


def evaluate_model(model, data_module: PairedDataModule, output_dir: str):
    """Evaluate the trained model"""
    
    logger.info("Evaluating trained model...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get test dataloader
    test_loader = data_module.test_dataloader()
    
    # Evaluate on test set
    model.eval()
    predictions = []
    targets = []
    conditions = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Forward pass
            output = model(batch)
            
            # Collect predictions and targets
            predictions.append(output['predictions'].cpu().numpy())
            targets.append(batch['target'].cpu().numpy())
            conditions.extend(batch['condition'])
    
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    
    # Calculate metrics
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Overall correlations
    pearson_r, _ = pearsonr(predictions.flatten(), targets.flatten())
    spearman_r, _ = spearmanr(predictions.flatten(), targets.flatten())
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Pearson R: {pearson_r:.4f}")
    logger.info(f"  Spearman R: {spearman_r:.4f}")
    logger.info(f"  MSE: {mse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    
    # Save evaluation results
    eval_results = {
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'mse': float(mse),
        'mae': float(mae),
        'n_samples': len(predictions),
        'conditions': list(set(conditions))
    }
    
    eval_file = output_path / "evaluation_results.yaml"
    with open(eval_file, 'w') as f:
        yaml.dump(eval_results, f)
    
    # Create evaluation plots
    create_evaluation_plots(predictions, targets, conditions, output_path)
    
    return eval_results


def create_evaluation_plots(predictions, targets, conditions, output_dir):
    """Create evaluation plots"""
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Prediction vs Target scatter
    plt.subplot(2, 3, 1)
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.5, s=1)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    
    # Plot 2: Residuals
    plt.subplot(2, 3, 2)
    residuals = predictions.flatten() - targets.flatten()
    plt.scatter(targets.flatten(), residuals, alpha=0.5, s=1)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    # Plot 3: Distribution of predictions vs targets
    plt.subplot(2, 3, 3)
    plt.hist(targets.flatten(), bins=50, alpha=0.7, label='True', density=True)
    plt.hist(predictions.flatten(), bins=50, alpha=0.7, label='Predicted', density=True)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Value Distributions')
    plt.legend()
    
    # Plot 4: Per-sample correlations
    plt.subplot(2, 3, 4)
    sample_correlations = []
    for i in range(len(predictions)):
        if np.var(targets[i]) > 0 and np.var(predictions[i]) > 0:
            corr, _ = pearsonr(targets[i], predictions[i])
            sample_correlations.append(corr)
    
    plt.hist(sample_correlations, bins=20, alpha=0.7)
    plt.xlabel('Per-Sample Correlation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Per-Sample Correlations')
    
    # Plot 5: Mean profiles by condition
    plt.subplot(2, 3, 5)
    unique_conditions = list(set(conditions))
    for condition in unique_conditions:
        condition_mask = np.array(conditions) == condition
        if np.any(condition_mask):
            mean_true = np.mean(targets[condition_mask], axis=0)
            mean_pred = np.mean(predictions[condition_mask], axis=0)
            x = np.arange(len(mean_true))
            plt.plot(x, mean_true, label=f'{condition} (True)', linestyle='-')
            plt.plot(x, mean_pred, label=f'{condition} (Pred)', linestyle='--')
    
    plt.xlabel('Position')
    plt.ylabel('Mean Signal')
    plt.title('Mean Profiles by Condition')
    plt.legend()
    
    # Plot 6: Error by position
    plt.subplot(2, 3, 6)
    position_errors = np.mean(np.abs(predictions - targets), axis=0)
    plt.plot(position_errors)
    plt.xlabel('Position')
    plt.ylabel('Mean Absolute Error')
    plt.title('Error by Genomic Position')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main workflow function"""
    
    print("🚀 EpiBERT Multi-Sample Workflow Example")
    print("=" * 60)
    
    # Configuration
    config = {
        'batch_size': 4,
        'num_workers': 2,
        'input_length': 2048,    # Smaller for demo
        'output_length': 512,
        'atac_mask_dropout': 0.15,
        'balance_conditions': True,
        'use_condition_sampler': True,
        'max_samples_per_condition': 10,
        'cache_size': 20,
        'model_type': 'pretraining',
        'learning_rate': 1e-4,
        'max_epochs': 5,         # Short for demo
        'num_gpus': 1 if torch.cuda.is_available() else 0,
        'output_dir': 'demo_output',
        'enable_early_stopping': False  # Disabled for short demo
    }
    
    try:
        # Step 1: Create demo dataset
        print("\n📊 Step 1: Creating demo dataset...")
        manifest_file, samples = create_demo_dataset("demo_data", num_samples_per_condition=3)
        
        # Step 2: Analyze dataset
        print("\n📈 Step 2: Analyzing dataset...")
        samples_df = analyze_dataset(manifest_file)
        
        # Step 3: Setup data module
        print("\n🔧 Step 3: Setting up data module...")
        data_module = setup_data_module(manifest_file, config)
        data_module.setup("fit")
        
        # Test data loading
        print("\n🧪 Step 4: Testing data loading...")
        train_loader = data_module.train_dataloader()
        sample_batch = next(iter(train_loader))
        
        print(f"Sample batch shapes:")
        for key, value in sample_batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: list of {len(value)} items")
        
        # Step 5: Train model
        print("\n🏋️ Step 5: Training model...")
        model, trainer = train_model(data_module, config)
        
        # Step 6: Evaluate model
        print("\n📊 Step 6: Evaluating model...")
        eval_results = evaluate_model(model, data_module, config['output_dir'])
        
        print("\n✅ Workflow completed successfully!")
        print(f"Results saved to: {config['output_dir']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Multi-sample EpiBERT workflow completed successfully!")
    else:
        print("\n❌ Workflow failed. Check logs for details.")
        sys.exit(1)