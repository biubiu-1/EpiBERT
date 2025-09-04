#!/usr/bin/env python3
"""
EpiBERT Model Evaluation and Performance Assessment

This script provides comprehensive evaluation metrics for EpiBERT models including:
- Pearson and Spearman correlations
- MSE and MAE metrics
- ROC-AUC and PR-AUC for peak prediction
- Profile-level correlations
- Motif attribution analysis
- Cross-cell-type generalization assessment
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import h5py
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set up paths for imports
script_dir = Path(__file__).parent.absolute()
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

class EpiBERTEvaluator:
    """Comprehensive evaluation class for EpiBERT models"""
    
    def __init__(self, implementation: str = "tensorflow"):
        """
        Initialize evaluator
        
        Args:
            implementation: Either "tensorflow" or "lightning"
        """
        self.implementation = implementation
        self.results = {}
        
        # Import appropriate modules based on implementation
        if implementation == "lightning":
            try:
                import torch
                import pytorch_lightning as pl
                from lightning_transfer.epibert_lightning import EpiBERTLightning
                from lightning_transfer.data_module import EpiBERTDataModule
                self.torch = torch
                self.pl = pl
                self.EpiBERTLightning = EpiBERTLightning
                self.EpiBERTDataModule = EpiBERTDataModule
            except ImportError as e:
                raise ImportError(f"Lightning dependencies not available: {e}")
        else:
            try:
                import tensorflow as tf
                import src.models.epibert_atac_pretrain as epibert_pretrain
                import src.models.epibert_rampage_finetune as epibert_finetune
                import training_utils_atac_pretrain
                import training_utils_rampage_finetune
                self.tf = tf
                self.epibert_pretrain = epibert_pretrain
                self.epibert_finetune = epibert_finetune
            except ImportError as e:
                raise ImportError(f"TensorFlow dependencies not available: {e}")
    
    def load_model(self, model_path: str, model_type: str = "pretraining") -> Union[object, None]:
        """Load trained model from checkpoint"""
        print(f"Loading {self.implementation} model from {model_path}")
        
        if self.implementation == "lightning":
            model = self.EpiBERTLightning.load_from_checkpoint(model_path, model_type=model_type)
            model.eval()
            return model
        else:
            # TensorFlow model loading
            if model_type == "pretraining":
                model = self.epibert_pretrain.epibert()
            else:
                model = self.epibert_finetune.epibert()
            
            # Load weights
            model.load_weights(model_path)
            return model
    
    def load_test_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data from various formats"""
        print(f"Loading test data from {data_path}")
        
        if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
            # HDF5 format
            with h5py.File(data_path, 'r') as f:
                inputs = f['inputs'][:]
                targets = f['targets'][:]
            return inputs, targets
            
        elif data_path.endswith('.npz'):
            # NumPy format
            data = np.load(data_path)
            return data['inputs'], data['targets']
            
        elif data_path.endswith('.tfr') or 'tfrecord' in data_path:
            # TensorFlow record format
            return self._load_tfrecord_data(data_path)
        
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
    
    def _load_tfrecord_data(self, tfrecord_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from TensorFlow records"""
        if self.implementation != "tensorflow":
            raise ValueError("TFRecord loading only supported with TensorFlow implementation")
        
        # Use the training utilities to parse TFRecords
        dataset = self.tf.data.TFRecordDataset(tfrecord_path)
        
        inputs_list = []
        targets_list = []
        
        for record in dataset.take(1000):  # Limit for evaluation
            # Parse using training utils
            parsed = training_utils_atac_pretrain.parse_proto_v2(record)
            inputs_list.append(parsed['sequence'].numpy())
            targets_list.append(parsed['target'].numpy())
        
        return np.array(inputs_list), np.array(targets_list)
    
    def predict(self, model, inputs: np.ndarray) -> np.ndarray:
        """Generate predictions from model"""
        print(f"Generating predictions for {len(inputs)} samples")
        
        if self.implementation == "lightning":
            model.eval()
            with self.torch.no_grad():
                if isinstance(inputs, np.ndarray):
                    inputs = self.torch.tensor(inputs, dtype=self.torch.float32)
                
                # Handle batch processing
                batch_size = 4  # Adjust based on memory
                predictions = []
                
                for i in range(0, len(inputs), batch_size):
                    batch = inputs[i:i+batch_size]
                    if len(batch.shape) == 3:  # Add batch dimension if needed
                        batch = batch.unsqueeze(0)
                    
                    pred = model(batch)
                    predictions.append(pred.cpu().numpy())
                
                return np.concatenate(predictions, axis=0)
        else:
            # TensorFlow prediction
            return model.predict(inputs, batch_size=4, verbose=1)
    
    def compute_correlation_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Compute correlation-based metrics"""
        print("Computing correlation metrics...")
        
        # Flatten arrays for global correlation
        targets_flat = targets.flatten()
        predictions_flat = predictions.flatten()
        
        # Remove any NaN or infinite values
        mask = np.isfinite(targets_flat) & np.isfinite(predictions_flat)
        targets_clean = targets_flat[mask]
        predictions_clean = predictions_flat[mask]
        
        # Global correlations
        pearson_r, pearson_p = pearsonr(targets_clean, predictions_clean)
        spearman_r, spearman_p = spearmanr(targets_clean, predictions_clean)
        
        # Per-sample correlations
        sample_pearson = []
        sample_spearman = []
        
        for i in range(len(targets)):
            t_sample = targets[i].flatten()
            p_sample = predictions[i].flatten()
            
            # Remove NaN/inf for this sample
            mask_sample = np.isfinite(t_sample) & np.isfinite(p_sample)
            if np.sum(mask_sample) > 10:  # Need sufficient points
                t_clean = t_sample[mask_sample]
                p_clean = p_sample[mask_sample]
                
                if np.std(t_clean) > 0 and np.std(p_clean) > 0:
                    r_p, _ = pearsonr(t_clean, p_clean)
                    r_s, _ = spearmanr(t_clean, p_clean)
                    sample_pearson.append(r_p)
                    sample_spearman.append(r_s)
        
        metrics = {
            'global_pearson_r': pearson_r,
            'global_pearson_p': pearson_p,
            'global_spearman_r': spearman_r,
            'global_spearman_p': spearman_p,
            'mean_sample_pearson': np.mean(sample_pearson),
            'median_sample_pearson': np.median(sample_pearson),
            'std_sample_pearson': np.std(sample_pearson),
            'mean_sample_spearman': np.mean(sample_spearman),
            'median_sample_spearman': np.median(sample_spearman),
            'std_sample_spearman': np.std(sample_spearman),
            'n_samples_evaluated': len(sample_pearson)
        }
        
        return metrics
    
    def compute_regression_metrics(self, targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Compute regression-based metrics"""
        print("Computing regression metrics...")
        
        # Flatten for global metrics
        targets_flat = targets.flatten()
        predictions_flat = predictions.flatten()
        
        # Remove any NaN or infinite values
        mask = np.isfinite(targets_flat) & np.isfinite(predictions_flat)
        targets_clean = targets_flat[mask]
        predictions_clean = predictions_flat[mask]
        
        mse = mean_squared_error(targets_clean, predictions_clean)
        mae = mean_absolute_error(targets_clean, predictions_clean)
        rmse = np.sqrt(mse)
        
        # Normalized metrics
        target_var = np.var(targets_clean)
        explained_var = 1 - (np.var(targets_clean - predictions_clean) / target_var)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'explained_variance': explained_var,
            'target_mean': np.mean(targets_clean),
            'target_std': np.std(targets_clean),
            'pred_mean': np.mean(predictions_clean),
            'pred_std': np.std(predictions_clean)
        }
        
        return metrics
    
    def compute_peak_prediction_metrics(self, targets: np.ndarray, predictions: np.ndarray, 
                                      threshold: float = 0.5) -> Dict[str, float]:
        """Compute binary classification metrics for peak prediction"""
        print("Computing peak prediction metrics...")
        
        # Convert to binary (peak vs no peak)
        targets_binary = (targets > threshold).astype(int)
        predictions_binary = (predictions > threshold).astype(int)
        
        # Flatten
        targets_flat = targets_binary.flatten()
        predictions_flat = predictions_binary.flatten()
        
        # Use continuous predictions for ROC/PR curves
        targets_continuous = targets.flatten()
        predictions_continuous = predictions.flatten()
        
        # Remove NaN/inf
        mask = np.isfinite(targets_continuous) & np.isfinite(predictions_continuous)
        targets_cont_clean = targets_continuous[mask]
        predictions_cont_clean = predictions_continuous[mask]
        targets_bin_clean = targets_flat[mask]
        
        # Convert to binary for classification metrics
        targets_bin_clean = (targets_cont_clean > threshold).astype(int)
        
        try:
            # ROC-AUC
            roc_auc = roc_auc_score(targets_bin_clean, predictions_cont_clean)
        except:
            roc_auc = np.nan
        
        try:
            # PR-AUC
            pr_auc = average_precision_score(targets_bin_clean, predictions_cont_clean)
        except:
            pr_auc = np.nan
        
        # Basic classification metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        pred_bin_clean = (predictions_cont_clean > threshold).astype(int)
        
        try:
            precision = precision_score(targets_bin_clean, pred_bin_clean, zero_division=0)
            recall = recall_score(targets_bin_clean, pred_bin_clean, zero_division=0)
            f1 = f1_score(targets_bin_clean, pred_bin_clean, zero_division=0)
        except:
            precision = recall = f1 = np.nan
        
        metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'positive_rate_targets': np.mean(targets_bin_clean),
            'positive_rate_predictions': np.mean(pred_bin_clean),
            'threshold': threshold
        }
        
        return metrics
    
    def evaluate_model(self, model, test_data_path: str, output_dir: str, 
                      model_name: str = "epibert_model") -> Dict[str, any]:
        """Run comprehensive evaluation"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        inputs, targets = self.load_test_data(test_data_path)
        print(f"Loaded {len(inputs)} test samples")
        print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
        
        # Generate predictions
        predictions = self.predict(model, inputs)
        print(f"Generated predictions shape: {predictions.shape}")
        
        # Compute all metrics
        print("\n" + "="*50)
        print("COMPUTING EVALUATION METRICS")
        print("="*50)
        
        correlation_metrics = self.compute_correlation_metrics(targets, predictions)
        regression_metrics = self.compute_regression_metrics(targets, predictions)
        peak_metrics = self.compute_peak_prediction_metrics(targets, predictions)
        
        # Combine all metrics
        all_metrics = {
            'model_name': model_name,
            'implementation': self.implementation,
            'n_samples': len(inputs),
            'input_shape': list(inputs.shape),
            'target_shape': list(targets.shape),
            'prediction_shape': list(predictions.shape),
            **correlation_metrics,
            **regression_metrics,
            **peak_metrics
        }
        
        # Save metrics
        metrics_file = os.path.join(output_dir, f"{model_name}_evaluation_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        print(f"\nMetrics saved to: {metrics_file}")
        
        # Generate plots
        self.generate_evaluation_plots(targets, predictions, output_dir, model_name)
        
        # Print summary
        self.print_evaluation_summary(all_metrics)
        
        return all_metrics
    
    def generate_evaluation_plots(self, targets: np.ndarray, predictions: np.ndarray, 
                                output_dir: str, model_name: str):
        """Generate evaluation plots"""
        print("Generating evaluation plots...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Scatter plot of targets vs predictions
        plt.figure(figsize=(10, 8))
        
        # Sample data for plotting if too large
        if len(targets.flatten()) > 10000:
            idx = np.random.choice(len(targets.flatten()), 10000, replace=False)
            targets_sample = targets.flatten()[idx]
            predictions_sample = predictions.flatten()[idx]
        else:
            targets_sample = targets.flatten()
            predictions_sample = predictions.flatten()
        
        plt.subplot(2, 2, 1)
        plt.scatter(targets_sample, predictions_sample, alpha=0.6, s=1)
        plt.xlabel('Targets')
        plt.ylabel('Predictions')
        plt.title('Targets vs Predictions')
        
        # Add diagonal line
        lims = [
            np.min([plt.xlim(), plt.ylim()]),
            np.max([plt.xlim(), plt.ylim()])
        ]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        
        # 2. Residuals plot
        plt.subplot(2, 2, 2)
        residuals = predictions_sample - targets_sample
        plt.scatter(targets_sample, residuals, alpha=0.6, s=1)
        plt.xlabel('Targets')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Targets')
        plt.axhline(y=0, color='r', linestyle='--')
        
        # 3. Distribution comparison
        plt.subplot(2, 2, 3)
        plt.hist(targets_sample, bins=50, alpha=0.7, label='Targets', density=True)
        plt.hist(predictions_sample, bins=50, alpha=0.7, label='Predictions', density=True)
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.title('Value Distributions')
        plt.legend()
        
        # 4. Q-Q plot
        plt.subplot(2, 2, 4)
        from scipy.stats import probplot
        probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_evaluation_plots.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Profile-level correlation plot
        if len(targets.shape) > 1 and targets.shape[1] > 1:
            self._plot_profile_correlations(targets, predictions, output_dir, model_name)
    
    def _plot_profile_correlations(self, targets: np.ndarray, predictions: np.ndarray, 
                                 output_dir: str, model_name: str):
        """Plot per-sample profile correlations"""
        
        correlations = []
        for i in range(len(targets)):
            t_flat = targets[i].flatten()
            p_flat = predictions[i].flatten()
            
            mask = np.isfinite(t_flat) & np.isfinite(p_flat)
            if np.sum(mask) > 10:
                t_clean = t_flat[mask]
                p_clean = p_flat[mask]
                
                if np.std(t_clean) > 0 and np.std(p_clean) > 0:
                    r, _ = pearsonr(t_clean, p_clean)
                    correlations.append(r)
        
        if correlations:
            plt.figure(figsize=(10, 6))
            plt.hist(correlations, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Per-Sample Pearson Correlation')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Per-Sample Correlations\nMean: {np.mean(correlations):.3f}, Median: {np.median(correlations):.3f}')
            plt.axvline(np.mean(correlations), color='red', linestyle='--', label=f'Mean: {np.mean(correlations):.3f}')
            plt.axvline(np.median(correlations), color='orange', linestyle='--', label=f'Median: {np.median(correlations):.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(output_dir, f"{model_name}_profile_correlations.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def print_evaluation_summary(self, metrics: Dict[str, any]):
        """Print formatted evaluation summary"""
        print("\n" + "="*60)
        print(f"EVALUATION SUMMARY: {metrics['model_name']}")
        print("="*60)
        
        print(f"Implementation: {metrics['implementation']}")
        print(f"Samples evaluated: {metrics['n_samples']}")
        print(f"Input shape: {metrics['input_shape']}")
        print(f"Target shape: {metrics['target_shape']}")
        
        print("\n--- CORRELATION METRICS ---")
        print(f"Global Pearson r: {metrics['global_pearson_r']:.4f} (p={metrics['global_pearson_p']:.2e})")
        print(f"Global Spearman r: {metrics['global_spearman_r']:.4f} (p={metrics['global_spearman_p']:.2e})")
        print(f"Mean per-sample Pearson: {metrics['mean_sample_pearson']:.4f} ± {metrics['std_sample_pearson']:.4f}")
        print(f"Mean per-sample Spearman: {metrics['mean_sample_spearman']:.4f} ± {metrics['std_sample_spearman']:.4f}")
        
        print("\n--- REGRESSION METRICS ---")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"Explained Variance: {metrics['explained_variance']:.4f}")
        
        print("\n--- PEAK PREDICTION METRICS ---")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate EpiBERT model performance')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data (HDF5, NPZ, or TFRecord)')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for results')
    parser.add_argument('--model_name', type=str, default='epibert_model',
                       help='Name for the model (used in output files)')
    parser.add_argument('--model_type', type=str, default='pretraining',
                       choices=['pretraining', 'finetuning'],
                       help='Type of model (pretraining or finetuning)')
    parser.add_argument('--implementation', type=str, default='tensorflow',
                       choices=['tensorflow', 'lightning'],
                       help='Implementation to use (tensorflow or lightning)')
    
    args = parser.parse_args()
    
    print("EpiBERT Model Evaluation")
    print("=" * 40)
    print(f"Model path: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print(f"Output directory: {args.output_dir}")
    print(f"Implementation: {args.implementation}")
    print(f"Model type: {args.model_type}")
    
    # Initialize evaluator
    evaluator = EpiBERTEvaluator(implementation=args.implementation)
    
    # Load model
    model = evaluator.load_model(args.model_path, args.model_type)
    
    # Run evaluation
    metrics = evaluator.evaluate_model(
        model=model,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")
    

if __name__ == "__main__":
    main()