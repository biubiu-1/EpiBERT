"""
PyTorch Lightning training utilities converted from TensorFlow EpiBERT implementation.

This module contains training utility functions for the Lightning version of EpiBERT,
converted from the original TensorFlow training_utils_*.py files.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from .metrics import PearsonR, R2Score, MetricDict
from .losses import PoissonMultinomialLoss, MultiTaskLoss, create_multitask_loss


class EpiBERTTrainingMixin:
    """
    Mixin class providing training utilities for EpiBERT Lightning modules.
    
    This consolidates the key training functions from the original TensorFlow
    training_utils files into a reusable mixin.
    """
    
    def setup_metrics(self, predict_atac: bool = True, unmask_loss: bool = False) -> Dict[str, Any]:
        """
        Set up metrics for training and validation.
        
        Converted from TensorFlow metric initialization in training_utils files.
        
        Args:
            predict_atac: Whether to include ATAC prediction metrics
            unmask_loss: Whether to include unmasked loss metrics
            
        Returns:
            Dictionary of initialized metrics
        """
        metrics = {}
        
        # Core training metrics
        metrics['train_loss'] = pl.metrics.MeanMetric()
        metrics['val_loss'] = pl.metrics.MeanMetric() 
        
        # ATAC metrics
        if predict_atac:
            metrics['train_loss_atac'] = pl.metrics.MeanMetric()
            metrics['val_loss_atac'] = pl.metrics.MeanMetric()
            metrics['ATAC_PearsonR_tr'] = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})
            metrics['ATAC_R2_tr'] = MetricDict({'R2': R2Score(reduce_axis=(0, 1))})
            metrics['ATAC_PearsonR'] = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})
            metrics['ATAC_R2'] = MetricDict({'R2': R2Score(reduce_axis=(0, 1))})
            
            if unmask_loss:
                metrics['val_loss_atac_ho'] = pl.metrics.MeanMetric()
                metrics['ATAC_PearsonR_um'] = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})
                metrics['ATAC_R2_um'] = MetricDict({'R2': R2Score(reduce_axis=(0, 1))})
                metrics['ATAC_PearsonR_ho'] = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})
                metrics['ATAC_R2_ho'] = MetricDict({'R2': R2Score(reduce_axis=(0, 1))})
        
        # RNA metrics (for finetuning)
        if hasattr(self, 'predict_rna') and self.predict_rna:
            metrics['train_loss_rna'] = pl.metrics.MeanMetric()
            metrics['val_loss_rna'] = pl.metrics.MeanMetric()
            metrics['val_loss_ho'] = pl.metrics.MeanMetric()
            metrics['RNA_PearsonR_tr'] = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})
            metrics['RNA_R2_tr'] = MetricDict({'R2': R2Score(reduce_axis=(0, 1))})
            metrics['RNA_PearsonR'] = MetricDict({'PearsonR': PearsonR(reduce_axis=(0, 1))})
            metrics['RNA_R2'] = MetricDict({'R2': R2Score(reduce_axis=(0, 1))})
        
        return metrics
    
    def compute_loss(self, 
                    predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    masks: Optional[Dict[str, torch.Tensor]] = None,
                    loss_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss for EpiBERT predictions.
        
        Args:
            predictions: Dictionary of model predictions
            targets: Dictionary of target values
            masks: Optional dictionary of masks
            loss_weights: Optional dictionary of loss weights
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # ATAC loss (pretraining)
        if 'atac' in predictions and 'atac' in targets:
            if masks and 'atac' in masks:
                # Apply masking for pretraining
                mask = masks['atac']
                mask_indices = torch.where(mask == 1)
                
                # Extract masked positions
                target_atac = targets['atac'][mask_indices].unsqueeze(0).unsqueeze(-1)
                pred_atac = predictions['atac'][mask_indices].unsqueeze(0).unsqueeze(-1)
                
                atac_loss_fn = PoissonMultinomialLoss()
                losses['atac'] = atac_loss_fn(pred_atac, target_atac).mean()
            else:
                # Standard ATAC loss
                atac_loss_fn = PoissonMultinomialLoss()
                losses['atac'] = atac_loss_fn(predictions['atac'], targets['atac']).mean()
        
        # RNA loss (finetuning)
        if 'rna' in predictions and 'rna' in targets:
            rna_loss_fn = nn.PoissonNLLLoss(log_input=False, reduction='mean')
            losses['rna'] = rna_loss_fn(predictions['rna'], targets['rna'])
        
        # Combine losses
        if loss_weights is None:
            loss_weights = {'atac': 0.1, 'rna': 1.0}
        
        total_loss = sum(loss_weights.get(k, 1.0) * v for k, v in losses.items())
        losses['total'] = total_loss
        
        return losses
    
    def update_metrics(self,
                      predictions: Dict[str, torch.Tensor],
                      targets: Dict[str, torch.Tensor],
                      losses: Dict[str, torch.Tensor],
                      metrics: Dict[str, Any],
                      stage: str = 'train',
                      masks: Optional[Dict[str, torch.Tensor]] = None):
        """
        Update metrics during training/validation.
        
        Args:
            predictions: Model predictions
            targets: Target values  
            losses: Computed losses
            metrics: Metrics dictionary
            stage: Training stage ('train', 'val', 'test')
            masks: Optional masks
        """
        # Update loss metrics
        if 'total' in losses:
            metrics[f'{stage}_loss'].update(losses['total'])
        
        if 'atac' in losses:
            metrics[f'{stage}_loss_atac'].update(losses['atac'])
        
        if 'rna' in losses:
            metrics[f'{stage}_loss_rna'].update(losses['rna'])
        
        # Update performance metrics
        if 'atac' in predictions and 'atac' in targets:
            suffix = '_tr' if stage == 'train' else ''
            
            if masks and 'atac' in masks:
                # Use masked positions
                mask = masks['atac']
                mask_indices = torch.where(mask == 1)
                target_masked = targets['atac'][mask_indices].unsqueeze(0).unsqueeze(-1)
                pred_masked = predictions['atac'][mask_indices].unsqueeze(0).unsqueeze(-1)
                
                metrics[f'ATAC_PearsonR{suffix}'].update(pred_masked, target_masked)
                metrics[f'ATAC_R2{suffix}'].update(pred_masked, target_masked)
            else:
                # Use full predictions
                metrics[f'ATAC_PearsonR{suffix}'].update(predictions['atac'], targets['atac'])
                metrics[f'ATAC_R2{suffix}'].update(predictions['atac'], targets['atac'])
        
        if 'rna' in predictions and 'rna' in targets:
            suffix = '_tr' if stage == 'train' else ''
            metrics[f'RNA_PearsonR{suffix}'].update(predictions['rna'], targets['rna'])
            metrics[f'RNA_R2{suffix}'].update(predictions['rna'], targets['rna'])


class DataAugmentation:
    """
    Data augmentation utilities for genomic sequences.
    
    Converted from TensorFlow augmentation functions in training_utils files.
    """
    
    @staticmethod
    def reverse_complement_sequence(sequence: torch.Tensor) -> torch.Tensor:
        """
        Apply reverse complement to one-hot encoded DNA sequence.
        
        Args:
            sequence: One-hot encoded sequence of shape (..., 4)
            
        Returns:
            Reverse complemented sequence
        """
        # Reverse the sequence along the length dimension
        reversed_seq = torch.flip(sequence, [-2])
        
        # Complement: A<->T, C<->G (swap channels 0<->3, 1<->2)
        complement_map = torch.tensor([3, 2, 1, 0], device=sequence.device)
        return reversed_seq[..., complement_map]
    
    @staticmethod
    def random_shift_sequence(sequence: torch.Tensor, 
                            targets: torch.Tensor,
                            max_shift: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random shift to sequence and corresponding targets.
        
        Args:
            sequence: Input sequence
            targets: Target profiles
            max_shift: Maximum shift amount
            
        Returns:
            Shifted sequence and targets
        """
        if max_shift == 0:
            return sequence, targets
        
        shift = torch.randint(0, max_shift + 1, (1,)).item()
        
        if shift > 0:
            # Shift sequence
            sequence = sequence[..., shift:, :]
            # Adjust targets accordingly (implementation depends on specific model)
            # This is a simplified version
            target_shift = shift // 128  # Assuming 128bp resolution
            if target_shift > 0:
                targets = targets[..., target_shift:, :]
        
        return sequence, targets
    
    @staticmethod  
    def add_noise_to_atac(atac_profile: torch.Tensor,
                         noise_std: float = 1e-4) -> torch.Tensor:
        """
        Add Gaussian noise to ATAC profile.
        
        Args:
            atac_profile: ATAC-seq profile
            noise_std: Standard deviation of noise
            
        Returns:
            Noisy ATAC profile
        """
        noise = torch.abs(torch.normal(
            mean=noise_std,
            std=noise_std,
            size=atac_profile.shape,
            device=atac_profile.device
        ))
        return atac_profile + noise
    
    @staticmethod
    def mask_atac_profile(atac_profile: torch.Tensor,
                         mask_dropout: float = 0.15,
                         mask_size: int = 1536) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masked ATAC profile for pretraining.
        
        Args:
            atac_profile: Original ATAC profile
            mask_dropout: Fraction of profile to mask
            mask_size: Size of masked regions
            
        Returns:
            Tuple of (masked_profile, mask)
        """
        batch_size, seq_len, channels = atac_profile.shape
        
        # Create mask
        mask = torch.zeros_like(atac_profile[..., 0])  # Remove channel dim
        
        # Calculate number of positions to mask
        num_mask = int(seq_len * mask_dropout)
        
        for i in range(batch_size):
            # Random positions to mask
            mask_starts = torch.randperm(seq_len - mask_size)[:num_mask]
            
            for start in mask_starts:
                end = min(start + mask_size, seq_len)
                mask[i, start:end] = 1
        
        # Apply mask to profile
        masked_profile = atac_profile * (1 - mask.unsqueeze(-1))
        
        return masked_profile, mask


def one_hot_encode_sequence(sequence: str) -> torch.Tensor:
    """
    One-hot encode DNA sequence string.
    
    Args:
        sequence: DNA sequence string
        
    Returns:
        One-hot encoded tensor of shape (len, 4)
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}  # N maps to A
    
    indices = [mapping.get(base.upper(), 0) for base in sequence]
    one_hot = torch.zeros((len(sequence), 4))
    one_hot[range(len(sequence)), indices] = 1
    
    return one_hot


def create_training_functions(model: nn.Module,
                            loss_type: str = 'poisson_multinomial',
                            predict_atac: bool = True,
                            atac_scale: float = 0.1) -> Dict[str, Callable]:
    """
    Create training and validation functions for EpiBERT.
    
    Args:
        model: EpiBERT model
        loss_type: Type of loss function
        predict_atac: Whether to predict ATAC
        atac_scale: Scale factor for ATAC loss
        
    Returns:
        Dictionary of training functions
    """
    # Create loss function
    if predict_atac:
        loss_fn = create_multitask_loss(predict_atac=True, atac_weight=atac_scale)
    else:
        loss_fn = nn.PoissonNLLLoss(log_input=False, reduction='mean')
    
    def training_step(batch, batch_idx):
        """Training step function."""
        # Unpack batch based on task type
        if predict_atac:
            sequence, atac, mask, unmask, target, motif_activity = batch
            inputs = (sequence, atac, motif_activity)
        else:
            # For finetuning
            sequence, atac, mask, unmask, target_atac, motif_activity, target_rna, tss_tokens, gene_token, cell_type = batch
            inputs = (sequence, atac, motif_activity)
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss and metrics
        if predict_atac:
            predictions = {'atac': outputs}
            targets = {'atac': target}
            masks = {'atac': mask}
        else:
            if isinstance(outputs, tuple):
                pred_atac, pred_rna = outputs
                predictions = {'atac': pred_atac, 'rna': pred_rna}
                targets = {'atac': target_atac, 'rna': target_rna}
                masks = {'atac': mask}
            else:
                predictions = {'rna': outputs}
                targets = {'rna': target_rna}
                masks = None
        
        # Compute losses
        training_mixin = EpiBERTTrainingMixin()
        losses = training_mixin.compute_loss(predictions, targets, masks)
        
        return losses['total']
    
    def validation_step(batch, batch_idx):
        """Validation step function."""
        # Similar to training step but without gradients
        with torch.no_grad():
            return training_step(batch, batch_idx)
    
    return {
        'training_step': training_step,
        'validation_step': validation_step,
        'loss_fn': loss_fn
    }