"""
PyTorch Lightning loss functions converted from TensorFlow EpiBERT implementation.

This module contains loss functions for the Lightning version of EpiBERT,
converted from the original TensorFlow src/losses.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PoissonMultinomialLoss(nn.Module):
    """
    Poisson multinomial loss for genomic sequence prediction.
    
    Converted from TensorFlow implementation in src/losses.py.
    This loss combines a Poisson loss for total counts with a multinomial
    loss for the distribution across the sequence.
    """
    
    def __init__(self, total_weight: float = 0.15, epsilon: float = 1e-6, rescale: bool = True):
        super().__init__()
        self.total_weight = total_weight
        self.epsilon = epsilon
        self.rescale = rescale
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the poisson multinomial loss.
        
        Args:
            y_pred: Predicted values of shape (batch, length, channels)
            y_true: True values of shape (batch, length, channels)
            
        Returns:
            Loss tensor of shape (batch, channels)
        """
        # Add epsilon to protect against tiny values
        y_true = y_true + self.epsilon
        y_pred = y_pred + self.epsilon
        
        seq_len = y_true.shape[1]
        
        # Sum across sequence length dimension
        s_true = y_true.sum(dim=1, keepdim=True)  # (batch, 1, channels)
        s_pred = y_pred.sum(dim=1, keepdim=True)  # (batch, 1, channels)
        
        # Normalize to sum to one (multinomial probabilities)
        p_pred = y_pred / s_pred  # (batch, length, channels)
        
        # Total count Poisson loss
        poisson_term = F.poisson_nll_loss(
            s_pred.squeeze(1), s_true.squeeze(1), 
            log_input=False, full=False, reduction='none'
        )  # (batch, channels)
        poisson_term = poisson_term / seq_len
        
        # Multinomial loss (negative log likelihood)
        log_p_pred = torch.log(p_pred)  # (batch, length, channels)
        multinomial_dot = -y_true * log_p_pred  # (batch, length, channels)
        multinomial_term = multinomial_dot.sum(dim=1)  # (batch, channels)
        multinomial_term = multinomial_term / seq_len
        
        # Combine losses
        loss_raw = multinomial_term + self.total_weight * poisson_term
        
        if self.rescale:
            loss_rescale = loss_raw * 2.0 / (1.0 + self.total_weight)
        else:
            loss_rescale = loss_raw
        
        return loss_rescale


class RegularMSELoss(nn.Module):
    """
    Regular Mean Squared Error loss.
    
    Converted from TensorFlow implementation in src/losses.py.
    """
    
    def __init__(self, reduction: str = 'none'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss."""
        return self.mse(y_pred, y_true)


class PoissonLoss(nn.Module):
    """
    Poisson loss for count data.
    
    Converted from TensorFlow implementation in src/losses.py.
    """
    
    def __init__(self, reduction: str = 'none', log_input: bool = False):
        super().__init__()
        self.reduction = reduction
        self.log_input = log_input
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute Poisson loss."""
        return F.poisson_nll_loss(
            y_pred, y_true,
            log_input=self.log_input,
            full=False,
            reduction=self.reduction
        )


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining different loss types for different outputs.
    
    This is useful for models that predict multiple targets (e.g., ATAC-seq and RNA-seq).
    """
    
    def __init__(self, 
                 loss_functions: dict,
                 loss_weights: Optional[dict] = None):
        """
        Initialize multi-task loss.
        
        Args:
            loss_functions: Dictionary mapping task names to loss functions
            loss_weights: Dictionary mapping task names to loss weights
        """
        super().__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)
        self.loss_weights = loss_weights or {name: 1.0 for name in loss_functions.keys()}
    
    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dictionary mapping task names to predictions
            targets: Dictionary mapping task names to targets
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        total_loss = 0.0
        
        for task_name, loss_fn in self.loss_functions.items():
            if task_name in predictions and task_name in targets:
                task_loss = loss_fn(predictions[task_name], targets[task_name])
                # Take mean if loss has multiple dimensions
                if task_loss.dim() > 0:
                    task_loss = task_loss.mean()
                
                losses[f'loss_{task_name}'] = task_loss
                total_loss += self.loss_weights[task_name] * task_loss
        
        losses['loss_total'] = total_loss
        return losses


class MaskedLoss(nn.Module):
    """
    Wrapper to apply loss only to non-masked positions.
    
    Useful for masked language modeling tasks in genomics.
    """
    
    def __init__(self, base_loss: nn.Module, mask_token_id: Optional[int] = None):
        super().__init__()
        self.base_loss = base_loss
        self.mask_token_id = mask_token_id
    
    def forward(self, 
                y_pred: torch.Tensor, 
                y_true: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute masked loss.
        
        Args:
            y_pred: Predictions
            y_true: Targets
            mask: Boolean mask (True for positions to include in loss)
            
        Returns:
            Masked loss
        """
        loss = self.base_loss(y_pred, y_true)
        
        if mask is not None:
            # Apply mask
            if mask.dim() < loss.dim():
                # Broadcast mask to match loss dimensions
                for _ in range(loss.dim() - mask.dim()):
                    mask = mask.unsqueeze(-1)
            
            masked_loss = loss * mask.float()
            # Return average loss over unmasked positions
            return masked_loss.sum() / mask.float().sum().clamp(min=1.0)
        
        return loss.mean()


# Convenience functions for common loss combinations
def create_atac_loss(total_weight: float = 0.15, epsilon: float = 1e-6) -> PoissonMultinomialLoss:
    """Create loss function for ATAC-seq prediction."""
    return PoissonMultinomialLoss(total_weight=total_weight, epsilon=epsilon)


def create_rna_loss() -> PoissonLoss:
    """Create loss function for RNA-seq prediction.""" 
    return PoissonLoss(reduction='none', log_input=False)


def create_multitask_loss(predict_atac: bool = True, 
                         atac_weight: float = 0.1,
                         rna_weight: float = 1.0) -> MultiTaskLoss:
    """
    Create multi-task loss for combined ATAC and RNA prediction.
    
    Args:
        predict_atac: Whether to include ATAC prediction
        atac_weight: Weight for ATAC loss
        rna_weight: Weight for RNA loss
        
    Returns:
        MultiTaskLoss instance
    """
    loss_functions = {'rna': create_rna_loss()}
    loss_weights = {'rna': rna_weight}
    
    if predict_atac:
        loss_functions['atac'] = create_atac_loss()
        loss_weights['atac'] = atac_weight
    
    return MultiTaskLoss(loss_functions, loss_weights)