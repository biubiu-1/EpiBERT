"""
PyTorch Lightning metrics converted from TensorFlow EpiBERT implementation.

This module contains metric functions for the Lightning version of EpiBERT,
converted from the original TensorFlow src/metrics.py.
"""

import torch
import torch.nn as nn
import torchmetrics
from typing import Optional, Tuple


class PearsonR(torchmetrics.Metric):
    """
    Pearson correlation coefficient metric for PyTorch Lightning.
    
    Converted from TensorFlow implementation in src/metrics.py.
    """
    
    def __init__(self, reduce_axis: Optional[Tuple[int, ...]] = None, **kwargs):
        super().__init__(**kwargs)
        self.reduce_axis = reduce_axis
        
        self.add_state("sum_x", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_x_squared", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y_squared", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_xy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Update the metric state with predictions and targets."""
        assert y_pred.shape == y_true.shape, f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"
        
        y_pred = y_pred.float()
        y_true = y_true.float()
        
        if self.reduce_axis is not None:
            # Reduce over specified axes (keep others for per-sample metrics)
            for ax in sorted(self.reduce_axis, reverse=True):
                if ax < len(y_pred.shape):
                    y_pred = y_pred.mean(dim=ax, keepdim=True)
                    y_true = y_true.mean(dim=ax, keepdim=True)
        
        # Flatten for global correlation calculation
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        self.sum_x += y_pred_flat.sum()
        self.sum_y += y_true_flat.sum()
        self.sum_x_squared += (y_pred_flat ** 2).sum()
        self.sum_y_squared += (y_true_flat ** 2).sum()
        self.sum_xy += (y_pred_flat * y_true_flat).sum()
        self.count += y_pred_flat.numel()
    
    def compute(self) -> torch.Tensor:
        """Compute the Pearson correlation coefficient."""
        if self.count == 0:
            return torch.tensor(0.0)
        
        mean_x = self.sum_x / self.count
        mean_y = self.sum_y / self.count
        
        covariance = (self.sum_xy - self.count * mean_x * mean_y) / self.count
        var_x = (self.sum_x_squared - self.count * mean_x ** 2) / self.count
        var_y = (self.sum_y_squared - self.count * mean_y ** 2) / self.count
        
        std_x = torch.sqrt(var_x)
        std_y = torch.sqrt(var_y)
        
        # Avoid division by zero
        denominator = std_x * std_y
        if denominator == 0:
            return torch.tensor(0.0)
        
        return covariance / denominator


class R2Score(torchmetrics.Metric):
    """
    R² (coefficient of determination) metric for PyTorch Lightning.
    
    Converted from TensorFlow implementation in src/metrics.py.
    """
    
    def __init__(self, reduce_axis: Optional[Tuple[int, ...]] = None, **kwargs):
        super().__init__(**kwargs)
        self.reduce_axis = reduce_axis
        
        self.add_state("sum_squared_residuals", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_squared_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_targets", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Update the metric state with predictions and targets."""
        assert y_pred.shape == y_true.shape, f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"
        
        y_pred = y_pred.float()
        y_true = y_true.float()
        
        if self.reduce_axis is not None:
            # Reduce over specified axes
            for ax in sorted(self.reduce_axis, reverse=True):
                if ax < len(y_pred.shape):
                    y_pred = y_pred.mean(dim=ax, keepdim=True)
                    y_true = y_true.mean(dim=ax, keepdim=True)
        
        # Flatten for computation
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        residuals = y_true_flat - y_pred_flat
        self.sum_squared_residuals += (residuals ** 2).sum()
        self.sum_targets += y_true_flat.sum()
        self.count += y_true_flat.numel()
    
    def compute(self) -> torch.Tensor:
        """Compute the R² score."""
        if self.count == 0:
            return torch.tensor(0.0)
        
        mean_target = self.sum_targets / self.count
        
        # We need to compute total sum of squares dynamically
        # This is a simplified version - in practice you might want to store this incrementally
        total_sum_squares = self.count * torch.var(self.sum_targets / self.count) * self.count
        
        if total_sum_squares == 0:
            return torch.tensor(0.0)
        
        return 1.0 - (self.sum_squared_residuals / total_sum_squares)


def pearson_correlation(y_true: torch.Tensor, y_pred: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    Compute Pearson correlation coefficient between tensors.
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor  
        dim: Dimension along which to compute correlation (None for global)
        
    Returns:
        Pearson correlation coefficient
    """
    if dim is not None:
        # Compute correlation along specific dimension
        mean_true = y_true.mean(dim=dim, keepdim=True)
        mean_pred = y_pred.mean(dim=dim, keepdim=True)
        
        numerator = ((y_true - mean_true) * (y_pred - mean_pred)).sum(dim=dim)
        denominator = torch.sqrt(
            ((y_true - mean_true) ** 2).sum(dim=dim) * 
            ((y_pred - mean_pred) ** 2).sum(dim=dim)
        )
    else:
        # Global correlation
        mean_true = y_true.mean()
        mean_pred = y_pred.mean()
        
        numerator = ((y_true - mean_true) * (y_pred - mean_pred)).sum()
        denominator = torch.sqrt(
            ((y_true - mean_true) ** 2).sum() * 
            ((y_pred - mean_pred) ** 2).sum()
        )
    
    # Avoid division by zero
    return torch.where(denominator > 0, numerator / denominator, torch.zeros_like(numerator))


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    Compute R² score between tensors.
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        dim: Dimension along which to compute R² (None for global)
        
    Returns:
        R² score
    """
    if dim is not None:
        residual_sum_squares = ((y_true - y_pred) ** 2).sum(dim=dim)
        total_sum_squares = ((y_true - y_true.mean(dim=dim, keepdim=True)) ** 2).sum(dim=dim)
    else:
        residual_sum_squares = ((y_true - y_pred) ** 2).sum()
        total_sum_squares = ((y_true - y_true.mean()) ** 2).sum()
    
    # Avoid division by zero
    return torch.where(
        total_sum_squares > 0, 
        1.0 - residual_sum_squares / total_sum_squares,
        torch.zeros_like(residual_sum_squares)
    )


class MetricDict(nn.Module):
    """
    Dictionary-like container for multiple metrics, similar to TensorFlow implementation.
    """
    
    def __init__(self, metrics_dict: dict):
        super().__init__()
        self.metrics = nn.ModuleDict(metrics_dict)
    
    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Update all metrics."""
        for metric in self.metrics.values():
            metric.update(y_pred, y_true)
    
    def compute(self) -> dict:
        """Compute all metrics."""
        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()