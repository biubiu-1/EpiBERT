"""
PyTorch Lightning optimizers converted from TensorFlow EpiBERT implementation.

This module contains optimizer implementations for the Lightning version of EpiBERT,
converted from the original TensorFlow src/optimizers.py.
"""

import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import math
from typing import Any, Dict, Optional, Tuple


class AdafactorOptimizer(Optimizer):
    """
    PyTorch implementation of Adafactor optimizer.
    
    Converted from TensorFlow implementation in src/optimizers.py.
    
    Adafactor is described in https://arxiv.org/abs/1804.04235.
    This optimizer is particularly well-suited for training large transformer models
    with memory efficiency improvements over Adam.
    """
    
    def __init__(self,
                 params,
                 lr: Optional[float] = None,
                 beta2: float = -0.8,
                 cliping_threshold: float = 1.0,
                 decay_rate: float = -0.8,
                 beta1: Optional[float] = None,
                 weight_decay: float = 0.0,
                 scale_parameter: bool = True,
                 relative_step: bool = True,
                 eps: float = 1e-30):
        """
        Initialize Adafactor optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: External learning rate (default: None for relative step size)
            beta2: Exponential decay rate for second moment estimation
            cliping_threshold: Threshold for update clipping
            decay_rate: Rate for relative step size decay
            beta1: Exponential decay rate for first moment (None disables momentum)
            weight_decay: Weight decay coefficient
            scale_parameter: Whether to scale updates by parameter scale
            relative_step: Whether to use relative step size
            eps: Small value to avoid division by zero
        """
        if lr is not None and relative_step:
            raise ValueError("Cannot use external learning rate with relative_step=True")
        
        defaults = dict(
            lr=lr,
            beta2=beta2,
            cliping_threshold=cliping_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            eps=eps
        )
        super().__init__(params, defaults)
    
    def _get_lr(self, param_group: Dict[str, Any], param_state: Dict[str, Any]) -> float:
        """Compute learning rate based on current step and parameter scale."""
        min_step = 1e-6 * param_state['step'] if param_group['scale_parameter'] else 1e-2
        rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
        param_scale = 1.0
        
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps'], param_state['RMS'])
        
        return param_scale * rel_step_sz
    
    def _get_options(self, param_group: Dict[str, Any], param_shape: torch.Size) -> Tuple[bool, bool]:
        """Determine factorization options based on parameter shape."""
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment
    
    def _rms(self, tensor: torch.Tensor) -> float:
        """Compute RMS of tensor."""
        return tensor.norm(2) / (tensor.numel() ** 0.5)
    
    def _approx_sq_grad(self, exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        """Approximation of exponential moving average of square of gradient."""
        r_factor = ((exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
                   .rsqrt_().unsqueeze(-1).clamp_(0, math.inf))
        c_factor = (exp_avg_sq_col.rsqrt()).unsqueeze(0).clamp_(0, math.inf)
        return torch.mul(r_factor, c_factor)
    
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                grad_shape = grad.shape
                
                factored, use_first_moment = self._get_options(group, grad_shape)
                
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    if use_first_moment:
                        state['exp_avg'] = torch.zeros_like(grad).float()
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1]).float()
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).float()
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad).float()
                    
                    state['RMS'] = 0
                
                p_data_fp32 = p.data.float() if p.data.dtype in {torch.float16, torch.bfloat16} else p.data
                
                state['step'] += 1
                state['RMS'] = self._rms(p_data_fp32)
                
                lr = group['lr']
                if group['relative_step']:
                    lr = self._get_lr(group, state)
                
                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = grad**2 + group['eps']
                
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    
                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)
                    
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)
                
                update.div_((self._rms(update) / group['cliping_threshold']).clamp_(min=1.0))
                
                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(update, alpha=1 - group['beta1'])
                    update = exp_avg
                
                if group['weight_decay'] != 0:
                    p_data_fp32.mul_(1 - group['weight_decay'] * lr)
                
                p_data_fp32.add_(update, alpha=-lr)
                
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)
        
        return loss


def create_adafactor_optimizer(model_parameters,
                              learning_rate: Optional[float] = None,
                              weight_decay: float = 0.0,
                              beta2: float = -0.8,
                              cliping_threshold: float = 1.0,
                              decay_rate: float = -0.8,
                              beta1: Optional[float] = None,
                              relative_step: bool = True,
                              scale_parameter: bool = True) -> AdafactorOptimizer:
    """
    Create Adafactor optimizer with EpiBERT-specific defaults.
    
    Args:
        model_parameters: Model parameters to optimize
        learning_rate: Learning rate (None for relative step)
        weight_decay: Weight decay coefficient
        beta2: Second moment decay rate
        cliping_threshold: Gradient clipping threshold
        decay_rate: Decay rate for relative step size
        beta1: First moment decay rate (None disables momentum)
        relative_step: Whether to use relative step size
        scale_parameter: Whether to scale by parameter scale
        
    Returns:
        Configured Adafactor optimizer
    """
    return AdafactorOptimizer(
        model_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        beta2=beta2,
        cliping_threshold=cliping_threshold,
        decay_rate=decay_rate,
        beta1=beta1,
        relative_step=relative_step,
        scale_parameter=scale_parameter
    )


def create_adam_optimizer(model_parameters,
                         learning_rate: float = 1e-4,
                         weight_decay: float = 0.0,
                         beta1: float = 0.9,
                         beta2: float = 0.999,
                         eps: float = 1e-8) -> optim.Adam:
    """
    Create Adam optimizer with EpiBERT-specific defaults.
    
    Args:
        model_parameters: Model parameters to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Small value for numerical stability
        
    Returns:
        Configured Adam optimizer
    """
    return optim.Adam(
        model_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=eps
    )


def create_adamw_optimizer(model_parameters,
                          learning_rate: float = 1e-4,
                          weight_decay: float = 0.01,
                          beta1: float = 0.9,
                          beta2: float = 0.999,
                          eps: float = 1e-8) -> optim.AdamW:
    """
    Create AdamW optimizer with EpiBERT-specific defaults.
    
    Args:
        model_parameters: Model parameters to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay coefficient (decoupled)
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Small value for numerical stability
        
    Returns:
        Configured AdamW optimizer
    """
    return optim.AdamW(
        model_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=eps
    )