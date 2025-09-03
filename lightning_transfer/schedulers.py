"""
PyTorch Lightning learning rate schedulers converted from TensorFlow EpiBERT implementation.

This module contains scheduler implementations for the Lightning version of EpiBERT,
converted from the original TensorFlow src/schedulers.py.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Union


class CosineDecayWithWarmup(_LRScheduler):
    """
    Cosine decay learning rate scheduler with linear warmup.
    
    Converted from TensorFlow implementation in src/schedulers.py.
    """
    
    def __init__(self,
                 optimizer,
                 warmup_steps: int,
                 decay_steps: int,
                 target_lr: Optional[float] = None,
                 alpha: float = 0.0,
                 last_epoch: int = -1,
                 return_constant: bool = False):
        """
        Initialize cosine decay scheduler with warmup.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            decay_steps: Total number of decay steps
            target_lr: Target learning rate at end of warmup (default: optimizer's lr)
            alpha: Minimum learning rate as fraction of target_lr
            last_epoch: The index of last epoch
            return_constant: Whether to return constant lr after warmup
        """
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.return_constant = return_constant
        
        # Set target learning rate
        if target_lr is None:
            self.target_lr = optimizer.param_groups[0]['lr']
        else:
            self.target_lr = target_lr
        
        # Initial learning rate is 0.1% of target
        self.initial_lr = 0.001 * self.target_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        current_step = self.last_epoch + 1
        
        lrs = []
        for base_lr in self.base_lrs:
            if current_step < self.warmup_steps:
                # Linear warmup phase
                completed_fraction = current_step / self.warmup_steps
                total_delta = self.target_lr - self.initial_lr
                lr = self.initial_lr + completed_fraction * total_delta
            else:
                if self.return_constant:
                    lr = self.target_lr
                else:
                    # Cosine decay phase
                    decay_step = min(current_step - self.warmup_steps, self.decay_steps)
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / self.decay_steps))
                    decayed = (1 - self.alpha) * cosine_decay + self.alpha
                    lr = self.target_lr * decayed
            
            lrs.append(lr)
        
        return lrs


class LinearWarmup(_LRScheduler):
    """
    Linear warmup learning rate scheduler.
    """
    
    def __init__(self,
                 optimizer,
                 warmup_steps: int,
                 target_lr: Optional[float] = None,
                 last_epoch: int = -1):
        """
        Initialize linear warmup scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            target_lr: Target learning rate at end of warmup
            last_epoch: The index of last epoch
        """
        self.warmup_steps = warmup_steps
        
        if target_lr is None:
            self.target_lr = optimizer.param_groups[0]['lr']
        else:
            self.target_lr = target_lr
        
        # Start from very small learning rate
        self.initial_lr = 1e-8
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        current_step = self.last_epoch + 1
        
        if current_step >= self.warmup_steps:
            return [self.target_lr for _ in self.base_lrs]
        
        # Linear interpolation during warmup
        lr_scale = current_step / self.warmup_steps
        return [self.initial_lr + lr_scale * (self.target_lr - self.initial_lr) 
                for _ in self.base_lrs]


class ExponentialDecay(_LRScheduler):
    """
    Exponential decay learning rate scheduler.
    """
    
    def __init__(self,
                 optimizer,
                 decay_steps: int,
                 decay_rate: float,
                 staircase: bool = False,
                 last_epoch: int = -1):
        """
        Initialize exponential decay scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            decay_steps: Steps for one decay period
            decay_rate: Decay rate
            staircase: Whether to use staircase (discrete) decay
            last_epoch: The index of last epoch
        """
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        current_step = self.last_epoch + 1
        
        if self.staircase:
            power = current_step // self.decay_steps
        else:
            power = current_step / self.decay_steps
        
        decay_factor = self.decay_rate ** power
        
        return [base_lr * decay_factor for base_lr in self.base_lrs]


class PolynomialDecay(_LRScheduler):
    """
    Polynomial decay learning rate scheduler.
    """
    
    def __init__(self,
                 optimizer,
                 decay_steps: int,
                 end_learning_rate: float = 0.0001,
                 power: float = 1.0,
                 cycle: bool = False,
                 last_epoch: int = -1):
        """
        Initialize polynomial decay scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            decay_steps: Number of steps to decay over
            end_learning_rate: Final learning rate
            power: Power of polynomial
            cycle: Whether to cycle the decay
            last_epoch: The index of last epoch
        """
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        current_step = self.last_epoch + 1
        
        lrs = []
        for base_lr in self.base_lrs:
            if self.cycle:
                # Cycling version
                multiplier = 1.0 if current_step == 0 else math.ceil(current_step / self.decay_steps)
                decay_steps = multiplier * self.decay_steps
                current_step = current_step % self.decay_steps if current_step % self.decay_steps != 0 else self.decay_steps
            else:
                # Non-cycling version
                current_step = min(current_step, self.decay_steps)
                decay_steps = self.decay_steps
            
            lr = (base_lr - self.end_learning_rate) * \
                 ((1 - current_step / decay_steps) ** self.power) + self.end_learning_rate
            
            lrs.append(lr)
        
        return lrs


def create_cosine_warmup_scheduler(optimizer,
                                 warmup_steps: int,
                                 decay_steps: int,
                                 target_lr: Optional[float] = None,
                                 alpha: float = 0.0) -> CosineDecayWithWarmup:
    """
    Create cosine decay scheduler with warmup for EpiBERT training.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        decay_steps: Number of decay steps
        target_lr: Target learning rate
        alpha: Minimum learning rate fraction
        
    Returns:
        Configured scheduler
    """
    return CosineDecayWithWarmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        target_lr=target_lr,
        alpha=alpha
    )


def create_linear_warmup_scheduler(optimizer,
                                 warmup_steps: int,
                                 target_lr: Optional[float] = None) -> LinearWarmup:
    """
    Create linear warmup scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        target_lr: Target learning rate
        
    Returns:
        Configured scheduler
    """
    return LinearWarmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        target_lr=target_lr
    )


def create_polynomial_decay_scheduler(optimizer,
                                    decay_steps: int,
                                    end_learning_rate: float = 0.0001,
                                    power: float = 1.0) -> PolynomialDecay:
    """
    Create polynomial decay scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        decay_steps: Number of decay steps
        end_learning_rate: Final learning rate
        power: Polynomial power
        
    Returns:
        Configured scheduler
    """
    return PolynomialDecay(
        optimizer=optimizer,
        decay_steps=decay_steps,
        end_learning_rate=end_learning_rate,
        power=power
    )


# Helper function to match TensorFlow's cosine decay function signature
def cos_w_warmup(current_step: int,
                target_warmup: float,
                warmup_steps: int,
                decay_steps: int,
                alpha: float,
                return_constant: bool = False) -> float:
    """
    Standalone function to compute learning rate with cosine decay and warmup.
    
    Matches the original TensorFlow implementation signature.
    
    Args:
        current_step: Current training step
        target_warmup: Target learning rate at end of warmup
        warmup_steps: Number of warmup steps
        decay_steps: Total decay steps
        alpha: Minimum learning rate fraction
        return_constant: Whether to return constant rate after warmup
        
    Returns:
        Learning rate for current step
    """
    initial_learning_rate = 0.001 * target_warmup
    
    if current_step < warmup_steps:
        # Linear warm-up phase
        completed_fraction = current_step / warmup_steps
        total_delta = target_warmup - initial_learning_rate
        lr = initial_learning_rate + completed_fraction * total_delta
    else:
        if return_constant:
            lr = target_warmup
        else:
            # Decay phase
            decay_step = min(current_step - warmup_steps, decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / decay_steps))
            decayed = (1 - alpha) * cosine_decay + alpha
            lr = target_warmup * decayed
    
    return lr