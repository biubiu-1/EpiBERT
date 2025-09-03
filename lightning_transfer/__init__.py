"""
PyTorch Lightning transfer utilities for EpiBERT.

This package contains all the utilities needed to run EpiBERT training and 
finetuning using PyTorch Lightning framework, converted from the original 
TensorFlow implementation.

Modules:
    utils: Core utility functions (positional embeddings, channel generation, etc.)
    metrics: Performance metrics (Pearson correlation, R², etc.)
    losses: Loss functions (Poisson multinomial, MSE, Poisson, etc.)
    optimizers: Optimizers (Adafactor, Adam, AdamW)
    schedulers: Learning rate schedulers (cosine decay with warmup, etc.)
    training_utils: Training and validation utilities
    epibert_lightning: Main Lightning model implementation
    data_module: Data loading and preprocessing
    train_lightning: Training script
"""

from .utils import (
    sinusoidal_positional_encoding,
    gen_channels_list,
    exponential_linspace_int,
    SoftmaxPooling1D,
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb
)

from .metrics import (
    PearsonR,
    R2Score,
    MetricDict,
    pearson_correlation,
    r2_score
)

from .losses import (
    PoissonMultinomialLoss,
    RegularMSELoss,
    PoissonLoss,
    MultiTaskLoss,
    MaskedLoss,
    create_atac_loss,
    create_rna_loss,
    create_multitask_loss
)

from .optimizers import (
    AdafactorOptimizer,
    create_adafactor_optimizer,
    create_adam_optimizer,
    create_adamw_optimizer
)

from .schedulers import (
    CosineDecayWithWarmup,
    LinearWarmup,
    ExponentialDecay,
    PolynomialDecay,
    create_cosine_warmup_scheduler,
    create_linear_warmup_scheduler,
    create_polynomial_decay_scheduler,
    cos_w_warmup
)

from .training_utils import (
    EpiBERTTrainingMixin,
    DataAugmentation,
    one_hot_encode_sequence,
    create_training_functions
)

__version__ = "1.0.0"
__author__ = "EpiBERT Lightning Transfer"

__all__ = [
    # Utils
    'sinusoidal_positional_encoding',
    'gen_channels_list', 
    'exponential_linspace_int',
    'SoftmaxPooling1D',
    'RotaryPositionalEmbedding',
    'apply_rotary_pos_emb',
    
    # Metrics
    'PearsonR',
    'R2Score', 
    'MetricDict',
    'pearson_correlation',
    'r2_score',
    
    # Losses
    'PoissonMultinomialLoss',
    'RegularMSELoss',
    'PoissonLoss',
    'MultiTaskLoss',
    'MaskedLoss',
    'create_atac_loss',
    'create_rna_loss',
    'create_multitask_loss',
    
    # Optimizers
    'AdafactorOptimizer',
    'create_adafactor_optimizer',
    'create_adam_optimizer', 
    'create_adamw_optimizer',
    
    # Schedulers
    'CosineDecayWithWarmup',
    'LinearWarmup',
    'ExponentialDecay',
    'PolynomialDecay',
    'create_cosine_warmup_scheduler',
    'create_linear_warmup_scheduler',
    'create_polynomial_decay_scheduler',
    'cos_w_warmup',
    
    # Training utilities
    'EpiBERTTrainingMixin',
    'DataAugmentation',
    'one_hot_encode_sequence',
    'create_training_functions'
]