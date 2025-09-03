"""
EpiBERT PyTorch Lightning Implementation

This demonstrates how the TensorFlow-based EpiBERT model could be structured
in PyTorch Lightning, using the transferred utilities from the original implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torchmetrics
from typing import Dict, Any, Optional, List, Tuple
import math

# Import our transferred utilities
from .utils import (
    sinusoidal_positional_encoding, 
    RotaryPositionalEmbedding, 
    SoftmaxPooling1D,
    gen_channels_list,
    exponential_linspace_int
)
from .metrics import PearsonR, R2Score, MetricDict
from .losses import create_multitask_loss, PoissonMultinomialLoss
from .optimizers import create_adafactor_optimizer, create_adamw_optimizer
from .schedulers import create_cosine_warmup_scheduler
from .training_utils import EpiBERTTrainingMixin


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation"""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int, 
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: str = 'same',
                 dilation: int = 1,
                 bn_momentum: float = 0.9):
        super().__init__()
        
        if padding == 'same':
            padding = (kernel_size - 1) // 2 * dilation
            
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels, momentum=1-bn_momentum)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary positional embeddings"""
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 use_rotary: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rotary = use_rotary
        
        assert d_model % num_heads == 0
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        if use_rotary:
            self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
            
    def apply_rotary_embeddings(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to input tensor"""
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        # Split into even and odd dimensions
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Apply rotation
        rotated = torch.cat([
            x1 * cos_emb - x2 * sin_emb,
            x1 * sin_emb + x2 * cos_emb
        ], dim=-1)
        
        return rotated
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if self.use_rotary:
            rotary_emb = self.rotary_emb(x)
            q = self.apply_rotary_embeddings(q, rotary_emb)
            k = self.apply_rotary_embeddings(k, rotary_emb)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward network"""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 ff_dim: int,
                 dropout: float = 0.1,
                 use_rotary: bool = True):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout, use_rotary)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class EpiBERTLightning(pl.LightningModule, EpiBERTTrainingMixin):
    """
    EpiBERT PyTorch Lightning Module
    
    This demonstrates how the TensorFlow EpiBERT model could be structured
    in PyTorch Lightning, using the transferred utilities from the original implementation.
    """
    
    def __init__(self,
                 input_length: int = 524288,
                 output_length: int = 4096,
                 num_heads: int = 8,
                 num_transformer_layers: int = 8,
                 d_model: int = 1024,
                 filter_list_seq: List[int] = [512, 640, 640, 768, 896, 1024],
                 filter_list_atac: List[int] = [32, 64],
                 num_motifs: int = 693,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 1e-4,
                 warmup_steps: int = 1000,
                 total_steps: int = 100000,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture parameters
        self.input_length = input_length
        self.output_length = output_length
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.d_model = d_model
        self.filter_list_seq = filter_list_seq
        self.filter_list_atac = filter_list_atac
        self.num_motifs = num_motifs
        self.dropout_rate = dropout_rate
        
        # Training parameters
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        # Build model components
        self._build_model()
        
        # Initialize metrics using our transferred utilities
        self.metrics = self.setup_metrics(predict_atac=True, unmask_loss=False)
        
    def _build_model(self):
        """Build the EpiBERT model architecture"""
        
        # Sequence processing stem
        self.seq_stem = nn.Sequential(
            nn.Conv1d(4, self.filter_list_seq[0], kernel_size=15, padding=7),
            nn.BatchNorm1d(self.filter_list_seq[0]),
            nn.GELU(),
            SoftmaxPooling1D(kernel_size=2)
        )
        
        # ATAC processing stem  
        self.atac_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=50, padding=25),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Sequence convolutional tower
        seq_layers = []
        in_channels = self.filter_list_seq[0]
        for out_channels in self.filter_list_seq:
            seq_layers.extend([
                ConvBlock(in_channels, out_channels, kernel_size=5),
                SoftmaxPooling1D(kernel_size=2)
            ])
            in_channels = out_channels
        self.seq_tower = nn.Sequential(*seq_layers)
        
        # ATAC convolutional tower
        atac_layers = []
        in_channels = 32
        for out_channels in self.filter_list_atac:
            atac_layers.extend([
                ConvBlock(in_channels, out_channels, kernel_size=5),
                nn.MaxPool1d(kernel_size=4)
            ])
            in_channels = out_channels
        self.atac_tower = nn.Sequential(*atac_layers)
        
        # Motif activity processing
        self.motif_fc = nn.Sequential(
            nn.Linear(self.num_motifs, 64),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 32)
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.d_model * 4,
                dropout=self.dropout_rate,
                use_rotary=True
            ) for _ in range(self.num_transformer_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 256),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 1)
        )
        
    def forward(self, 
                sequence: torch.Tensor,
                atac: torch.Tensor, 
                motif_activity: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EpiBERT model
        
        Args:
            sequence: DNA sequence tensor (batch_size, 4, input_length)
            atac: ATAC-seq signal tensor (batch_size, 1, input_length) 
            motif_activity: Motif activity tensor (batch_size, num_motifs)
            
        Returns:
            Predicted ATAC signal (batch_size, output_length, 1)
        """
        
        # Process sequence
        seq_features = self.seq_stem(sequence)
        seq_features = self.seq_tower(seq_features)
        
        # Process ATAC 
        atac_features = self.atac_stem(atac)
        atac_features = self.atac_tower(atac_features)
        
        # Process motif activity
        motif_features = self.motif_fc(motif_activity)
        
        # Combine features - this is a simplified combination
        # In the actual implementation, you'd need more sophisticated feature fusion
        combined_features = seq_features + atac_features  # Simple addition for demo
        
        # Transpose for transformer (batch, seq_len, features)
        combined_features = combined_features.transpose(1, 2)
        
        # Apply transformer layers
        for transformer in self.transformer_layers:
            combined_features = transformer(combined_features)
            
        # Generate output
        output = self.output_head(combined_features)
        
        return output
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step using transferred utilities"""
        sequence = batch['sequence']
        atac = batch['atac'] 
        target = batch['target']
        motif_activity = batch['motif_activity']
        mask = batch.get('mask', None)
        
        # Forward pass
        predictions = self(sequence, atac, motif_activity)
        
        # Prepare data for loss computation using transferred utilities
        pred_dict = {'atac': predictions}
        target_dict = {'atac': target}
        mask_dict = {'atac': mask} if mask is not None else None
        
        # Compute loss using transferred utilities
        losses = self.compute_loss(pred_dict, target_dict, mask_dict)
        
        # Update metrics using transferred utilities
        self.update_metrics(pred_dict, target_dict, losses, self.metrics, 'train', mask_dict)
        
        # Log metrics
        self.log('train_loss', losses['total'], prog_bar=True)
        
        return losses['total']
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step using transferred utilities"""
        sequence = batch['sequence']
        atac = batch['atac']
        target = batch['target'] 
        motif_activity = batch['motif_activity']
        mask = batch.get('mask', None)
        
        # Forward pass
        predictions = self(sequence, atac, motif_activity)
        
        # Prepare data for loss computation using transferred utilities
        pred_dict = {'atac': predictions}
        target_dict = {'atac': target}
        mask_dict = {'atac': mask} if mask is not None else None
        
        # Compute loss using transferred utilities
        losses = self.compute_loss(pred_dict, target_dict, mask_dict)
        
        # Update metrics using transferred utilities
        self.update_metrics(pred_dict, target_dict, losses, self.metrics, 'val', mask_dict)
        
        # Log metrics
        self.log('val_loss', losses['total'], prog_bar=True, sync_dist=True)
        
        return losses['total']
        
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler using transferred utilities"""
        # Use our transferred Adafactor optimizer (or AdamW as fallback)
        try:
            optimizer = create_adafactor_optimizer(
                self.parameters(),
                learning_rate=self.learning_rate,
                weight_decay=0.01
            )
        except:
            # Fallback to AdamW if Adafactor has issues
            optimizer = create_adamw_optimizer(
                self.parameters(),
                learning_rate=self.learning_rate,
                weight_decay=0.01
            )
        
        # Use our transferred cosine warmup scheduler
        scheduler = create_cosine_warmup_scheduler(
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
            decay_steps=self.total_steps - self.warmup_steps,
            target_lr=self.learning_rate,
            alpha=0.0
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


def create_trainer(max_epochs: int = 100,
                   gpus: int = 1, 
                   accumulate_grad_batches: int = 1,
                   precision: str = "16-mixed",
                   log_every_n_steps: int = 50) -> pl.Trainer:
    """Create a PyTorch Lightning trainer with appropriate callbacks"""
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        filename='epibert-{epoch:02d}-{val_loss:.4f}',
        auto_insert_metric_name=False
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Logger
    wandb_logger = WandbLogger(
        project="epibert-lightning",
        log_model="all"
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=gpus,
        accelerator='gpu' if gpus > 0 else 'cpu',
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=wandb_logger,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    model = EpiBERTLightning(
        input_length=524288,
        output_length=4096,
        num_heads=8,
        num_transformer_layers=8,
        learning_rate=1e-4
    )
    
    # Print model summary
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")