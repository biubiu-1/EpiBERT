"""
EpiBERT PyTorch Lightning Implementation

This demonstrates how the TensorFlow-based EpiBERT model could be structured
in PyTorch Lightning. This is a structural example showing the key components
and training loop organization.
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


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding implementation in PyTorch"""
    
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute the rotary embedding
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[:seq_len]


class SoftmaxPooling1D(nn.Module):
    """Softmax-based pooling layer (equivalent to TensorFlow implementation)"""
    
    def __init__(self, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, length)
        batch_size, channels, length = x.shape
        
        # Reshape for pooling
        x = x.view(batch_size, channels, -1, self.kernel_size)
        weights = F.softmax(x, dim=-1)
        pooled = torch.sum(x * weights, dim=-1)
        
        return pooled


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


class EpiBERTLightning(pl.LightningModule):
    """
    EpiBERT PyTorch Lightning Module
    
    This demonstrates how the TensorFlow EpiBERT model could be structured
    in PyTorch Lightning, maintaining the same architectural principles.
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
        
        # Initialize metrics
        self.train_pearson = torchmetrics.PearsonCorrCoef()
        self.val_pearson = torchmetrics.PearsonCorrCoef()
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        
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
        """Training step"""
        sequence = batch['sequence']
        atac = batch['atac'] 
        target = batch['target']
        motif_activity = batch['motif_activity']
        mask = batch.get('mask', None)
        
        # Forward pass
        predictions = self(sequence, atac, motif_activity)
        
        # Calculate loss (Poisson loss for count data)
        if mask is not None:
            # Apply mask to focus on specific regions
            predictions = predictions * mask.unsqueeze(-1)
            target = target * mask.unsqueeze(-1)
            
        loss = F.poisson_nll_loss(predictions.squeeze(-1), target, log_input=False)
        
        # Calculate metrics
        pred_flat = predictions.view(-1)
        target_flat = target.view(-1)
        
        # Only calculate metrics on valid (non-masked) positions
        if mask is not None:
            valid_mask = mask.view(-1) > 0
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]
            
        train_pearson = self.train_pearson(pred_flat, target_flat)
        train_r2 = self.train_r2(pred_flat, target_flat)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_pearson', train_pearson, prog_bar=True)
        self.log('train_r2', train_r2)
        
        return loss
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        sequence = batch['sequence']
        atac = batch['atac']
        target = batch['target'] 
        motif_activity = batch['motif_activity']
        mask = batch.get('mask', None)
        
        # Forward pass
        predictions = self(sequence, atac, motif_activity)
        
        # Calculate loss
        if mask is not None:
            predictions = predictions * mask.unsqueeze(-1)
            target = target * mask.unsqueeze(-1)
            
        val_loss = F.poisson_nll_loss(predictions.squeeze(-1), target, log_input=False)
        
        # Calculate metrics
        pred_flat = predictions.view(-1)
        target_flat = target.view(-1)
        
        if mask is not None:
            valid_mask = mask.view(-1) > 0
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]
            
        val_pearson = self.val_pearson(pred_flat, target_flat)
        val_r2 = self.val_r2(pred_flat, target_flat)
        
        # Log metrics
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)
        self.log('val_pearson', val_pearson, prog_bar=True, sync_dist=True)
        self.log('val_r2', val_r2, sync_dist=True)
        
        return val_loss
        
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Cosine annealing with warmup (similar to original)
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
                
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
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