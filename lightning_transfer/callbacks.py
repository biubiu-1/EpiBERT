"""
PyTorch Lightning callbacks for EpiBERT training
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import os
import logging

logger = logging.getLogger(__name__)


class EpiBERTCheckpointCallback(ModelCheckpoint):
    """Custom checkpoint callback for EpiBERT models"""
    
    def __init__(
        self,
        dirpath: str = "models/checkpoints",
        filename: str = "epibert-{epoch:02d}-{val_loss:.2f}",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        every_n_epochs: int = 1,
        **kwargs
    ):
        """
        Initialize EpiBERT checkpoint callback
        
        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename pattern
            monitor: Metric to monitor for saving
            mode: Whether to maximize or minimize the monitored metric
            save_top_k: Number of best models to save
            save_last: Whether to save the last checkpoint
            every_n_epochs: Save checkpoint every N epochs
        """
        os.makedirs(dirpath, exist_ok=True)
        
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            every_n_epochs=every_n_epochs,
            **kwargs
        )
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Add custom metadata to checkpoint"""
        checkpoint['model_type'] = getattr(pl_module, 'model_type', 'unknown')
        checkpoint['epibert_version'] = '1.0.0'
        return checkpoint


class EpiBERTEarlyStopping(EarlyStopping):
    """Custom early stopping callback for EpiBERT"""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.001,
        **kwargs
    ):
        """
        Initialize EpiBERT early stopping
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait before stopping
            mode: Whether to maximize or minimize the monitored metric
            min_delta: Minimum change to qualify as improvement
        """
        super().__init__(
            monitor=monitor,
            patience=patience,
            mode=mode,
            min_delta=min_delta,
            **kwargs
        )


class EpiBERTProgressBar(TQDMProgressBar):
    """Custom progress bar for EpiBERT training"""
    
    def __init__(self, refresh_rate: int = 1):
        super().__init__(refresh_rate=refresh_rate)
    
    def get_metrics(self, trainer, pl_module):
        """Add custom metrics to progress bar"""
        items = super().get_metrics(trainer, pl_module)
        
        # Add learning rate if available
        if hasattr(pl_module, 'learning_rate'):
            items['lr'] = f"{pl_module.learning_rate:.2e}"
        
        return items


def get_default_callbacks(
    checkpoint_dir: str = "models/checkpoints",
    monitor_metric: str = "val_loss",
    patience: int = 10,
    save_top_k: int = 3
) -> list:
    """
    Get default callbacks for EpiBERT training
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor_metric: Metric to monitor
        patience: Early stopping patience
        save_top_k: Number of best models to save
        
    Returns:
        List of callbacks
    """
    callbacks = [
        EpiBERTCheckpointCallback(
            dirpath=checkpoint_dir,
            monitor=monitor_metric,
            save_top_k=save_top_k,
            mode="min" if "loss" in monitor_metric else "max"
        ),
        EpiBERTEarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            mode="min" if "loss" in monitor_metric else "max"
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EpiBERTProgressBar()
    ]
    
    return callbacks


class MetricsLogger(pl.Callback):
    """Callback to log metrics to file"""
    
    def __init__(self, log_dir: str = "logs/metrics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
    def on_validation_end(self, trainer, pl_module):
        """Log validation metrics"""
        if trainer.logged_metrics:
            # Log metrics to file
            epoch = trainer.current_epoch
            metrics_file = os.path.join(self.log_dir, "validation_metrics.txt")
            
            with open(metrics_file, "a") as f:
                f.write(f"Epoch {epoch}: {trainer.logged_metrics}\n")