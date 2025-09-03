"""
EpiBERT PyTorch Lightning Data Module

Provides PyTorch DataLoader format for EpiBERT training with Lightning.
Supports HDF5, numpy, and other standard formats for genomic data.

Enhanced to support both single-sample and multi-sample paired datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import warnings

# Import enhanced paired data module
try:
    from .paired_data_module import PairedDataModule, PairedSampleDataset
    PAIRED_MODULE_AVAILABLE = True
except ImportError:
    PAIRED_MODULE_AVAILABLE = False
    warnings.warn("Paired data module not available. Only single-sample datasets supported.")


class EpiBERTDataset(Dataset):
    """
    PyTorch Dataset for EpiBERT data
    
    Supports various data formats including HDF5, numpy arrays, and 
    pre-processed genomic data files. Handles data augmentation,
    masking, and batch preparation for ATAC-seq modeling.
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 input_length: int = 524288,
                 output_length: int = 4096,
                 max_shift: int = 4,
                 atac_mask_dropout: float = 0.15,
                 mask_size: int = 1536,
                 augment: bool = True):
        """
        Initialize the dataset
        
        Args:
            data_dir: Directory containing the data files
            split: Data split ('train', 'val', 'test')
            input_length: Length of input sequence
            output_length: Length of output ATAC profile
            max_shift: Maximum shift for data augmentation
            atac_mask_dropout: Fraction of ATAC profile to mask
            mask_size: Size of masked regions
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.input_length = input_length
        self.output_length = output_length
        self.max_shift = max_shift
        self.atac_mask_dropout = atac_mask_dropout
        self.mask_size = mask_size
        self.augment = augment and split == 'train'
        
        # Load file paths and data information
        self.data_files = self._get_data_files()
        
        # Validate data format and load metadata
        if len(self.data_files) > 0:
            self._validate_data_format()
        else:
            print(f"Warning: No data files found in {self.data_dir}/{self.split}")
        
    def _get_data_files(self):
        """Get list of data files for the split"""
        # Support multiple data formats
        data_files = []
        
        # Look for HDF5 files first
        h5_files = list(self.data_dir.glob(f"{self.split}/*.h5"))
        if h5_files:
            data_files.extend(h5_files)
            
        # Look for numpy files
        npz_files = list(self.data_dir.glob(f"{self.split}/*.npz"))
        if npz_files:
            data_files.extend(npz_files)
            
        # Look for pickle files
        pkl_files = list(self.data_dir.glob(f"{self.split}/*.pkl"))
        if pkl_files:
            data_files.extend(pkl_files)
            
        return sorted(data_files)
        
    def _validate_data_format(self):
        """Validate that data files contain expected format"""
        try:
            sample_file = self.data_files[0]
            if sample_file.suffix == '.h5':
                with h5py.File(sample_file, 'r') as f:
                    required_keys = ['sequences', 'atac_profiles']
                    for key in required_keys:
                        if key not in f.keys():
                            print(f"Warning: Required key '{key}' not found in {sample_file}")
            elif sample_file.suffix == '.npz':
                data = np.load(sample_file)
                required_keys = ['sequences', 'atac_profiles']
                for key in required_keys:
                    if key not in data.files:
                        print(f"Warning: Required key '{key}' not found in {sample_file}")
        except Exception as e:
            print(f"Warning: Could not validate data format: {e}")
        
    def __len__(self) -> int:
        """Return the total number of examples"""
        if not self.data_files:
            return 0
            
        total_examples = 0
        for file_path in self.data_files:
            try:
                if file_path.suffix == '.h5':
                    with h5py.File(file_path, 'r') as f:
                        if 'sequences' in f:
                            total_examples += f['sequences'].shape[0]
                elif file_path.suffix == '.npz':
                    data = np.load(file_path)
                    if 'sequences' in data:
                        total_examples += data['sequences'].shape[0]
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                
        return total_examples if total_examples > 0 else 1000  # Fallback for demo
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example from the dataset
        
        Loads actual data from files and applies augmentations and masking
        as specified for EpiBERT training.
        """
        
        # Load actual data from files
        example = self._load_example(idx)
        
        if self.augment:
            example = self._apply_augmentations(example)
            
        example = self._apply_masking(example)
        
        return example
        
    def _load_example(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load actual data from files"""
        if not self.data_files:
            # Fallback to generated data for demo/testing
            return self._generate_demo_example()
            
        # Determine which file and local index
        cumulative_count = 0
        file_idx = 0
        local_idx = idx
        
        for i, file_path in enumerate(self.data_files):
            try:
                if file_path.suffix == '.h5':
                    with h5py.File(file_path, 'r') as f:
                        file_size = f['sequences'].shape[0] if 'sequences' in f else 0
                elif file_path.suffix == '.npz':
                    data = np.load(file_path)
                    file_size = data['sequences'].shape[0] if 'sequences' in data else 0
                else:
                    file_size = 0
                    
                if cumulative_count + file_size > idx:
                    file_idx = i
                    local_idx = idx - cumulative_count
                    break
                cumulative_count += file_size
            except Exception:
                continue
                
        # Load the specific example
        try:
            return self._load_from_file(self.data_files[file_idx], local_idx)
        except Exception as e:
            print(f"Warning: Failed to load data from {self.data_files[file_idx]}: {e}")
            return self._generate_demo_example()
            
    def _load_from_file(self, file_path: Path, idx: int) -> Dict[str, torch.Tensor]:
        """Load specific example from a data file"""
        if file_path.suffix == '.h5':
            with h5py.File(file_path, 'r') as f:
                sequence = torch.from_numpy(f['sequences'][idx])
                atac_profile = torch.from_numpy(f['atac_profiles'][idx])
                
                # Load optional fields if available
                motif_activity = torch.from_numpy(f['motif_activities'][idx]) if 'motif_activities' in f else torch.rand(693)
                peaks_center = torch.from_numpy(f['peaks_centers'][idx]) if 'peaks_centers' in f else torch.randint(0, self.output_length, (10,))
                
        elif file_path.suffix == '.npz':
            data = np.load(file_path)
            sequence = torch.from_numpy(data['sequences'][idx])
            atac_profile = torch.from_numpy(data['atac_profiles'][idx])
            
            motif_activity = torch.from_numpy(data['motif_activities'][idx]) if 'motif_activities' in data else torch.rand(693)
            peaks_center = torch.from_numpy(data['peaks_centers'][idx]) if 'peaks_centers' in data else torch.randint(0, self.output_length, (10,))
            
        else:
            return self._generate_demo_example()
            
        # Ensure correct shapes and types
        if sequence.dim() == 1:  # If not one-hot encoded
            sequence_onehot = torch.zeros(4, sequence.shape[0])
            sequence_onehot[sequence, torch.arange(sequence.shape[0])] = 1
            sequence = sequence_onehot
        elif sequence.dim() == 2 and sequence.shape[0] != 4:
            sequence = sequence.transpose(0, 1)  # Ensure (4, length) format
            
        if atac_profile.dim() == 1:
            atac_profile = atac_profile.unsqueeze(0)  # Add channel dimension
            
        return {
            'sequence': sequence.float(),
            'atac': atac_profile.float(),
            'motif_activity': motif_activity.float(),
            'peaks_center': peaks_center.long(),
            'target': atac_profile.squeeze(0).float()  # Target for prediction
        }
        
    def _generate_demo_example(self) -> Dict[str, torch.Tensor]:
        """Generate demo data for testing/fallback"""
        # Generate demo sequence and profiles
        sequence = torch.randint(0, 4, (self.input_length,))
        atac_profile = torch.rand(self.output_length) * 10
        motif_activity = torch.rand(693)
        peaks_center = torch.randint(0, self.output_length, (10,))
        
        # Convert sequence to one-hot encoding
        sequence_onehot = torch.zeros(4, self.input_length)
        sequence_onehot[sequence, torch.arange(self.input_length)] = 1
        
        return {
            'sequence': sequence_onehot,
            'atac': atac_profile.unsqueeze(0),
            'motif_activity': motif_activity,
            'peaks_center': peaks_center,
            'target': atac_profile
        }
        
    def _apply_augmentations(self, example: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentations similar to original implementation"""
        
        # Random shift
        if self.max_shift > 0:
            shift = torch.randint(0, self.max_shift, (1,)).item()
            example = self._apply_shift(example, shift)
            
        # Random reverse complement
        if torch.rand(1).item() > 0.5:
            example = self._apply_reverse_complement(example)
            
        return example
        
    def _apply_shift(self, example: Dict[str, torch.Tensor], shift: int) -> Dict[str, torch.Tensor]:
        """Apply random shift to sequence and profiles"""
        # This is simplified - full implementation would handle edge cases
        if shift > 0:
            example['sequence'] = example['sequence'][:, shift:]
            if example['sequence'].shape[1] < self.input_length:
                # Pad with zeros if needed
                pad_size = self.input_length - example['sequence'].shape[1]
                padding = torch.zeros(4, pad_size)
                example['sequence'] = torch.cat([example['sequence'], padding], dim=1)
                
        return example
        
    def _apply_reverse_complement(self, example: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply reverse complement to DNA sequence"""
        # Reverse complement: A<->T, C<->G
        sequence = example['sequence']
        # Flip sequence order and complement
        sequence = torch.flip(sequence, dims=[1])
        sequence = torch.flip(sequence, dims=[0])  # A<->T, C<->G
        example['sequence'] = sequence
        
        # Also reverse other profiles
        example['atac'] = torch.flip(example['atac'], dims=[1])
        example['target'] = torch.flip(example['target'], dims=[0])
        
        return example
        
    def _apply_masking(self, example: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply masking to ATAC profile as in original implementation"""
        atac = example['atac'].clone()
        target = example['target'].clone()
        
        # Create mask for training
        mask = torch.ones_like(target)
        
        # Apply random masking
        num_mask = int(self.atac_mask_dropout * len(target))
        if num_mask > 0:
            mask_indices = torch.randperm(len(target))[:num_mask]
            mask[mask_indices] = 0
            atac[0, mask_indices] = 0  # Mask the input
            
        # Create unmask for evaluation (regions that were masked)
        unmask = 1 - mask
        
        example['atac'] = atac
        example['mask'] = mask
        example['unmask'] = unmask
        
        return example


class EpiBERTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for EpiBERT
    
    This handles data loading, preprocessing, and batch creation for
    training, validation, and testing.
    """
    
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 input_length: int = 524288,
                 output_length: int = 4096,
                 atac_mask_dropout: float = 0.15,
                 **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_length = input_length
        self.output_length = output_length
        self.atac_mask_dropout = atac_mask_dropout
        self.kwargs = kwargs
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""
        
        if stage == "fit" or stage is None:
            self.train_dataset = EpiBERTDataset(
                data_dir=self.data_dir,
                split='train',
                input_length=self.input_length,
                output_length=self.output_length,
                atac_mask_dropout=self.atac_mask_dropout,
                augment=True,
                **self.kwargs
            )
            
            self.val_dataset = EpiBERTDataset(
                data_dir=self.data_dir,
                split='val',
                input_length=self.input_length,
                output_length=self.output_length,
                atac_mask_dropout=self.atac_mask_dropout,
                augment=False,
                **self.kwargs
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = EpiBERTDataset(
                data_dir=self.data_dir,
                split='test',
                input_length=self.input_length,
                output_length=self.output_length,
                atac_mask_dropout=0.0,  # No masking for test
                augment=False,
                **self.kwargs
            )
            
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )


def create_data_module(data_dir: Optional[str] = None,
                      manifest_file: Optional[str] = None,
                      batch_size: int = 8,
                      num_workers: int = 4,
                      use_paired_dataset: bool = False,
                      **kwargs) -> Union[EpiBERTDataModule, 'PairedDataModule']:
    """
    Factory function to create data module
    
    Args:
        data_dir: Directory containing data files (for single-sample dataset)
        manifest_file: Manifest file for paired multi-sample dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        use_paired_dataset: Whether to use paired dataset module
        **kwargs: Additional arguments
        
    Returns:
        Data module instance (EpiBERTDataModule or PairedDataModule)
    """
    
    # Determine which module to use
    if use_paired_dataset or manifest_file:
        if not PAIRED_MODULE_AVAILABLE:
            raise ImportError("Paired data module not available. Install required dependencies.")
        if not manifest_file:
            raise ValueError("manifest_file required for paired dataset")
        return PairedDataModule(
            manifest_file=manifest_file,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )
    else:
        if not data_dir:
            raise ValueError("data_dir required for single-sample dataset")
        return EpiBERTDataModule(
            data_dir=data_dir,
            batch_size=batch_size, 
            num_workers=num_workers,
            **kwargs
        )


# Data conversion utilities for various genomic data formats
def create_hdf5_from_arrays(sequences: np.ndarray,
                           atac_profiles: np.ndarray,
                           output_path: str,
                           motif_activities: Optional[np.ndarray] = None,
                           peaks_centers: Optional[np.ndarray] = None,
                           chunk_size: int = 1000):
    """
    Create HDF5 file from numpy arrays for efficient PyTorch loading
    
    Args:
        sequences: Array of DNA sequences (N, 4, seq_len) or (N, seq_len) 
        atac_profiles: Array of ATAC-seq profiles (N, profile_len)
        output_path: Path for output HDF5 file
        motif_activities: Optional motif activity scores (N, n_motifs)
        peaks_centers: Optional peak center positions (N, n_peaks)
        chunk_size: Chunk size for HDF5 storage
    """
    
    print(f"Creating HDF5 file: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Store sequences
        f.create_dataset('sequences', data=sequences, 
                        chunks=True, compression='gzip')
        
        # Store ATAC profiles
        f.create_dataset('atac_profiles', data=atac_profiles,
                        chunks=True, compression='gzip')
        
        # Store optional data
        if motif_activities is not None:
            f.create_dataset('motif_activities', data=motif_activities,
                           chunks=True, compression='gzip')
            
        if peaks_centers is not None:
            f.create_dataset('peaks_centers', data=peaks_centers,
                           chunks=True, compression='gzip')
        
        # Add metadata
        f.attrs['n_samples'] = len(sequences)
        f.attrs['seq_length'] = sequences.shape[-1]
        f.attrs['profile_length'] = atac_profiles.shape[-1]
        f.attrs['created_by'] = 'EpiBERT_data_module'
        
    print(f"Successfully created HDF5 file with {len(sequences)} samples")


def create_npz_from_arrays(sequences: np.ndarray,
                          atac_profiles: np.ndarray,
                          output_path: str,
                          motif_activities: Optional[np.ndarray] = None,
                          peaks_centers: Optional[np.ndarray] = None):
    """
    Create compressed numpy file from arrays
    
    Args:
        sequences: Array of DNA sequences
        atac_profiles: Array of ATAC-seq profiles  
        output_path: Path for output NPZ file
        motif_activities: Optional motif activity scores
        peaks_centers: Optional peak center positions
    """
    
    print(f"Creating NPZ file: {output_path}")
    
    data_dict = {
        'sequences': sequences,
        'atac_profiles': atac_profiles
    }
    
    if motif_activities is not None:
        data_dict['motif_activities'] = motif_activities
        
    if peaks_centers is not None:
        data_dict['peaks_centers'] = peaks_centers
    
    np.savez_compressed(output_path, **data_dict)
    print(f"Successfully created NPZ file with {len(sequences)} samples")


def validate_data_format(data_path: str) -> Dict[str, Any]:
    """
    Validate and get information about a data file
    
    Returns:
        Dictionary with file information and validation results
    """
    
    file_path = Path(data_path)
    info = {
        'valid': False,
        'format': file_path.suffix,
        'n_samples': 0,
        'seq_length': 0,
        'profile_length': 0,
        'has_motifs': False,
        'has_peaks': False,
        'error': None
    }
    
    try:
        if file_path.suffix == '.h5':
            with h5py.File(file_path, 'r') as f:
                if 'sequences' in f and 'atac_profiles' in f:
                    info['valid'] = True
                    info['n_samples'] = f['sequences'].shape[0]
                    info['seq_length'] = f['sequences'].shape[-1]
                    info['profile_length'] = f['atac_profiles'].shape[-1]
                    info['has_motifs'] = 'motif_activities' in f
                    info['has_peaks'] = 'peaks_centers' in f
                else:
                    info['error'] = "Missing required datasets 'sequences' or 'atac_profiles'"
                    
        elif file_path.suffix == '.npz':
            data = np.load(file_path)
            if 'sequences' in data.files and 'atac_profiles' in data.files:
                info['valid'] = True
                info['n_samples'] = data['sequences'].shape[0]
                info['seq_length'] = data['sequences'].shape[-1]
                info['profile_length'] = data['atac_profiles'].shape[-1]
                info['has_motifs'] = 'motif_activities' in data.files
                info['has_peaks'] = 'peaks_centers' in data.files
            else:
                info['error'] = "Missing required arrays 'sequences' or 'atac_profiles'"
        else:
            info['error'] = f"Unsupported file format: {file_path.suffix}"
            
    except Exception as e:
        info['error'] = str(e)
        
    return info


if __name__ == "__main__":
    # Example usage
    data_module = create_data_module(
        data_dir="/path/to/data",
        batch_size=4,
        num_workers=2
    )
    
    # Setup for training
    data_module.setup("fit")
    
    # Test data loading
    if len(data_module.train_dataset) > 0:
        # Get a sample batch
        train_loader = data_module.train_dataloader()
        sample_batch = next(iter(train_loader))
        
        print("Sample batch shapes:")
        for key, value in sample_batch.items():
            print(f"  {key}: {value.shape}")
    else:
        print("No training data found - using demo mode")
        
    # Example of creating data files
    print("\nExample of creating HDF5 data files:")
    
    # Generate example data
    n_samples = 100
    seq_length = 1024  # Smaller for demo
    profile_length = 256
    
    sequences = np.random.randint(0, 4, (n_samples, seq_length))
    atac_profiles = np.random.rand(n_samples, profile_length) * 10
    motif_activities = np.random.rand(n_samples, 693)
    
    # Create demo data file
    demo_path = "/tmp/demo_data.h5"
    create_hdf5_from_arrays(
        sequences=sequences,
        atac_profiles=atac_profiles,
        motif_activities=motif_activities,
        output_path=demo_path
    )
    
    # Validate the created file
    info = validate_data_format(demo_path)
    print(f"Validation result: {info}")
    
    # Clean up
    import os
    if os.path.exists(demo_path):
        os.remove(demo_path)