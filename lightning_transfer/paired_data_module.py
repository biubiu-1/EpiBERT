#!/usr/bin/env python3
"""
Enhanced EpiBERT Data Module for Paired Multi-Sample Datasets

Handles multiple paired ATAC-seq and RAMPAGE-seq samples efficiently,
with support for batch composition, sample balancing, and large-scale training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import yaml
import json
from collections import defaultdict
import random
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SampleMetadata:
    """Metadata for a paired ATAC/RAMPAGE sample"""
    sample_id: str
    condition: str
    cell_type: str
    atac_file: str
    rampage_file: Optional[str] = None
    batch: Optional[str] = None
    replicate: Optional[int] = None
    additional_metadata: Optional[Dict[str, Any]] = None


class PairedSampleDataset(Dataset):
    """
    Dataset for handling multiple paired ATAC-seq and RAMPAGE-seq samples
    
    Features:
    - Efficient loading of multiple paired samples
    - Sample balancing across conditions
    - Configurable batch composition
    - Memory-efficient data loading with caching
    """
    
    def __init__(self,
                 manifest_file: str,
                 split: str = 'train',
                 input_length: int = 524288,
                 output_length: int = 4096,
                 max_shift: int = 4,
                 atac_mask_dropout: float = 0.15,
                 mask_size: int = 1536,
                 augment: bool = True,
                 balance_conditions: bool = True,
                 max_samples_per_condition: Optional[int] = None,
                 cache_size: int = 100):
        """
        Initialize paired sample dataset
        
        Args:
            manifest_file: Path to sample manifest (CSV/YAML) with sample metadata
            split: Data split ('train', 'val', 'test')
            input_length: Length of input sequence
            output_length: Length of output ATAC profile
            max_shift: Maximum shift for data augmentation
            atac_mask_dropout: Fraction of ATAC profile to mask
            mask_size: Size of masked regions
            augment: Whether to apply data augmentation
            balance_conditions: Whether to balance samples across conditions
            max_samples_per_condition: Maximum samples per condition (for balancing)
            cache_size: Number of samples to keep in memory cache
        """
        self.manifest_file = Path(manifest_file)
        self.split = split
        self.input_length = input_length
        self.output_length = output_length
        self.max_shift = max_shift
        self.atac_mask_dropout = atac_mask_dropout
        self.mask_size = mask_size
        self.augment = augment and split == 'train'
        self.balance_conditions = balance_conditions
        self.max_samples_per_condition = max_samples_per_condition
        self.cache_size = cache_size
        
        # Load sample metadata
        self.sample_metadata = self._load_sample_manifest()
        
        # Filter samples for current split
        self.samples = self._filter_samples_for_split()
        
        # Balance samples if requested
        if self.balance_conditions and self.split == 'train':
            self.samples = self._balance_samples()
        
        # Initialize data cache
        self.data_cache = {}
        self.cache_order = []
        
        logger.info(f"Initialized {self.__class__.__name__} with {len(self.samples)} samples for {self.split} split")
        if self.sample_metadata:
            conditions = set(s.condition for s in self.samples)
            logger.info(f"Conditions: {list(conditions)}")
            
    def _load_sample_manifest(self) -> List[SampleMetadata]:
        """Load sample metadata from manifest file"""
        samples = []
        
        if not self.manifest_file.exists():
            logger.warning(f"Manifest file not found: {self.manifest_file}")
            return samples
            
        try:
            if self.manifest_file.suffix.lower() in ['.yaml', '.yml']:
                with open(self.manifest_file, 'r') as f:
                    data = yaml.safe_load(f)
                    
                if 'samples' in data:
                    for sample_data in data['samples']:
                        samples.append(SampleMetadata(**sample_data))
                        
            elif self.manifest_file.suffix.lower() == '.csv':
                df = pd.read_csv(self.manifest_file)
                for _, row in df.iterrows():
                    sample_data = row.to_dict()
                    # Handle additional metadata
                    metadata_cols = [c for c in sample_data.keys() 
                                   if c not in ['sample_id', 'condition', 'cell_type', 
                                              'atac_file', 'rampage_file', 'batch', 'replicate']]
                    additional_metadata = {k: sample_data.pop(k) for k in metadata_cols}
                    sample_data['additional_metadata'] = additional_metadata
                    samples.append(SampleMetadata(**sample_data))
                    
            elif self.manifest_file.suffix.lower() == '.json':
                with open(self.manifest_file, 'r') as f:
                    data = json.load(f)
                for sample_data in data['samples']:
                    samples.append(SampleMetadata(**sample_data))
                    
        except Exception as e:
            logger.error(f"Error loading manifest file {self.manifest_file}: {e}")
            
        return samples
        
    def _filter_samples_for_split(self) -> List[SampleMetadata]:
        """Filter samples for the current data split"""
        if not self.sample_metadata:
            return []
            
        # If samples have explicit split information, use it
        samples_with_split = [s for s in self.sample_metadata if hasattr(s, 'split')]
        if samples_with_split:
            return [s for s in samples_with_split if s.split == self.split]
            
        # Otherwise, use batch information or split randomly
        samples_with_batch = [s for s in self.sample_metadata if s.batch]
        if samples_with_batch:
            # Use batch as split identifier
            batch_mapping = {
                'train': ['train', 'training'],
                'val': ['val', 'validation', 'valid'],
                'test': ['test', 'testing']
            }
            target_batches = batch_mapping.get(self.split, [self.split])
            return [s for s in samples_with_batch 
                   if any(batch in s.batch.lower() for batch in target_batches)]
        
        # Random split if no explicit split info
        random.seed(42)  # For reproducibility
        samples = self.sample_metadata[:]
        random.shuffle(samples)
        
        total = len(samples)
        if self.split == 'train':
            return samples[:int(0.7 * total)]
        elif self.split == 'val':
            return samples[int(0.7 * total):int(0.85 * total)]
        else:  # test
            return samples[int(0.85 * total):]
            
    def _balance_samples(self) -> List[SampleMetadata]:
        """Balance samples across conditions"""
        condition_samples = defaultdict(list)
        for sample in self.samples:
            condition_samples[sample.condition].append(sample)
            
        # Determine target count per condition
        if self.max_samples_per_condition:
            target_count = self.max_samples_per_condition
        else:
            # Use the minimum count across conditions
            target_count = min(len(samples) for samples in condition_samples.values())
            
        balanced_samples = []
        for condition, samples in condition_samples.items():
            if len(samples) >= target_count:
                # Randomly sample if we have more than needed
                balanced_samples.extend(random.sample(samples, target_count))
            else:
                # Use all samples and potentially repeat
                balanced_samples.extend(samples)
                remaining = target_count - len(samples)
                if remaining > 0:
                    # Add random repeats
                    balanced_samples.extend(random.choices(samples, k=remaining))
                    
        random.shuffle(balanced_samples)
        logger.info(f"Balanced dataset: {len(balanced_samples)} samples across {len(condition_samples)} conditions")
        return balanced_samples
        
    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.samples) if self.samples else 1000  # Fallback for demo
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample with its paired data"""
        if not self.samples:
            return self._generate_demo_example()
            
        sample_metadata = self.samples[idx]
        
        # Check cache first
        cache_key = f"{sample_metadata.sample_id}_{self.split}"
        if cache_key in self.data_cache:
            example = self.data_cache[cache_key].copy()
        else:
            example = self._load_sample_data(sample_metadata)
            self._update_cache(cache_key, example)
            
        # Apply augmentations and masking
        if self.augment:
            example = self._apply_augmentations(example)
        example = self._apply_masking(example)
        
        # Add metadata
        example['sample_id'] = sample_metadata.sample_id
        example['condition'] = sample_metadata.condition
        example['cell_type'] = sample_metadata.cell_type
        
        return example
        
    def _load_sample_data(self, sample_metadata: SampleMetadata) -> Dict[str, torch.Tensor]:
        """Load data for a specific sample"""
        try:
            # Load ATAC-seq data
            atac_data = self._load_data_file(sample_metadata.atac_file)
            
            # Load RAMPAGE-seq data if available
            rampage_data = None
            if sample_metadata.rampage_file and Path(sample_metadata.rampage_file).exists():
                rampage_data = self._load_data_file(sample_metadata.rampage_file)
                
            return self._combine_sample_data(atac_data, rampage_data, sample_metadata)
            
        except Exception as e:
            logger.warning(f"Failed to load data for sample {sample_metadata.sample_id}: {e}")
            return self._generate_demo_example()
            
    def _load_data_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load data from HDF5, NPZ, or other format"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        if file_path.suffix == '.h5':
            with h5py.File(file_path, 'r') as f:
                data = {}
                for key in f.keys():
                    if len(f[key].shape) > 0:  # Skip scalar datasets
                        data[key] = f[key][:]
                return data
                
        elif file_path.suffix == '.npz':
            data = np.load(file_path)
            return {key: data[key] for key in data.files}
            
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
    def _combine_sample_data(self, atac_data: Dict[str, np.ndarray], 
                           rampage_data: Optional[Dict[str, np.ndarray]],
                           sample_metadata: SampleMetadata) -> Dict[str, torch.Tensor]:
        """Combine ATAC and RAMPAGE data into a single example"""
        
        # Extract sequence and ATAC profile
        sequence = torch.from_numpy(atac_data.get('sequences', atac_data.get('sequence', np.random.randint(0, 4, self.input_length))))
        atac_profile = torch.from_numpy(atac_data.get('atac_profiles', atac_data.get('atac', np.random.rand(self.output_length))))
        
        # Handle sequence format
        if sequence.dim() == 1:  # Convert to one-hot if needed
            sequence_onehot = torch.zeros(4, len(sequence))
            sequence_onehot[sequence, torch.arange(len(sequence))] = 1
            sequence = sequence_onehot
        elif sequence.shape[0] != 4:
            sequence = sequence.transpose(0, 1)
            
        # Ensure correct shapes
        if sequence.shape[1] != self.input_length:
            # Resize sequence
            if sequence.shape[1] > self.input_length:
                sequence = sequence[:, :self.input_length]
            else:
                padding = torch.zeros(4, self.input_length - sequence.shape[1])
                sequence = torch.cat([sequence, padding], dim=1)
                
        if len(atac_profile) != self.output_length:
            # Resize ATAC profile
            if len(atac_profile) > self.output_length:
                atac_profile = atac_profile[:self.output_length]
            else:
                padding = torch.zeros(self.output_length - len(atac_profile))
                atac_profile = torch.cat([atac_profile, padding])
                
        # Add RAMPAGE data if available
        rampage_profile = torch.zeros(self.output_length)
        if rampage_data is not None:
            rampage_raw = torch.from_numpy(rampage_data.get('rampage_profiles', rampage_data.get('rampage', np.random.rand(self.output_length))))
            if len(rampage_raw) == self.output_length:
                rampage_profile = rampage_raw
            elif len(rampage_raw) > self.output_length:
                rampage_profile = rampage_raw[:self.output_length]
            else:
                rampage_profile[:len(rampage_raw)] = rampage_raw
                
        # Load optional data with fallbacks
        motif_activity = torch.from_numpy(atac_data.get('motif_activities', np.random.rand(693)))
        peaks_center = torch.from_numpy(atac_data.get('peaks_centers', np.random.randint(0, self.output_length, 10)))
        
        return {
            'sequence': sequence.float(),
            'atac': atac_profile.unsqueeze(0).float(),
            'rampage': rampage_profile.unsqueeze(0).float(),
            'motif_activity': motif_activity.float(),
            'peaks_center': peaks_center.long(),
            'target': atac_profile.float()
        }
        
    def _update_cache(self, cache_key: str, example: Dict[str, torch.Tensor]):
        """Update the data cache with LRU eviction"""
        if len(self.data_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = self.cache_order.pop(0)
            del self.data_cache[oldest_key]
            
        self.data_cache[cache_key] = {k: v.clone() for k, v in example.items() if torch.is_tensor(v)}
        self.cache_order.append(cache_key)
        
    def _generate_demo_example(self) -> Dict[str, torch.Tensor]:
        """Generate demo data for testing/fallback"""
        sequence = torch.randint(0, 4, (self.input_length,))
        atac_profile = torch.rand(self.output_length) * 10
        rampage_profile = torch.rand(self.output_length) * 5
        motif_activity = torch.rand(693)
        peaks_center = torch.randint(0, self.output_length, (10,))
        
        # Convert sequence to one-hot
        sequence_onehot = torch.zeros(4, self.input_length)
        sequence_onehot[sequence, torch.arange(self.input_length)] = 1
        
        return {
            'sequence': sequence_onehot.float(),
            'atac': atac_profile.unsqueeze(0),
            'rampage': rampage_profile.unsqueeze(0),
            'motif_activity': motif_activity,
            'peaks_center': peaks_center,
            'target': atac_profile,
            'sample_id': 'demo_sample',
            'condition': 'demo_condition',
            'cell_type': 'demo_cell_type'
        }
        
    def _apply_augmentations(self, example: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentations (similar to original implementation)"""
        if self.max_shift > 0:
            shift = torch.randint(0, self.max_shift, (1,)).item()
            example = self._apply_shift(example, shift)
            
        if torch.rand(1).item() > 0.5:
            example = self._apply_reverse_complement(example)
            
        return example
        
    def _apply_shift(self, example: Dict[str, torch.Tensor], shift: int) -> Dict[str, torch.Tensor]:
        """Apply random shift to sequence and profiles"""
        if shift > 0 and example['sequence'].shape[1] > shift:
            example['sequence'] = example['sequence'][:, shift:]
            if example['sequence'].shape[1] < self.input_length:
                pad_size = self.input_length - example['sequence'].shape[1]
                padding = torch.zeros(4, pad_size)
                example['sequence'] = torch.cat([example['sequence'], padding], dim=1)
        return example
        
    def _apply_reverse_complement(self, example: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply reverse complement to DNA sequence"""
        sequence = example['sequence']
        sequence = torch.flip(sequence, dims=[1])  # Reverse
        sequence = torch.flip(sequence, dims=[0])  # Complement A<->T, C<->G
        example['sequence'] = sequence
        
        # Also reverse profiles
        for key in ['atac', 'rampage', 'target']:
            if key in example:
                if example[key].dim() == 1:
                    example[key] = torch.flip(example[key], dims=[0])
                else:
                    example[key] = torch.flip(example[key], dims=[-1])
                    
        return example
        
    def _apply_masking(self, example: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply masking to ATAC profile"""
        atac = example['atac'].clone()
        target = example['target'].clone()
        
        mask = torch.ones_like(target)
        
        # Apply random masking for training
        if self.split == 'train' and self.atac_mask_dropout > 0:
            num_mask = int(self.atac_mask_dropout * len(target))
            if num_mask > 0:
                mask_indices = torch.randperm(len(target))[:num_mask]
                mask[mask_indices] = 0
                atac[0, mask_indices] = 0  # Mask the input
                
        unmask = 1 - mask
        
        example['atac'] = atac
        example['mask'] = mask
        example['unmask'] = unmask
        
        return example
        
    def get_sample_info(self) -> pd.DataFrame:
        """Get summary information about samples in the dataset"""
        if not self.samples:
            return pd.DataFrame()
            
        data = []
        for sample in self.samples:
            data.append({
                'sample_id': sample.sample_id,
                'condition': sample.condition,
                'cell_type': sample.cell_type,
                'atac_file': sample.atac_file,
                'rampage_file': sample.rampage_file,
                'batch': sample.batch,
                'replicate': sample.replicate
            })
            
        return pd.DataFrame(data)


class ConditionBalancedSampler(Sampler):
    """Custom sampler that ensures balanced representation of conditions in each batch"""
    
    def __init__(self, dataset: PairedSampleDataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group samples by condition
        self.condition_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.condition_indices[sample.condition].append(idx)
            
        self.conditions = list(self.condition_indices.keys())
        self.num_conditions = len(self.conditions)
        
    def __iter__(self):
        """Generate batches with balanced condition representation"""
        # Create iterators for each condition
        condition_iterators = {}
        for condition in self.conditions:
            indices = self.condition_indices[condition][:]
            random.shuffle(indices)
            condition_iterators[condition] = iter(indices)
            
        batches = []
        current_batch = []
        
        while True:
            # Try to add one sample from each condition
            added_any = False
            for condition in self.conditions:
                try:
                    idx = next(condition_iterators[condition])
                    current_batch.append(idx)
                    added_any = True
                    
                    if len(current_batch) == self.batch_size:
                        batches.append(current_batch)
                        current_batch = []
                except StopIteration:
                    continue
                    
            if not added_any:
                break
                
        # Handle remaining samples
        if current_batch and not self.drop_last:
            batches.append(current_batch)
            
        # Shuffle batch order
        random.shuffle(batches)
        
        for batch in batches:
            for idx in batch:
                yield idx
                
    def __len__(self):
        """Return total number of samples"""
        total_samples = sum(len(indices) for indices in self.condition_indices.values())
        if self.drop_last:
            return (total_samples // self.batch_size) * self.batch_size
        return total_samples


class PairedDataModule(pl.LightningDataModule):
    """
    Enhanced Lightning DataModule for paired multi-sample datasets
    
    Features:
    - Multiple paired sample management
    - Configurable batch composition
    - Condition balancing
    - Efficient data loading
    """
    
    def __init__(self,
                 manifest_file: str,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 input_length: int = 524288,
                 output_length: int = 4096,
                 atac_mask_dropout: float = 0.15,
                 balance_conditions: bool = True,
                 use_condition_sampler: bool = True,
                 max_samples_per_condition: Optional[int] = None,
                 cache_size: int = 100,
                 **kwargs):
        super().__init__()
        self.manifest_file = manifest_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_length = input_length
        self.output_length = output_length
        self.atac_mask_dropout = atac_mask_dropout
        self.balance_conditions = balance_conditions
        self.use_condition_sampler = use_condition_sampler
        self.max_samples_per_condition = max_samples_per_condition
        self.cache_size = cache_size
        self.kwargs = kwargs
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""
        
        if stage == "fit" or stage is None:
            self.train_dataset = PairedSampleDataset(
                manifest_file=self.manifest_file,
                split='train',
                input_length=self.input_length,
                output_length=self.output_length,
                atac_mask_dropout=self.atac_mask_dropout,
                augment=True,
                balance_conditions=self.balance_conditions,
                max_samples_per_condition=self.max_samples_per_condition,
                cache_size=self.cache_size,
                **self.kwargs
            )
            
            self.val_dataset = PairedSampleDataset(
                manifest_file=self.manifest_file,
                split='val',
                input_length=self.input_length,
                output_length=self.output_length,
                atac_mask_dropout=self.atac_mask_dropout,
                augment=False,
                balance_conditions=False,  # No balancing for validation
                cache_size=self.cache_size // 2,
                **self.kwargs
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = PairedSampleDataset(
                manifest_file=self.manifest_file,
                split='test',
                input_length=self.input_length,
                output_length=self.output_length,
                atac_mask_dropout=0.0,  # No masking for test
                augment=False,
                balance_conditions=False,
                cache_size=self.cache_size // 2,
                **self.kwargs
            )
            
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader with optional condition balancing"""
        sampler = None
        shuffle = True
        
        if (self.use_condition_sampler and 
            hasattr(self.train_dataset, 'samples') and 
            len(set(s.condition for s in self.train_dataset.samples)) > 1):
            sampler = ConditionBalancedSampler(self.train_dataset, self.batch_size)
            shuffle = False
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True
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


def create_sample_manifest(sample_list: List[Dict[str, Any]], 
                          output_path: str,
                          format: str = 'yaml') -> None:
    """
    Create a sample manifest file from a list of sample dictionaries
    
    Args:
        sample_list: List of dictionaries with sample information
        output_path: Path for output manifest file
        format: Output format ('yaml', 'csv', 'json')
    """
    
    output_path = Path(output_path)
    
    if format.lower() == 'yaml':
        manifest_data = {'samples': sample_list}
        with open(output_path, 'w') as f:
            yaml.dump(manifest_data, f, default_flow_style=False)
            
    elif format.lower() == 'csv':
        df = pd.DataFrame(sample_list)
        df.to_csv(output_path, index=False)
        
    elif format.lower() == 'json':
        manifest_data = {'samples': sample_list}
        with open(output_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
            
    else:
        raise ValueError(f"Unsupported format: {format}")
        
    logger.info(f"Created manifest file: {output_path}")


def validate_manifest(manifest_file: str) -> Dict[str, Any]:
    """
    Validate a sample manifest file
    
    Returns:
        Dictionary with validation results
    """
    
    manifest_path = Path(manifest_file)
    result = {
        'valid': False,
        'num_samples': 0,
        'conditions': [],
        'cell_types': [],
        'missing_files': [],
        'errors': []
    }
    
    try:
        # Create temporary dataset to load manifest
        temp_dataset = PairedSampleDataset(manifest_file, split='train')
        samples = temp_dataset.sample_metadata
        
        result['num_samples'] = len(samples)
        result['conditions'] = list(set(s.condition for s in samples))
        result['cell_types'] = list(set(s.cell_type for s in samples))
        
        # Check for missing files
        for sample in samples:
            if not Path(sample.atac_file).exists():
                result['missing_files'].append(f"ATAC file missing: {sample.atac_file}")
            if sample.rampage_file and not Path(sample.rampage_file).exists():
                result['missing_files'].append(f"RAMPAGE file missing: {sample.rampage_file}")
                
        result['valid'] = len(result['errors']) == 0
        
    except Exception as e:
        result['errors'].append(str(e))
        
    return result


if __name__ == "__main__":
    # Example usage and testing
    
    # Create example manifest
    sample_data = [
        {
            'sample_id': 'sample1',
            'condition': 'condition1',
            'cell_type': 'cell_type1',
            'atac_file': '/path/to/sample1_atac.h5',
            'rampage_file': '/path/to/sample1_rampage.h5',
            'batch': 'train',
            'replicate': 1
        },
        {
            'sample_id': 'sample2',
            'condition': 'condition2',
            'cell_type': 'cell_type1',
            'atac_file': '/path/to/sample2_atac.h5',
            'rampage_file': '/path/to/sample2_rampage.h5',
            'batch': 'train',
            'replicate': 1
        }
    ]
    
    manifest_path = "/tmp/example_manifest.yaml"
    create_sample_manifest(sample_data, manifest_path, format='yaml')
    
    # Test validation
    validation_result = validate_manifest(manifest_path)
    print("Validation result:", validation_result)
    
    # Test data module
    try:
        data_module = PairedDataModule(
            manifest_file=manifest_path,
            batch_size=2,
            num_workers=0  # Avoid multiprocessing issues in example
        )
        
        data_module.setup("fit")
        
        print(f"Train dataset size: {len(data_module.train_dataset)}")
        print(f"Val dataset size: {len(data_module.val_dataset)}")
        
        # Test sample loading
        train_loader = data_module.train_dataloader()
        sample_batch = next(iter(train_loader))
        
        print("\nSample batch shapes:")
        for key, value in sample_batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)} - {value[:2] if isinstance(value, list) else value}")
                
    except Exception as e:
        print(f"Error in example: {e}")
        
    # Clean up
    import os
    if os.path.exists(manifest_path):
        os.remove(manifest_path)