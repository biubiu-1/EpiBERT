#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced EpiBERT Data Handling

Tests all new functionality for batched data input with multiple paired
ATAC-seq and RAMPAGE samples.
"""

import os
import sys
import tempfile
import shutil
import yaml
import json
import numpy as np
import h5py
from pathlib import Path
import logging
from typing import Dict, Any, List

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data_files(temp_dir: Path, num_samples: int = 5) -> Dict[str, Any]:
    """Create test data files for testing"""
    
    logger.info(f"Creating test data files in {temp_dir}")
    
    # Create sample data
    samples = []
    
    for i in range(num_samples):
        sample_id = f"test_sample_{i+1}"
        condition = f"condition_{(i % 2) + 1}"
        cell_type = f"cell_type_{(i % 3) + 1}"
        
        # Create ATAC data file
        atac_file = temp_dir / f"{sample_id}_atac.h5"
        rampage_file = temp_dir / f"{sample_id}_rampage.h5"
        
        # Generate synthetic data
        seq_length = 1024  # Smaller for testing
        profile_length = 256
        n_regions = 100
        
        # Create ATAC file
        with h5py.File(atac_file, 'w') as f:
            # Sequences (one-hot encoded)
            sequences = np.random.rand(n_regions, 4, seq_length).astype(np.float32)
            sequences = (sequences == sequences.max(axis=1, keepdims=True)).astype(np.float32)
            
            # ATAC profiles
            atac_profiles = np.random.exponential(2.0, (n_regions, profile_length)).astype(np.float32)
            
            # Motif activities
            motif_activities = np.random.rand(n_regions, 693).astype(np.float32)
            
            # Peak centers
            peaks_centers = np.random.randint(0, profile_length, (n_regions, 10)).astype(np.int32)
            
            f.create_dataset('sequences', data=sequences)
            f.create_dataset('atac_profiles', data=atac_profiles)
            f.create_dataset('motif_activities', data=motif_activities)
            f.create_dataset('peaks_centers', data=peaks_centers)
            
        # Create RAMPAGE file (similar structure)
        with h5py.File(rampage_file, 'w') as f:
            rampage_profiles = np.random.exponential(1.5, (n_regions, profile_length)).astype(np.float32)
            f.create_dataset('sequences', data=sequences)  # Same sequences
            f.create_dataset('rampage_profiles', data=rampage_profiles)
            f.create_dataset('motif_activities', data=motif_activities)
            f.create_dataset('peaks_centers', data=peaks_centers)
            
        # Add to samples list
        samples.append({
            'sample_id': sample_id,
            'condition': condition,
            'cell_type': cell_type,
            'atac_file': str(atac_file),
            'rampage_file': str(rampage_file),
            'batch': 'train' if i < 3 else ('val' if i == 3 else 'test'),
            'replicate': (i % 2) + 1
        })
        
    # Create manifest file
    manifest_file = temp_dir / "test_manifest.yaml"
    manifest_data = {'samples': samples}
    
    with open(manifest_file, 'w') as f:
        yaml.dump(manifest_data, f, default_flow_style=False)
        
    return {
        'manifest_file': str(manifest_file),
        'samples': samples,
        'temp_dir': str(temp_dir)
    }


def test_paired_data_module():
    """Test the enhanced paired data module"""
    
    logger.info("Testing PairedDataModule...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        test_data = create_test_data_files(temp_path)
        
        try:
            # Import and test the paired data module
            from lightning_transfer.paired_data_module import PairedDataModule, PairedSampleDataset
            
            # Test dataset creation
            dataset = PairedSampleDataset(
                manifest_file=test_data['manifest_file'],
                split='train',
                input_length=1024,
                output_length=256,
                balance_conditions=True
            )
            
            logger.info(f"Created dataset with {len(dataset)} samples")
            
            # Test sample loading
            sample = dataset[0]
            logger.info(f"Sample keys: {list(sample.keys())}")
            
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    logger.info(f"  {key}: {value.shape}")
                else:
                    logger.info(f"  {key}: {type(value)} - {value}")
                    
            # Test data module
            data_module = PairedDataModule(
                manifest_file=test_data['manifest_file'],
                batch_size=2,
                num_workers=0,  # Avoid multiprocessing in test
                input_length=1024,
                output_length=256
            )
            
            data_module.setup("fit")
            
            # Test dataloaders
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            
            # Test batch loading
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))
            
            logger.info(f"Train batch size: {train_batch['sequence'].shape[0]}")
            logger.info(f"Val batch size: {val_batch['sequence'].shape[0]}")
            
            # Test sample info
            info_df = dataset.get_sample_info()
            logger.info(f"Sample info shape: {info_df.shape}")
            logger.info(f"Conditions: {info_df['condition'].unique()}")
            
            logger.info("✓ PairedDataModule test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ PairedDataModule test failed: {e}")
            return False


def test_data_converter():
    """Test the data converter utilities"""
    
    logger.info("Testing DataConverter...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            from scripts.data_converter import DataConverter
            
            # Create dummy reference files
            genome_fasta = temp_path / "test_genome.fa"
            genome_sizes = temp_path / "test_genome.sizes"
            
            with open(genome_fasta, 'w') as f:
                f.write(">chr1\nACGTACGT\n>chr2\nTGCATGCA\n")
                
            with open(genome_sizes, 'w') as f:
                f.write("chr1\t1000000\nchr2\t1000000\n")
                
            # Create test bedGraph files
            atac_bed = temp_path / "test_atac.bedgraph"
            rampage_bed = temp_path / "test_rampage.bedgraph"
            
            with open(atac_bed, 'w') as f:
                f.write("chr1\t0\t1000\t10.5\n")
                f.write("chr1\t1000\t2000\t15.2\n")
                
            with open(rampage_bed, 'w') as f:
                f.write("chr1\t0\t1000\t5.5\n")
                f.write("chr1\t1000\t2000\t8.2\n")
                
            # Initialize converter
            converter = DataConverter(
                genome_fasta=str(genome_fasta),
                genome_sizes=str(genome_sizes),
                sequence_length=1024,
                output_length=256
            )
            
            # Test conversion
            output_file = temp_path / "test_converted.h5"
            
            converted_file = converter.convert_sample_data(
                sample_id="test_sample",
                atac_bed=str(atac_bed),
                rampage_bed=str(rampage_bed),
                output_file=str(output_file)
            )
            
            # Validate output
            if Path(converted_file).exists():
                with h5py.File(converted_file, 'r') as f:
                    logger.info(f"Output file datasets: {list(f.keys())}")
                    logger.info(f"Sequences shape: {f['sequences'].shape}")
                    logger.info(f"ATAC profiles shape: {f['atac_profiles'].shape}")
                    
                logger.info("✓ DataConverter test passed")
                return True
            else:
                logger.error("✗ DataConverter test failed: output file not created")
                return False
                
        except Exception as e:
            logger.error(f"✗ DataConverter test failed: {e}")
            return False


def test_batch_processor():
    """Test the batch data processor"""
    
    logger.info("Testing BatchDataProcessor...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            from scripts.batch_data_processor import BatchDataProcessor
            
            # Create test config
            config_file = temp_path / "test_config.yaml"
            config_data = {
                'genome_fasta': '/path/to/genome.fa',
                'genome_sizes': '/path/to/genome.sizes',
                'threads': 1
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
                
            # Create test samples
            samples = [
                {
                    'sample_id': 'test1',
                    'condition': 'cond1',
                    'cell_type': 'type1',
                    'atac_bam': '/path/to/test1.bam'
                },
                {
                    'sample_id': 'test2',
                    'condition': 'cond2',
                    'cell_type': 'type1',
                    'atac_bam': '/path/to/test2.bam'
                }
            ]
            
            # Initialize processor
            processor = BatchDataProcessor(
                base_config=str(config_file),
                output_dir=str(temp_path / "output"),
                num_workers=1
            )
            
            # Test job creation
            jobs = processor._create_processing_jobs(samples, ['atac'])
            
            logger.info(f"Created {len(jobs)} jobs")
            
            for job in jobs:
                logger.info(f"Job: {job.sample_id} -> {job.output_dir}")
                
            logger.info("✓ BatchDataProcessor test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ BatchDataProcessor test failed: {e}")
            return False


def test_integration():
    """Test integration between components"""
    
    logger.info("Testing integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Create test data
            test_data = create_test_data_files(temp_path, num_samples=3)
            
            # Test the enhanced data module factory
            from lightning_transfer.data_module import create_data_module
            
            # Test paired data module creation
            data_module = create_data_module(
                manifest_file=test_data['manifest_file'],
                batch_size=2,
                num_workers=0,
                use_paired_dataset=True,
                input_length=1024,
                output_length=256
            )
            
            logger.info(f"Created data module: {type(data_module).__name__}")
            
            # Test setup and loading
            data_module.setup("fit")
            
            train_loader = data_module.train_dataloader()
            batch = next(iter(train_loader))
            
            logger.info(f"Integration test batch shape: {batch['sequence'].shape}")
            logger.info(f"Batch conditions: {batch['condition']}")
            
            logger.info("✓ Integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Integration test failed: {e}")
            return False


def test_validation_utilities():
    """Test validation and utility functions"""
    
    logger.info("Testing validation utilities...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Create test data
            test_data = create_test_data_files(temp_path, num_samples=2)
            
            from lightning_transfer.paired_data_module import (
                validate_manifest, 
                create_sample_manifest
            )
            
            # Test manifest validation
            validation_result = validate_manifest(test_data['manifest_file'])
            
            logger.info(f"Validation result: {validation_result}")
            
            # Test manifest creation
            new_samples = [
                {
                    'sample_id': 'new_sample',
                    'condition': 'new_condition',
                    'cell_type': 'new_type',
                    'atac_file': '/path/to/new_atac.h5'
                }
            ]
            
            new_manifest = temp_path / "new_manifest.yaml"
            create_sample_manifest(new_samples, str(new_manifest))
            
            if new_manifest.exists():
                logger.info("✓ Validation utilities test passed")
                return True
            else:
                logger.error("✗ Validation utilities test failed: manifest not created")
                return False
                
        except Exception as e:
            logger.error(f"✗ Validation utilities test failed: {e}")
            return False


def run_all_tests():
    """Run all tests and report results"""
    
    logger.info("Running comprehensive test suite for enhanced EpiBERT data handling...")
    
    tests = [
        ("Paired Data Module", test_paired_data_module),
        ("Data Converter", test_data_converter),
        ("Batch Processor", test_batch_processor),
        ("Integration", test_integration),
        ("Validation Utilities", test_validation_utilities)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} test...")
        logger.info('='*60)
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
            
    # Report summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info('='*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:30} {status}")
        
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced EpiBERT data handling is working correctly.")
        return True
    else:
        logger.warning("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)