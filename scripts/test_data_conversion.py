#!/usr/bin/env python3
"""
Test Suite for Data Conversion Utilities

Tests TFRecord to HDF5 conversion and PyTorch dataset generation to ensure
all utilities work correctly with synthetic and real data.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import h5py
import tensorflow as tf
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our conversion utilities
try:
    from scripts.convert_tfrecord_to_hdf5 import TFRecordToHDF5Converter, validate_converted_hdf5
    from scripts.generate_pytorch_dataset import PyTorchDatasetGenerator
    from lightning_transfer.data_module import EpiBERTDataModule
    CONVERSION_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import conversion modules: {e}")
    CONVERSION_MODULES_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataConversionTester:
    """
    Comprehensive test suite for data conversion utilities
    """
    
    def __init__(self, test_dir: str = None):
        """Initialize tester with temporary directory"""
        if test_dir is None:
            self.test_dir = Path(tempfile.mkdtemp(prefix="epibert_data_test_"))
        else:
            self.test_dir = Path(test_dir)
            self.test_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Using test directory: {self.test_dir}")
        
        # Test parameters
        self.sequence_length = 1024  # Smaller for testing
        self.output_length = 64
        self.motif_features = 693
        self.n_test_examples = 10
        
    def __del__(self):
        """Clean up test directory"""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
            
    def create_synthetic_tfrecord(self, output_path: str, n_examples: int = 10) -> str:
        """Create synthetic TFRecord file for testing"""
        
        logger.info(f"Creating synthetic TFRecord with {n_examples} examples: {output_path}")
        
        def _bytes_feature(value):
            """Returns a bytes_list from a string / byte."""
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _tensor_feature_float(numpy_array):
            serialized_tensor = tf.io.serialize_tensor(tf.convert_to_tensor(numpy_array, dtype=tf.float16))
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_tensor.numpy()]))

        def _tensor_feature_int(numpy_array):
            serialized_tensor = tf.io.serialize_tensor(tf.convert_to_tensor(numpy_array, dtype=tf.int32))
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_tensor.numpy()]))
            
        with tf.io.TFRecordWriter(output_path) as writer:
            for i in range(n_examples):
                # Generate synthetic data
                sequence = ''.join(np.random.choice(['A', 'C', 'G', 'T'], self.sequence_length))
                atac_profile = np.random.rand(self.output_length, 1).astype(np.float16) * 10
                peaks_center = np.random.randint(0, self.output_length, self.output_length).astype(np.int32)
                motif_activity = np.random.rand(self.motif_features).astype(np.float16)
                
                # Create TFRecord features
                feature = {
                    'sequence': _bytes_feature(sequence.encode()),
                    'atac': _tensor_feature_float(atac_profile),
                    'peaks_center': _tensor_feature_int(peaks_center),
                    'motif_activity': _tensor_feature_float(motif_activity)
                }
                
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example_proto.SerializeToString())
                
        logger.info(f"Created synthetic TFRecord: {output_path}")
        return output_path
        
    def create_synthetic_genomic_files(self) -> Dict[str, str]:
        """Create synthetic genomic data files for testing"""
        
        files = {}
        
        # Create genome FASTA
        fasta_path = self.test_dir / "test_genome.fa"
        with open(fasta_path, 'w') as f:
            f.write(">chr1\n")
            sequence = ''.join(np.random.choice(['A', 'C', 'G', 'T'], 100000))
            f.write(sequence + "\n")
            f.write(">chr2\n")
            sequence = ''.join(np.random.choice(['A', 'C', 'G', 'T'], 50000))
            f.write(sequence + "\n")
        files['genome_fasta'] = str(fasta_path)
        
        # Create chromosome sizes
        sizes_path = self.test_dir / "test_genome.sizes"
        with open(sizes_path, 'w') as f:
            f.write("chr1\t100000\n")
            f.write("chr2\t50000\n")
        files['genome_sizes'] = str(sizes_path)
        
        # Create ATAC bedGraph
        atac_path = self.test_dir / "test_atac.bedgraph"
        with open(atac_path, 'w') as f:
            for i in range(0, 100000, 1000):
                value = np.random.rand() * 10
                f.write(f"chr1\t{i}\t{i+1000}\t{value:.3f}\n")
        files['atac_bedgraph'] = str(atac_path)
        
        # Create RAMPAGE bedGraph
        rampage_path = self.test_dir / "test_rampage.bedgraph"
        with open(rampage_path, 'w') as f:
            for i in range(0, 100000, 1000):
                value = np.random.rand() * 5
                f.write(f"chr1\t{i}\t{i+1000}\t{value:.3f}\n")
        files['rampage_bedgraph'] = str(rampage_path)
        
        # Create peaks BED
        peaks_path = self.test_dir / "test_peaks.bed"
        with open(peaks_path, 'w') as f:
            for i in range(10):
                start = np.random.randint(1000, 90000)
                end = start + np.random.randint(200, 2000)
                f.write(f"chr1\t{start}\t{end}\tpeak_{i}\n")
        files['peaks_bed'] = str(peaks_path)
        
        # Create regions BED
        regions_path = self.test_dir / "test_regions.bed"
        with open(regions_path, 'w') as f:
            f.write(f"chr1\t10000\t{10000 + self.sequence_length}\tregion_1\n")
            f.write(f"chr1\t30000\t{30000 + self.sequence_length}\tregion_2\n")
            f.write(f"chr2\t5000\t{5000 + self.sequence_length}\tregion_3\n")
        files['regions_bed'] = str(regions_path)
        
        return files
        
    def test_tfrecord_conversion(self) -> bool:
        """Test TFRecord to HDF5 conversion"""
        
        logger.info("Testing TFRecord to HDF5 conversion...")
        
        try:
            # Create synthetic TFRecord
            tfrecord_path = self.test_dir / "test_data.tfrecord"
            self.create_synthetic_tfrecord(str(tfrecord_path), self.n_test_examples)
            
            # Initialize converter
            converter = TFRecordToHDF5Converter(
                sequence_length=self.sequence_length,
                output_length=self.output_length,
                motif_features=self.motif_features,
                one_hot_encode=True
            )
            
            # Convert TFRecord to HDF5
            hdf5_path = self.test_dir / "converted_data.h5"
            converted_file = converter.convert_tfrecord_file(
                str(tfrecord_path), str(hdf5_path), "test_sample"
            )
            
            # Validate converted file
            validation_info = validate_converted_hdf5(converted_file)
            
            if not validation_info['valid']:
                logger.error(f"Validation failed: {validation_info['error']}")
                return False
                
            # Check data shapes and content
            with h5py.File(converted_file, 'r') as f:
                sequences = f['sequences'][:]
                atac_profiles = f['atac_profiles'][:]
                peaks_centers = f['peaks_centers'][:]
                motif_activities = f['motif_activities'][:]
                
                # Verify shapes
                expected_shapes = {
                    'sequences': (self.n_test_examples, 4, self.sequence_length),
                    'atac_profiles': (self.n_test_examples, self.output_length),
                    'peaks_centers': (self.n_test_examples, self.output_length),
                    'motif_activities': (self.n_test_examples, self.motif_features)
                }
                
                for dataset_name, expected_shape in expected_shapes.items():
                    actual_shape = f[dataset_name].shape
                    if actual_shape != expected_shape:
                        logger.error(f"Shape mismatch for {dataset_name}: expected {expected_shape}, got {actual_shape}")
                        return False
                        
                # Verify one-hot encoding
                if not np.allclose(sequences.sum(axis=1), 1.0, atol=1e-6):
                    logger.error("Sequences are not properly one-hot encoded")
                    return False
                    
                # Verify non-negative values
                if np.any(atac_profiles < 0):
                    logger.error("ATAC profiles contain negative values")
                    return False
                    
            logger.info("✓ TFRecord conversion test passed")
            return True
            
        except Exception as e:
            logger.error(f"TFRecord conversion test failed: {e}")
            return False
            
    def test_pytorch_dataset_generation(self) -> bool:
        """Test PyTorch dataset generation from genomic files"""
        
        logger.info("Testing PyTorch dataset generation...")
        
        try:
            # Create synthetic genomic files
            files = self.create_synthetic_genomic_files()
            
            # Initialize generator
            generator = PyTorchDatasetGenerator(
                genome_fasta=files['genome_fasta'],
                genome_sizes=files['genome_sizes'],
                sequence_length=self.sequence_length,
                output_length=self.output_length,
                resolution=self.sequence_length // self.output_length,
                motif_features=self.motif_features
            )
            
            # Create sample configuration
            sample_config = {
                'sample_id': 'test_sample',
                'atac_file': files['atac_bedgraph'],
                'rampage_file': files['rampage_bedgraph'],
                'peaks_file': files['peaks_bed'],
                'regions_file': files['regions_bed']
            }
            
            # Generate dataset
            hdf5_path = self.test_dir / "pytorch_dataset.h5"
            generated_file = generator.generate_sample_dataset(
                sample_config, str(hdf5_path)
            )
            
            # Validate generated file
            validation_info = validate_converted_hdf5(generated_file)
            
            if not validation_info['valid']:
                logger.error(f"Validation failed: {validation_info['error']}")
                return False
                
            # Check data content
            with h5py.File(generated_file, 'r') as f:
                sequences = f['sequences'][:]
                atac_profiles = f['atac_profiles'][:]
                rampage_profiles = f['rampage_profiles'][:]
                motif_activities = f['motif_activities'][:]
                peaks_centers = f['peaks_centers'][:]
                
                # Verify we have data
                if len(sequences) == 0:
                    logger.error("No sequences generated")
                    return False
                    
                # Verify one-hot encoding
                if not np.allclose(sequences.sum(axis=1), 1.0, atol=1e-6):
                    logger.error("Sequences are not properly one-hot encoded")
                    return False
                    
                # Verify profiles are non-negative
                if np.any(atac_profiles < 0) or np.any(rampage_profiles < 0):
                    logger.error("Signal profiles contain negative values")
                    return False
                    
                # Verify motif activities are in reasonable range
                if not (0 <= motif_activities.min() and motif_activities.max() <= 1):
                    logger.warning("Motif activities outside [0,1] range - this may be expected")
                    
            logger.info("✓ PyTorch dataset generation test passed")
            return True
            
        except Exception as e:
            logger.error(f"PyTorch dataset generation test failed: {e}")
            return False
            
    def test_data_module_integration(self) -> bool:
        """Test integration with PyTorch Lightning data module"""
        
        logger.info("Testing data module integration...")
        
        try:
            # Create test HDF5 file
            hdf5_path = self.test_dir / "data_module_test.h5"
            
            # Generate synthetic data
            n_samples = 20
            sequences = np.random.randint(0, 4, (n_samples, self.sequence_length))
            atac_profiles = np.random.rand(n_samples, self.output_length) * 10
            motif_activities = np.random.rand(n_samples, self.motif_features)
            peaks_centers = np.random.randint(0, self.output_length, (n_samples, 10))
            
            # Convert sequences to one-hot
            sequences_onehot = np.zeros((n_samples, 4, self.sequence_length), dtype=np.float32)
            for i, seq in enumerate(sequences):
                for j, base in enumerate(seq):
                    sequences_onehot[i, base, j] = 1.0
                    
            # Save to HDF5
            with h5py.File(hdf5_path, 'w') as f:
                f.create_dataset('sequences', data=sequences_onehot)
                f.create_dataset('atac_profiles', data=atac_profiles)
                f.create_dataset('motif_activities', data=motif_activities)
                f.create_dataset('peaks_centers', data=peaks_centers)
                
                f.attrs['sample_id'] = 'test_sample'
                f.attrs['n_regions'] = n_samples
                
            # Create data directory structure
            data_dir = self.test_dir / "data"
            train_dir = data_dir / "train"
            train_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy HDF5 file to train directory
            shutil.copy(hdf5_path, train_dir / "data.h5")
            
            # Test data module
            data_module = EpiBERTDataModule(
                data_dir=str(data_dir),
                batch_size=4,
                num_workers=0,  # Avoid multiprocessing issues in tests
                input_length=self.sequence_length,
                output_length=self.output_length
            )
            
            # Setup data module
            data_module.setup("fit")
            
            # Test data loading
            train_loader = data_module.train_dataloader()
            batch = next(iter(train_loader))
            
            # Verify batch structure
            expected_keys = ['sequence', 'atac', 'motif_activity', 'peaks_center', 'target', 'mask', 'unmask']
            for key in expected_keys:
                if key not in batch:
                    logger.error(f"Missing key in batch: {key}")
                    return False
                    
            # Verify batch shapes
            batch_size = batch['sequence'].shape[0]
            if batch['sequence'].shape != (batch_size, 4, self.sequence_length):
                logger.error(f"Incorrect sequence shape: {batch['sequence'].shape}")
                return False
                
            if batch['atac'].shape[1] != self.output_length:
                logger.error(f"Incorrect ATAC shape: {batch['atac'].shape}")
                return False
                
            logger.info("✓ Data module integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"Data module integration test failed: {e}")
            return False
            
    def test_batch_conversion(self) -> bool:
        """Test batch conversion of multiple files"""
        
        logger.info("Testing batch conversion...")
        
        try:
            # Create multiple TFRecord files
            tfrecord_files = []
            for i in range(3):
                tfrecord_path = self.test_dir / f"test_data_{i}.tfrecord"
                self.create_synthetic_tfrecord(str(tfrecord_path), 5)
                tfrecord_files.append(str(tfrecord_path))
                
            # Initialize converter
            converter = TFRecordToHDF5Converter(
                sequence_length=self.sequence_length,
                output_length=self.output_length,
                motif_features=self.motif_features
            )
            
            # Convert multiple files
            output_dir = self.test_dir / "batch_converted"
            converted_files = converter.convert_multiple_tfrecords(
                tfrecord_files,
                str(output_dir),
                num_workers=1  # Avoid multiprocessing issues in tests
            )
            
            # Verify all files were converted
            if len(converted_files) != len(tfrecord_files):
                logger.error(f"Expected {len(tfrecord_files)} converted files, got {len(converted_files)}")
                return False
                
            # Validate each converted file
            for converted_file in converted_files:
                validation_info = validate_converted_hdf5(converted_file)
                if not validation_info['valid']:
                    logger.error(f"Validation failed for {converted_file}: {validation_info['error']}")
                    return False
                    
            logger.info("✓ Batch conversion test passed")
            return True
            
        except Exception as e:
            logger.error(f"Batch conversion test failed: {e}")
            return False
            
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        
        logger.info("Running comprehensive data conversion tests...")
        
        results = {}
        
        if not CONVERSION_MODULES_AVAILABLE:
            logger.error("Conversion modules not available - skipping tests")
            return {'module_import': False}
            
        # Run individual tests
        test_methods = [
            ('tfrecord_conversion', self.test_tfrecord_conversion),
            ('pytorch_generation', self.test_pytorch_dataset_generation),
            ('data_module_integration', self.test_data_module_integration),
            ('batch_conversion', self.test_batch_conversion)
        ]
        
        for test_name, test_method in test_methods:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Running test: {test_name}")
                logger.info(f"{'='*50}")
                
                results[test_name] = test_method()
                
                if results[test_name]:
                    logger.info(f"✓ {test_name} PASSED")
                else:
                    logger.error(f"✗ {test_name} FAILED")
                    
            except Exception as e:
                logger.error(f"✗ {test_name} FAILED with exception: {e}")
                results[test_name] = False
                
        return results


def main():
    """Main test execution"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data conversion utilities")
    parser.add_argument('--test-dir', help='Directory for test files (will be created/cleaned)')
    parser.add_argument('--keep-files', action='store_true', help='Keep test files after completion')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Run tests
    tester = DataConversionTester(args.test_dir)
    
    try:
        results = tester.run_all_tests()
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for test_name, passed in results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{test_name:30} {status}")
            
        print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed!")
            exit_code = 0
        else:
            print("❌ Some tests failed!")
            exit_code = 1
            
        # Clean up unless requested to keep files
        if not args.keep_files:
            print(f"\nCleaning up test directory: {tester.test_dir}")
        else:
            print(f"\nTest files kept in: {tester.test_dir}")
            
        return exit_code
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTest execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)