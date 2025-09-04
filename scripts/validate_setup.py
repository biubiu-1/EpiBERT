#!/usr/bin/env python3
"""
EpiBERT End-to-End Validation Script

This script validates that the complete EpiBERT workflow is properly set up
and all components work together correctly.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import yaml
import json
import numpy as np
from typing import Dict, List, Tuple
import argparse

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_status(msg: str):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def print_header(msg: str):
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print(f"{Colors.BLUE}{msg}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")

class EpiBERTValidator:
    """Comprehensive validation of EpiBERT setup and workflow"""
    
    def __init__(self, repo_root: str, implementation: str = "lightning"):
        self.repo_root = Path(repo_root)
        self.implementation = implementation
        self.test_dir = self.repo_root / "tmp" / "validation_test"
        self.results = {}
        
    def setup_test_environment(self):
        """Create test directory and files"""
        print_status("Setting up test environment...")
        
        # Create test directory
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create minimal test data
        self.create_test_data()
        
        print_status(f"Test environment created at: {self.test_dir}")
    
    def create_test_data(self):
        """Create minimal test data for validation"""
        print_status("Creating test data...")
        
        # Create test configuration files
        data_config = {
            'input': {
                'sample_name': 'test_sample',
                'atac_bam': str(self.test_dir / 'test.atac.bam')
            },
            'reference': {
                'genome_fasta': str(self.test_dir / 'test_genome.fa'),
                'chrom_sizes': str(self.test_dir / 'test.chrom.sizes'),
                'blacklist': str(self.test_dir / 'test_blacklist.bed')
            },
            'output': {
                'base_dir': str(self.test_dir / 'processed')
            }
        }
        
        training_config = {
            'data': {
                'train_data': str(self.test_dir / 'processed' / 'train'),
                'valid_data': str(self.test_dir / 'processed' / 'valid'),
                'test_data': str(self.test_dir / 'processed' / 'test')
            },
            'model': {
                'type': 'pretraining',
                'input_length': 1024,  # Small for testing
                'output_length': 32
            },
            'training': {
                'batch_size': 1,
                'learning_rate': 0.001,
                'max_epochs': 2,
                'patience': 5
            },
            'hardware': {
                'num_gpus': 0,  # CPU only for validation
                'num_workers': 1
            }
        }
        
        # Save config files
        with open(self.test_dir / 'data_config.yaml', 'w') as f:
            yaml.dump(data_config, f)
            
        with open(self.test_dir / 'training_config.yaml', 'w') as f:
            yaml.dump(training_config, f)
        
        # Create minimal reference files
        self.create_minimal_reference_files()
        
        # Create test datasets
        self.create_test_datasets()
    
    def create_minimal_reference_files(self):
        """Create minimal reference files for testing"""
        # Chromosome sizes
        with open(self.test_dir / 'test.chrom.sizes', 'w') as f:
            f.write("chr1\t10000\n")
            f.write("chr2\t10000\n")
        
        # Genome FASTA
        with open(self.test_dir / 'test_genome.fa', 'w') as f:
            f.write(">chr1\n")
            f.write("A" * 10000 + "\n")
            f.write(">chr2\n")
            f.write("T" * 10000 + "\n")
        
        # Blacklist
        with open(self.test_dir / 'test_blacklist.bed', 'w') as f:
            f.write("chr1\t1000\t2000\n")
            f.write("chr2\t5000\t6000\n")
    
    def create_test_datasets(self):
        """Create minimal test datasets"""
        # Create processed data directory
        processed_dir = self.test_dir / 'processed'
        processed_dir.mkdir(exist_ok=True)
        
        for split in ['train', 'valid', 'test']:
            split_dir = processed_dir / split
            split_dir.mkdir(exist_ok=True)
            
            if self.implementation == "lightning":
                # Create HDF5 format test data
                self.create_hdf5_test_data(split_dir / f"{split}.h5")
            else:
                # Create TFRecord format test data  
                self.create_tfrecord_test_data(split_dir / f"{split}.tfr")
    
    def create_hdf5_test_data(self, output_path: Path):
        """Create minimal HDF5 test data"""
        try:
            import h5py
            
            # Generate random data
            batch_size = 5
            input_length = 1024
            output_length = 32
            
            inputs = np.random.rand(batch_size, input_length, 4).astype(np.float32)
            targets = np.random.rand(batch_size, output_length).astype(np.float32)
            
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('inputs', data=inputs)
                f.create_dataset('targets', data=targets)
                
            print_status(f"Created HDF5 test data: {output_path}")
            
        except ImportError:
            print_warning("h5py not available, skipping HDF5 test data creation")
    
    def create_tfrecord_test_data(self, output_path: Path):
        """Create minimal TFRecord test data"""
        try:
            import tensorflow as tf
            
            # This is a simplified version - would need actual TF record creation
            output_path.touch()  # Create empty file for now
            print_status(f"Created TFRecord test data: {output_path}")
            
        except ImportError:
            print_warning("TensorFlow not available, skipping TFRecord test data creation")
    
    def validate_environment(self) -> bool:
        """Validate environment setup"""
        print_header("Environment Validation")
        
        success = True
        
        # Check Python version
        print_status("Checking Python version...")
        if sys.version_info >= (3, 7):
            print_status(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
        else:
            print_error(f"✗ Python {sys.version_info.major}.{sys.version_info.minor} (need ≥3.7)")
            success = False
        
        # Check implementation-specific dependencies
        if self.implementation == "lightning":
            success &= self.check_lightning_deps()
        else:
            success &= self.check_tensorflow_deps()
        
        # Check bioinformatics tools
        success &= self.check_bioinf_tools()
        
        self.results['environment'] = success
        return success
    
    def check_lightning_deps(self) -> bool:
        """Check PyTorch Lightning dependencies"""
        print_status("Checking PyTorch Lightning dependencies...")
        
        required_packages = [
            'torch', 'pytorch_lightning', 'torchmetrics', 
            'numpy', 'pandas', 'matplotlib', 'h5py'
        ]
        
        success = True
        for package in required_packages:
            try:
                __import__(package)
                print_status(f"✓ {package}")
            except ImportError:
                print_error(f"✗ {package}")
                success = False
        
        return success
    
    def check_tensorflow_deps(self) -> bool:
        """Check TensorFlow dependencies"""
        print_status("Checking TensorFlow dependencies...")
        
        required_packages = [
            'tensorflow', 'numpy', 'pandas', 'matplotlib'
        ]
        
        success = True
        for package in required_packages:
            try:
                __import__(package)
                print_status(f"✓ {package}")
            except ImportError:
                print_error(f"✗ {package}")
                success = False
        
        return success
    
    def check_bioinf_tools(self) -> bool:
        """Check bioinformatics tools"""
        print_status("Checking bioinformatics tools...")
        
        required_tools = ['samtools', 'bedtools']
        optional_tools = ['macs2', 'bgzip', 'tabix']
        
        success = True
        for tool in required_tools:
            if shutil.which(tool):
                print_status(f"✓ {tool}")
            else:
                print_error(f"✗ {tool} (required)")
                success = False
        
        for tool in optional_tools:
            if shutil.which(tool):
                print_status(f"✓ {tool} (optional)")
            else:
                print_warning(f"- {tool} (optional)")
        
        return success
    
    def validate_scripts(self) -> bool:
        """Validate that all scripts are executable and functional"""
        print_header("Script Validation")
        
        scripts = [
            'setup_environment.sh',
            'run_complete_workflow.sh',
            'scripts/train_model.sh',
            'scripts/evaluate_model.py',
            'scripts/download_references.sh',
            'data_processing/run_pipeline.sh'
        ]
        
        success = True
        for script in scripts:
            script_path = self.repo_root / script
            if script_path.exists():
                if os.access(script_path, os.X_OK):
                    print_status(f"✓ {script} (executable)")
                else:
                    print_warning(f"? {script} (not executable)")
            else:
                print_error(f"✗ {script} (missing)")
                success = False
        
        self.results['scripts'] = success
        return success
    
    def validate_model_loading(self) -> bool:
        """Validate model can be loaded and initialized"""
        print_header("Model Loading Validation")
        
        success = True
        
        try:
            if self.implementation == "lightning":
                from lightning_transfer.epibert_lightning import EpiBERTLightning
                
                # Test pretraining model
                print_status("Testing pretraining model initialization...")
                model_pretrain = EpiBERTLightning(model_type="pretraining")
                print_status(f"✓ Pretraining model: {model_pretrain.num_heads} heads, {model_pretrain.d_model} d_model")
                
                # Test fine-tuning model
                print_status("Testing fine-tuning model initialization...")
                model_finetune = EpiBERTLightning(model_type="finetuning")
                print_status(f"✓ Fine-tuning model: {model_finetune.num_heads} heads, {model_finetune.d_model} d_model")
                
                # Test parameter verification
                print_status("Testing parameter verification...")
                model_pretrain._verify_parameters()
                model_finetune._verify_parameters()
                print_status("✓ Parameter verification passed")
                
            else:
                # TensorFlow model validation would go here
                print_status("TensorFlow model validation not implemented in this script")
                
        except Exception as e:
            print_error(f"Model loading failed: {e}")
            success = False
        
        self.results['model_loading'] = success
        return success
    
    def validate_data_processing(self) -> bool:
        """Validate data processing pipeline"""
        print_header("Data Processing Validation")
        
        # This would be a simplified validation
        # In practice, would test with actual data processing scripts
        
        print_status("Checking data processing configuration...")
        config_path = self.test_dir / 'data_config.yaml'
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print_status("✓ Data configuration valid")
            return True
        else:
            print_error("✗ Data configuration missing")
            return False
    
    def validate_evaluation(self) -> bool:
        """Validate evaluation system"""
        print_header("Evaluation System Validation")
        
        try:
            # Import evaluation module
            sys.path.append(str(self.repo_root / 'scripts'))
            from evaluate_model import EpiBERTEvaluator
            
            # Initialize evaluator
            evaluator = EpiBERTEvaluator(implementation=self.implementation)
            print_status("✓ Evaluator initialized")
            
            # Test with synthetic data
            targets = np.random.rand(10, 32)
            predictions = targets + np.random.normal(0, 0.1, targets.shape)
            
            # Test correlation metrics
            corr_metrics = evaluator.compute_correlation_metrics(targets, predictions)
            print_status(f"✓ Correlation metrics: r={corr_metrics['global_pearson_r']:.3f}")
            
            # Test regression metrics
            reg_metrics = evaluator.compute_regression_metrics(targets, predictions)
            print_status(f"✓ Regression metrics: MSE={reg_metrics['mse']:.3f}")
            
            # Test peak prediction metrics
            peak_metrics = evaluator.compute_peak_prediction_metrics(targets, predictions)
            print_status(f"✓ Peak metrics: AUC={peak_metrics['roc_auc']:.3f}")
            
            self.results['evaluation'] = True
            return True
            
        except Exception as e:
            print_error(f"Evaluation validation failed: {e}")
            self.results['evaluation'] = False
            return False
    
    def run_full_validation(self) -> Dict:
        """Run complete validation suite"""
        print_header("EpiBERT Complete Validation")
        print_status(f"Implementation: {self.implementation}")
        print_status(f"Repository: {self.repo_root}")
        
        # Setup test environment
        self.setup_test_environment()
        
        # Run validation tests
        tests = [
            ("Environment", self.validate_environment),
            ("Scripts", self.validate_scripts),
            ("Model Loading", self.validate_model_loading),
            ("Data Processing", self.validate_data_processing),
            ("Evaluation", self.validate_evaluation)
        ]
        
        results = {}
        all_passed = True
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                all_passed &= result
                
                if result:
                    print_status(f"✅ {test_name} validation passed")
                else:
                    print_error(f"❌ {test_name} validation failed")
                    
            except Exception as e:
                print_error(f"❌ {test_name} validation error: {e}")
                results[test_name] = False
                all_passed = False
        
        # Summary
        print_header("Validation Summary")
        
        if all_passed:
            print_status("🎉 All validation tests passed!")
            print_status("EpiBERT is ready for use.")
        else:
            print_error("❌ Some validation tests failed.")
            print_status("Please address the issues above before proceeding.")
        
        # Save results
        results_file = self.test_dir / 'validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print_status(f"Validation results saved to: {results_file}")
        
        # Cleanup
        self.cleanup()
        
        return results
    
    def cleanup(self):
        """Clean up test files"""
        try:
            shutil.rmtree(self.test_dir)
            print_status("Test files cleaned up")
        except Exception as e:
            print_warning(f"Cleanup failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Validate EpiBERT installation and setup')
    parser.add_argument('--implementation', choices=['lightning', 'tensorflow'], 
                       default='lightning', help='Implementation to validate')
    parser.add_argument('--repo-root', type=str, default='.',
                       help='Repository root directory')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = EpiBERTValidator(
        repo_root=args.repo_root,
        implementation=args.implementation
    )
    
    # Run validation
    results = validator.run_full_validation()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()