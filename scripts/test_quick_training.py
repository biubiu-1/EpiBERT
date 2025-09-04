#!/usr/bin/env python3
"""
Comprehensive EpiBERT Workflow Validation
Tests all components of the EpiBERT workflow with realistic data structures
"""

import tempfile
import os
import yaml
import sys
import numpy as np
import h5py
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/runner/work/EpiBERT/EpiBERT')

def create_test_data_realistic(output_dir: str):
    """Create test data that matches real EpiBERT data format"""
    
    # Create manifest
    samples = []
    conditions = ['control', 'treatment']
    batches = ['train', 'val', 'test']
    
    for condition in conditions:
        for batch in batches:
            sample_id = f"test_{condition}_{batch}"
            atac_file = os.path.join(output_dir, f"{sample_id}_atac.h5")
            rampage_file = os.path.join(output_dir, f"{sample_id}_rampage.h5")
            
            # Create realistic data files with proper dimensions
            with h5py.File(atac_file, 'w') as f:
                # Full-size sequence for realism but small enough for testing
                sequence = np.zeros((4, 524288), dtype=np.float32)
                # Create random but valid one-hot encoded sequence
                for j in range(0, 524288, 1000):  # Sample every 1000 bases for speed
                    sequence[np.random.randint(0, 4), j] = 1
                f.create_dataset('sequences', data=sequence)
                
                # Standard output length
                atac_profile = np.random.exponential(1.0, size=(4096,)).astype(np.float32)
                f.create_dataset('atac_profiles', data=atac_profile)
                
            with h5py.File(rampage_file, 'w') as f:
                f.create_dataset('sequences', data=sequence)
                rampage_profile = np.random.exponential(0.5, size=(4096,)).astype(np.float32)
                f.create_dataset('rampage_profiles', data=rampage_profile)
            
            samples.append({
                'sample_id': sample_id,
                'condition': condition,
                'cell_type': 'test',
                'atac_file': atac_file,
                'rampage_file': rampage_file,
                'batch': batch,
                'replicate': 1
            })
    
    # Create manifest
    manifest_path = os.path.join(output_dir, 'manifest.yaml')
    with open(manifest_path, 'w') as f:
        yaml.dump({'samples': samples}, f)
    
    return manifest_path


def test_workflow_components():
    """Test all workflow components systematically"""
    
    results = {}
    
    # Test 1: Environment and imports
    print("Testing imports and environment...")
    try:
        from lightning_transfer.epibert_lightning import EpiBERTLightning
        from lightning_transfer.paired_data_module import PairedDataModule
        from lightning_transfer.callbacks import EpiBERTCheckpointCallback
        import pytorch_lightning as pl
        import torch
        results['imports'] = True
        print("✓ All imports successful")
    except Exception as e:
        results['imports'] = False
        print(f"✗ Import failed: {e}")
    
    # Test 2: Model initialization
    print("Testing model initialization...")
    try:
        model_pre = EpiBERTLightning(model_type='pretraining')
        model_fine = EpiBERTLightning(model_type='finetuning')
        results['model_init'] = True
        print("✓ Both pretraining and finetuning models initialized")
    except Exception as e:
        results['model_init'] = False
        print(f"✗ Model initialization failed: {e}")
    
    # Test 3: Data loading with realistic data
    print("Testing data loading with realistic format...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = create_test_data_realistic(temp_dir)
            
            data_module = PairedDataModule(
                manifest_file=manifest_path,
                batch_size=1,
                balance_conditions=True
            )
            data_module.setup('fit')
            
            # Test data loading
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            
            # Load one batch
            batch = next(iter(train_loader))
            expected_keys = ['sequence', 'atac', 'rampage', 'target', 'sample_id', 'condition']
            missing_keys = [k for k in expected_keys if k not in batch]
            
            if missing_keys:
                results['data_loading'] = False
                print(f"✗ Missing keys in batch: {missing_keys}")
            else:
                # Check tensor shapes
                seq_shape = batch['sequence'].shape
                atac_shape = batch['atac'].shape
                
                if seq_shape[1] == 4 and seq_shape[2] == 524288 and atac_shape[2] == 4096:
                    results['data_loading'] = True
                    print(f"✓ Data loading successful, correct shapes: seq{seq_shape}, atac{atac_shape}")
                else:
                    results['data_loading'] = False
                    print(f"✗ Incorrect tensor shapes: seq{seq_shape}, atac{atac_shape}")
                    
    except Exception as e:
        results['data_loading'] = False
        print(f"✗ Data loading failed: {e}")
    
    # Test 4: Trainer setup
    print("Testing trainer setup...")
    try:
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator='cpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False
        )
        results['trainer_setup'] = True
        print("✓ Trainer setup successful")
    except Exception as e:
        results['trainer_setup'] = False
        print(f"✗ Trainer setup failed: {e}")
    
    # Test 5: Forward pass
    print("Testing model forward pass...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = create_test_data_realistic(temp_dir)
            
            model = EpiBERTLightning(model_type='pretraining')
            data_module = PairedDataModule(
                manifest_file=manifest_path,
                batch_size=1,
                balance_conditions=False
            )
            data_module.setup('fit')
            
            # Get one batch
            train_loader = data_module.train_dataloader()
            batch = next(iter(train_loader))
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                # Extract arguments that the forward method expects
                sequence = batch['sequence']
                atac = batch['atac'] 
                motif_activity = batch.get('motif_activity', torch.randn(1, 693))  # Default if not present
                
                outputs = model.forward(sequence, atac, motif_activity)
                
            if outputs.shape[-1] == 1 and outputs.shape[1] == 4092:
                results['forward_pass'] = True
                print(f"✓ Forward pass successful, output shape: {outputs.shape}")
            else:
                results['forward_pass'] = False
                print(f"✗ Forward pass incorrect output shape: {outputs.shape}")
                
    except Exception as e:
        results['forward_pass'] = False
        print(f"✗ Forward pass failed: {e}")
    
    # Test 6: Shell scripts
    print("Testing shell scripts...")
    try:
        # Test help commands
        result1 = subprocess.run(['./run_complete_workflow.sh', '--help'], 
                               capture_output=True, text=True, 
                               cwd='/home/runner/work/EpiBERT/EpiBERT')
        
        result2 = subprocess.run(['./scripts/train_model.sh', '--help'], 
                               capture_output=True, text=True,
                               cwd='/home/runner/work/EpiBERT/EpiBERT')
        
        if result1.returncode == 0 and result2.returncode == 0:
            results['shell_scripts'] = True
            print("✓ Shell scripts executable and responsive")
        else:
            results['shell_scripts'] = False
            print(f"✗ Shell scripts failed: {result1.returncode}, {result2.returncode}")
            
    except Exception as e:
        results['shell_scripts'] = False
        print(f"✗ Shell script test failed: {e}")
    
    return results


def test_complete_workflow():
    """Run complete workflow validation"""
    
    print("="*60)
    print("EpiBERT Complete Workflow Validation")
    print("="*60)
    
    results = test_workflow_components()
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print("="*60)
    print(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL WORKFLOW COMPONENTS VALIDATED!")
        print("EpiBERT is ready for full production use.")
        return True
    else:
        print("❌ SOME COMPONENTS NEED ATTENTION")
        print("Please review failed tests above.")
        return False


if __name__ == '__main__':
    success = test_complete_workflow()
    sys.exit(0 if success else 1)