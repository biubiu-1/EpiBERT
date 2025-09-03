#!/usr/bin/env python3
"""
EpiBERT Weight Management Utility

Comprehensive utility for managing EpiBERT model weights between
TensorFlow and PyTorch formats. Provides easy-to-use functions
for conversion, validation, and loading.
"""

import os
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from scripts.tf_to_pytorch_converter import (
        TensorFlowToPyTorchConverter,
        load_tensorflow_weights_to_pytorch,
        create_weight_mapping_guide
    )
    from lightning_transfer.epibert_lightning import EpiBERTLightning
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESS = False


def convert_tensorflow_checkpoint(tf_path: str, 
                                 pytorch_path: str,
                                 model_type: str = "pretraining",
                                 validate: bool = True) -> bool:
    """
    Convert TensorFlow checkpoint to PyTorch format
    
    Args:
        tf_path: Path to TensorFlow checkpoint
        pytorch_path: Output path for PyTorch checkpoint
        model_type: Model type ("pretraining" or "finetuning")
        validate: Whether to validate converted weights
        
    Returns:
        True if conversion successful
    """
    if not IMPORTS_SUCCESS:
        print("Error: Required imports failed")
        return False
        
    try:
        print(f"Converting {model_type} model from TensorFlow to PyTorch")
        print(f"Input: {tf_path}")
        print(f"Output: {pytorch_path}")
        
        converter = TensorFlowToPyTorchConverter(model_type=model_type)
        stats = converter.convert_checkpoint(tf_path, pytorch_path, validate=validate)
        
        print(f"Conversion completed successfully!")
        print(f"Converted {stats['tf_params']} -> {stats['pytorch_params']} parameters")
        
        return True
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False


def load_and_validate_model(tf_checkpoint_path: str,
                           model_type: str = "pretraining",
                           tolerance: float = 1e-5):
    """
    Load TensorFlow weights into PyTorch model and validate
    
    Args:
        tf_checkpoint_path: Path to TensorFlow checkpoint
        model_type: Model type ("pretraining" or "finetuning")
        tolerance: Tolerance for weight validation
        
    Returns:
        Loaded PyTorch model or None if failed
    """
    if not IMPORTS_SUCCESS:
        print("Error: Required imports failed")
        return None
        
    try:
        print(f"Creating {model_type} model...")
        model = EpiBERTLightning(model_type=model_type)
        
        print(f"Loading TensorFlow weights...")
        load_stats = model.load_tensorflow_weights(tf_checkpoint_path, verbose=True)
        
        if not load_stats.get('loaded_successfully', False):
            print(f"Failed to load weights: {load_stats}")
            return None
            
        print(f"Validating weights...")
        validation_stats = model.validate_weights_against_tensorflow(
            tf_checkpoint_path, tolerance=tolerance
        )
        
        if validation_stats.get('validation_passed', False):
            print("✓ Weight validation passed!")
        else:
            print("⚠ Weight validation failed!")
            print(f"Validation details: {validation_stats}")
            
        return model
        
    except Exception as e:
        print(f"Failed to load and validate model: {e}")
        return None


def create_demo_model(model_type: str = "pretraining",
                     save_path: Optional[str] = None):
    """
    Create a demo model with random weights for testing
    
    Args:
        model_type: Model type ("pretraining" or "finetuning")
        save_path: Optional path to save model
        
    Returns:
        EpiBERT Lightning model
    """
    if not IMPORTS_SUCCESS:
        print("Error: Required imports failed")
        return None
        
    print(f"Creating demo {model_type} model...")
    model = EpiBERTLightning(model_type=model_type)
    
    print(f"Model details:")
    print(f"  Heads: {model.num_heads}")
    print(f"  Layers: {model.num_transformer_layers}")
    print(f"  d_model: {model.d_model}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if save_path:
        model.save_pytorch_checkpoint(save_path)
        print(f"Saved demo model to: {save_path}")
        
    return model


def compare_model_outputs(model1, model2, tolerance: float = 1e-5) -> Dict[str, Any]:
    """
    Compare outputs between two models (e.g., original vs converted)
    
    Args:
        model1: First model
        model2: Second model
        tolerance: Tolerance for output comparison
        
    Returns:
        Comparison results
    """
    import torch
    
    # Create dummy input
    batch_size = 1
    seq_length = model1.input_length
    
    # DNA sequence (one-hot encoded)
    sequence = torch.randn(batch_size, 4, seq_length)
    
    # ATAC profile
    atac_profile = torch.randn(batch_size, 1, model1.output_length)
    
    # Motif activity
    motif_activity = torch.randn(batch_size, model1.num_motifs)
    
    # Set models to eval mode
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        output1 = model1(sequence, atac_profile, motif_activity)
        output2 = model2(sequence, atac_profile, motif_activity)
        
    # Compare outputs
    if isinstance(output1, tuple) and isinstance(output2, tuple):
        max_diff = max(torch.abs(o1 - o2).max().item() for o1, o2 in zip(output1, output2))
    else:
        max_diff = torch.abs(output1 - output2).max().item()
        
    results = {
        'max_difference': max_diff,
        'outputs_match': max_diff < tolerance,
        'tolerance': tolerance
    }
    
    print(f"Output comparison: max_diff={max_diff:.2e}, match={results['outputs_match']}")
    
    return results


def list_available_checkpoints(directory: str) -> Dict[str, list]:
    """
    List available TensorFlow and PyTorch checkpoints in directory
    
    Args:
        directory: Directory to search
        
    Returns:
        Dictionary with lists of found checkpoints
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return {'tensorflow': [], 'pytorch': []}
        
    # Common TensorFlow checkpoint patterns
    tf_patterns = ['*.ckpt*', '*.h5', '*.pb', 'saved_model']
    pytorch_patterns = ['*.pth', '*.pt', '*.pytorch']
    
    tf_checkpoints = []
    for pattern in tf_patterns:
        tf_checkpoints.extend(directory.glob(pattern))
        
    pytorch_checkpoints = []
    for pattern in pytorch_patterns:
        pytorch_checkpoints.extend(directory.glob(pattern))
        
    results = {
        'tensorflow': [str(p) for p in tf_checkpoints],
        'pytorch': [str(p) for p in pytorch_checkpoints]
    }
    
    print(f"Found in {directory}:")
    print(f"  TensorFlow checkpoints: {len(results['tensorflow'])}")
    print(f"  PyTorch checkpoints: {len(results['pytorch'])}")
    
    for tf_ckpt in results['tensorflow'][:5]:  # Show first 5
        print(f"    TF: {tf_ckpt}")
    if len(results['tensorflow']) > 5:
        print(f"    ... and {len(results['tensorflow']) - 5} more")
        
    for pt_ckpt in results['pytorch'][:5]:  # Show first 5
        print(f"    PT: {pt_ckpt}")
    if len(results['pytorch']) > 5:
        print(f"    ... and {len(results['pytorch']) - 5} more")
        
    return results


def main():
    """Command-line interface for weight management utilities"""
    
    parser = argparse.ArgumentParser(description="EpiBERT Weight Management Utility")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert TensorFlow checkpoint to PyTorch')
    convert_parser.add_argument('tf_checkpoint', help='Path to TensorFlow checkpoint')
    convert_parser.add_argument('pytorch_checkpoint', help='Output path for PyTorch checkpoint')
    convert_parser.add_argument('--model-type', choices=['pretraining', 'finetuning'], 
                               default='pretraining', help='Model type')
    convert_parser.add_argument('--no-validate', action='store_true', help='Skip validation')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load and validate TensorFlow weights in PyTorch model')
    load_parser.add_argument('tf_checkpoint', help='Path to TensorFlow checkpoint')
    load_parser.add_argument('--model-type', choices=['pretraining', 'finetuning'],
                            default='pretraining', help='Model type')
    load_parser.add_argument('--tolerance', type=float, default=1e-5, help='Validation tolerance')
    load_parser.add_argument('--save', help='Path to save validated PyTorch model')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Create demo model')
    demo_parser.add_argument('--model-type', choices=['pretraining', 'finetuning'],
                            default='pretraining', help='Model type')
    demo_parser.add_argument('--save', help='Path to save demo model')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available checkpoints')
    list_parser.add_argument('directory', help='Directory to search')
    
    # Guide command
    guide_parser = subparsers.add_parser('guide', help='Create weight mapping guide')
    guide_parser.add_argument('--model-type', choices=['pretraining', 'finetuning'],
                             default='pretraining', help='Model type')
    guide_parser.add_argument('--output', help='Output file for guide')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    if not IMPORTS_SUCCESS:
        print("Error: Failed to import required modules")
        print("Make sure you're running from the EpiBERT project directory")
        return 1
        
    # Execute commands
    if args.command == 'convert':
        success = convert_tensorflow_checkpoint(
            args.tf_checkpoint,
            args.pytorch_checkpoint, 
            args.model_type,
            validate=not args.no_validate
        )
        return 0 if success else 1
        
    elif args.command == 'load':
        model = load_and_validate_model(
            args.tf_checkpoint,
            args.model_type,
            args.tolerance
        )
        
        if model is None:
            return 1
            
        if args.save:
            model.save_pytorch_checkpoint(args.save)
            print(f"Saved validated model to: {args.save}")
            
        return 0
        
    elif args.command == 'demo':
        model = create_demo_model(args.model_type, args.save)
        return 0 if model is not None else 1
        
    elif args.command == 'list':
        list_available_checkpoints(args.directory)
        return 0
        
    elif args.command == 'guide':
        guide = create_weight_mapping_guide(args.model_type)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(guide)
            print(f"Saved mapping guide to: {args.output}")
        else:
            print(guide)
            
        return 0
        
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())