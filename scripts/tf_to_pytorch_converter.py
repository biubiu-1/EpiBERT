#!/usr/bin/env python3
"""
TensorFlow to PyTorch Weight Conversion Utility

Converts EpiBERT TensorFlow checkpoints to PyTorch format for use with
Lightning implementation. Handles layer name mapping, weight reshaping,
and parameter validation.
"""

import os
import re
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import OrderedDict

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Cannot read TF checkpoints directly.")


class TensorFlowToPyTorchConverter:
    """
    Converts TensorFlow EpiBERT checkpoints to PyTorch format
    
    Features:
    - Automatic layer name mapping
    - Weight shape validation and transformation
    - Support for both pretraining and finetuning models
    - Comprehensive parameter validation
    """
    
    def __init__(self, model_type: str = "pretraining"):
        """
        Initialize the converter
        
        Args:
            model_type: Either "pretraining" or "finetuning" 
        """
        self.model_type = model_type
        self.layer_mappings = self._get_layer_mappings()
        self.converted_weights = OrderedDict()
        
    def _get_layer_mappings(self) -> Dict[str, str]:
        """
        Define mapping between TensorFlow and PyTorch layer names
        
        Returns:
            Dictionary mapping TF layer names to PyTorch names
        """
        mappings = {
            # Transformer blocks
            'transformer/layer_': 'transformer.layers.',
            '/attention/self/': '.attention.',
            '/attention/output/dense/': '.attention.output_dense_layer.',
            '/intermediate/dense/': '.feed_forward.dense1.',
            '/output/dense/': '.feed_forward.dense2.',
            '/LayerNorm/': '.layer_norm.',
            
            # Attention components
            '/query/': '.query_dense_layer.',
            '/key/': '.key_dense_layer.',
            '/value/': '.value_dense_layer.',
            
            # Embeddings and stem
            'stem_conv/': 'stem_conv.',
            'stem_res_conv/': 'stem_res_conv.',
            'stem_pool/': 'stem_pool.',
            
            # Head layers
            'final_pointwise/': 'final_pointwise.',
            'final_dense/': 'final_dense.',
            'human_atac_head/': 'atac_head.',
            'human_rampage_head/': 'rampage_head.',
            
            # Normalization
            'layer_norm/': 'layer_norm.',
            'LayerNorm/': 'layer_norm.',
            
            # Position embeddings
            'pos_encoding/': 'pos_encoding.',
            
            # Kernel/bias naming
            'kernel:0': 'weight',
            'bias:0': 'bias',
            'gamma:0': 'weight',
            'beta:0': 'bias',
        }
        
        return mappings
        
    def convert_checkpoint(self, 
                          tf_checkpoint_path: str,
                          output_path: str,
                          validate: bool = True) -> Dict[str, Any]:
        """
        Convert TensorFlow checkpoint to PyTorch format
        
        Args:
            tf_checkpoint_path: Path to TensorFlow checkpoint
            output_path: Path for output PyTorch checkpoint
            validate: Whether to validate converted weights
            
        Returns:
            Dictionary with conversion statistics
        """
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for checkpoint conversion")
            
        print(f"Converting TensorFlow checkpoint: {tf_checkpoint_path}")
        print(f"Output path: {output_path}")
        
        # Load TensorFlow checkpoint
        tf_weights = self._load_tf_checkpoint(tf_checkpoint_path)
        
        # Convert weights
        pytorch_weights = self._convert_weights(tf_weights)
        
        # Validate if requested
        if validate:
            validation_results = self._validate_weights(pytorch_weights)
            print(f"Validation results: {validation_results}")
        
        # Save PyTorch checkpoint
        self._save_pytorch_checkpoint(pytorch_weights, output_path)
        
        stats = {
            'tf_params': len(tf_weights),
            'pytorch_params': len(pytorch_weights),
            'model_type': self.model_type,
            'converted_successfully': True
        }
        
        print(f"Conversion completed: {stats}")
        return stats
        
    def _load_tf_checkpoint(self, checkpoint_path: str) -> Dict[str, np.ndarray]:
        """Load weights from TensorFlow checkpoint"""
        
        print("Loading TensorFlow checkpoint...")
        
        # Handle different checkpoint formats
        if checkpoint_path.endswith('.h5'):
            # Keras saved model
            model = tf.keras.models.load_model(checkpoint_path, compile=False)
            weights = {}
            for layer in model.layers:
                for weight in layer.weights:
                    weights[weight.name] = weight.numpy()
                    
        elif os.path.isdir(checkpoint_path):
            # SavedModel format
            model = tf.saved_model.load(checkpoint_path)
            weights = {}
            for var in model.variables:
                weights[var.name] = var.numpy()
                
        else:
            # TensorFlow checkpoint format
            reader = tf.train.load_checkpoint(checkpoint_path)
            var_shape_map = reader.get_variable_to_shape_map()
            weights = {}
            
            for var_name, shape in var_shape_map.items():
                try:
                    weights[var_name] = reader.get_tensor(var_name)
                except Exception as e:
                    print(f"Warning: Could not load {var_name}: {e}")
                    
        print(f"Loaded {len(weights)} parameters from TensorFlow checkpoint")
        return weights
        
    def _convert_weights(self, tf_weights: Dict[str, np.ndarray]) -> OrderedDict:
        """Convert TensorFlow weights to PyTorch format"""
        
        print("Converting weights to PyTorch format...")
        
        pytorch_weights = OrderedDict()
        conversion_log = []
        
        for tf_name, tf_weight in tf_weights.items():
            pytorch_name = self._map_layer_name(tf_name)
            
            if pytorch_name is None:
                print(f"Skipping unmapped parameter: {tf_name}")
                continue
                
            # Convert weight tensor
            pytorch_weight = self._convert_weight_tensor(tf_weight, tf_name, pytorch_name)
            
            if pytorch_weight is not None:
                pytorch_weights[pytorch_name] = pytorch_weight
                conversion_log.append((tf_name, pytorch_name, tf_weight.shape, pytorch_weight.shape))
            else:
                print(f"Failed to convert: {tf_name}")
                
        print(f"Converted {len(pytorch_weights)} parameters")
        
        # Log conversion details
        print("\nConversion details:")
        for tf_name, pt_name, tf_shape, pt_shape in conversion_log[:10]:  # Show first 10
            print(f"  {tf_name} ({tf_shape}) -> {pt_name} ({pt_shape})")
        if len(conversion_log) > 10:
            print(f"  ... and {len(conversion_log) - 10} more")
            
        return pytorch_weights
        
    def _map_layer_name(self, tf_name: str) -> Optional[str]:
        """Map TensorFlow layer name to PyTorch equivalent"""
        
        pytorch_name = tf_name
        
        # Apply all mappings
        for tf_pattern, pytorch_pattern in self.layer_mappings.items():
            pytorch_name = pytorch_name.replace(tf_pattern, pytorch_pattern)
            
        # Handle special cases
        pytorch_name = self._handle_special_cases(pytorch_name, tf_name)
        
        # Clean up the name
        pytorch_name = self._clean_parameter_name(pytorch_name)
        
        return pytorch_name if pytorch_name != tf_name else None
        
    def _handle_special_cases(self, pytorch_name: str, tf_name: str) -> str:
        """Handle special layer name conversions"""
        
        # Handle transformer layer indexing
        layer_match = re.search(r'layer_(\d+)', pytorch_name)
        if layer_match:
            layer_idx = layer_match.group(1)
            pytorch_name = pytorch_name.replace(f'layer_{layer_idx}', f'{layer_idx}')
            
        # Handle attention head dimensions
        if 'attention' in pytorch_name and 'kernel' in tf_name:
            # TensorFlow uses different weight layouts for attention
            pass
            
        # Handle conv layer names
        if 'conv' in pytorch_name.lower():
            pytorch_name = pytorch_name.replace('kernel', 'weight')
            
        return pytorch_name
        
    def _clean_parameter_name(self, name: str) -> str:
        """Clean up parameter name for PyTorch format"""
        
        # Remove TensorFlow-specific suffixes
        name = re.sub(r':0$', '', name)
        
        # Ensure proper PyTorch parameter naming
        if name.endswith('kernel'):
            name = name.replace('kernel', 'weight')
        elif name.endswith('gamma'):
            name = name.replace('gamma', 'weight')
        elif name.endswith('beta'):
            name = name.replace('beta', 'bias')
            
        return name
        
    def _convert_weight_tensor(self, 
                              tf_weight: np.ndarray,
                              tf_name: str,
                              pytorch_name: str) -> Optional[torch.Tensor]:
        """Convert individual weight tensor"""
        
        try:
            # Convert to PyTorch tensor
            pytorch_tensor = torch.from_numpy(tf_weight.copy())
            
            # Handle weight transformations
            pytorch_tensor = self._transform_weight_shape(pytorch_tensor, tf_name, pytorch_name)
            
            return pytorch_tensor
            
        except Exception as e:
            print(f"Error converting {tf_name}: {e}")
            return None
            
    def _transform_weight_shape(self,
                              tensor: torch.Tensor,
                              tf_name: str,
                              pytorch_name: str) -> torch.Tensor:
        """Transform weight shapes between TensorFlow and PyTorch conventions"""
        
        # Convolutional layers: TF uses (H, W, in_channels, out_channels), PyTorch uses (out_channels, in_channels, H, W)
        if 'conv' in pytorch_name.lower() and 'weight' in pytorch_name and tensor.dim() == 4:
            tensor = tensor.permute(3, 2, 0, 1)
            
        # Dense/Linear layers: TF uses (in_features, out_features), PyTorch uses (out_features, in_features)
        elif ('dense' in pytorch_name.lower() or 'linear' in pytorch_name.lower()) and 'weight' in pytorch_name and tensor.dim() == 2:
            tensor = tensor.transpose(0, 1)
            
        # Attention weights may need special handling
        elif 'attention' in pytorch_name and 'weight' in pytorch_name:
            # Handle multi-head attention weight reshaping if needed
            pass
            
        return tensor
        
    def _validate_weights(self, pytorch_weights: OrderedDict) -> Dict[str, Any]:
        """Validate converted weights"""
        
        validation = {
            'total_params': len(pytorch_weights),
            'weight_tensors': 0,
            'bias_tensors': 0,
            'norm_tensors': 0,
            'shape_issues': [],
            'dtype_issues': []
        }
        
        for name, tensor in pytorch_weights.items():
            if 'weight' in name:
                validation['weight_tensors'] += 1
            elif 'bias' in name:
                validation['bias_tensors'] += 1
            elif 'norm' in name:
                validation['norm_tensors'] += 1
                
            # Check for shape issues
            if tensor.numel() == 0:
                validation['shape_issues'].append(f"{name}: empty tensor")
            elif tensor.dim() > 6:
                validation['shape_issues'].append(f"{name}: unusual dimension {tensor.dim()}")
                
            # Check dtype
            if tensor.dtype not in [torch.float32, torch.float16, torch.int32, torch.int64]:
                validation['dtype_issues'].append(f"{name}: unusual dtype {tensor.dtype}")
                
        return validation
        
    def _save_pytorch_checkpoint(self, weights: OrderedDict, output_path: str):
        """Save converted weights as PyTorch checkpoint"""
        
        checkpoint = {
            'state_dict': weights,
            'model_type': self.model_type,
            'conversion_info': {
                'source': 'tensorflow_checkpoint',
                'converter_version': '1.0',
                'total_params': len(weights)
            }
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        torch.save(checkpoint, output_path)
        print(f"Saved PyTorch checkpoint: {output_path}")
        
        # Also save human-readable summary
        summary_path = output_path.replace('.pth', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"EpiBERT PyTorch Checkpoint Summary\n")
            f.write(f"Model type: {self.model_type}\n")
            f.write(f"Total parameters: {len(weights)}\n")
            f.write(f"Converted from TensorFlow checkpoint\n\n")
            f.write("Parameter names and shapes:\n")
            for name, tensor in weights.items():
                f.write(f"  {name}: {tuple(tensor.shape)}\n")
                
        print(f"Saved summary: {summary_path}")


def load_tensorflow_weights_to_pytorch(model: nn.Module,
                                      tf_checkpoint_path: str,
                                      model_type: str = "pretraining",
                                      strict: bool = False) -> Dict[str, Any]:
    """
    Load TensorFlow weights directly into a PyTorch model
    
    Args:
        model: PyTorch model to load weights into
        tf_checkpoint_path: Path to TensorFlow checkpoint
        model_type: Model type ("pretraining" or "finetuning")
        strict: Whether to require exact parameter matching
        
    Returns:
        Loading statistics and any issues
    """
    
    # Convert checkpoint to PyTorch format
    converter = TensorFlowToPyTorchConverter(model_type=model_type)
    
    # Create temporary file for converted weights
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
        temp_path = tmp_file.name
        
    try:
        # Convert weights
        stats = converter.convert_checkpoint(tf_checkpoint_path, temp_path)
        
        # Load converted weights
        checkpoint = torch.load(temp_path, map_location='cpu')
        converted_weights = checkpoint['state_dict']
        
        # Load into model
        loading_info = model.load_state_dict(converted_weights, strict=strict)
        
        stats.update({
            'missing_keys': loading_info.missing_keys if hasattr(loading_info, 'missing_keys') else [],
            'unexpected_keys': loading_info.unexpected_keys if hasattr(loading_info, 'unexpected_keys') else [],
            'loaded_successfully': True
        })
        
        return stats
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def create_weight_mapping_guide(model_type: str = "pretraining") -> str:
    """
    Create a guide for manual weight mapping
    
    Args:
        model_type: Model type to create guide for
        
    Returns:
        String with mapping guide
    """
    
    converter = TensorFlowToPyTorchConverter(model_type)
    
    guide = f"""
EpiBERT Weight Mapping Guide - {model_type.title()} Model

This guide shows how TensorFlow parameter names map to PyTorch equivalents:

Layer Name Mappings:
"""
    
    for tf_pattern, pt_pattern in converter.layer_mappings.items():
        guide += f"  '{tf_pattern}' -> '{pt_pattern}'\n"
        
    guide += """
Weight Tensor Transformations:
  - Conv2D weights: (H, W, in_ch, out_ch) -> (out_ch, in_ch, H, W)  
  - Dense weights: (in_feat, out_feat) -> (out_feat, in_feat)
  - LayerNorm: gamma -> weight, beta -> bias
  - Bias tensors: no transformation needed

Usage Examples:
  # Convert checkpoint file
  converter = TensorFlowToPyTorchConverter("pretraining")
  converter.convert_checkpoint("tf_model.ckpt", "pytorch_model.pth")
  
  # Load directly into model
  stats = load_tensorflow_weights_to_pytorch(model, "tf_model.ckpt")
"""
    
    return guide


def main():
    """Command-line interface for weight conversion"""
    
    parser = argparse.ArgumentParser(description="Convert EpiBERT TensorFlow checkpoints to PyTorch")
    parser.add_argument("tf_checkpoint", help="Path to TensorFlow checkpoint")
    parser.add_argument("output_path", help="Output path for PyTorch checkpoint")
    parser.add_argument("--model-type", choices=["pretraining", "finetuning"], 
                       default="pretraining", help="Model type")
    parser.add_argument("--no-validate", action="store_true", 
                       help="Skip weight validation")
    parser.add_argument("--create-guide", action="store_true",
                       help="Create weight mapping guide")
    
    args = parser.parse_args()
    
    if args.create_guide:
        guide = create_weight_mapping_guide(args.model_type)
        guide_path = f"weight_mapping_guide_{args.model_type}.txt"
        with open(guide_path, 'w') as f:
            f.write(guide)
        print(f"Created mapping guide: {guide_path}")
        return
    
    # Perform conversion
    converter = TensorFlowToPyTorchConverter(args.model_type)
    
    try:
        stats = converter.convert_checkpoint(
            args.tf_checkpoint,
            args.output_path,
            validate=not args.no_validate
        )
        print("Conversion successful!")
        print(f"Statistics: {stats}")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())