#!/usr/bin/env python3
"""
Debug script to test what baseline and MTL models actually return.
"""

import torch
import sys
import os

# Add the src directory to the path for imports  
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from coral_mtl.ExperimentFactory import ExperimentFactory

def test_model_outputs():
    """Test what baseline and MTL models actually return."""
    print("=== Model Output Debug Test ===")
    
    # Test baseline model
    print("\n1. Testing BASELINE model...")
    baseline_config = "configs/baseline_comparisons/baseline_config.yaml"
    
    if not os.path.exists(baseline_config):
        print(f"Config not found: {baseline_config}")
        return False
    
    try:
        baseline_factory = ExperimentFactory(baseline_config)
        baseline_model = baseline_factory.get_model()
        
        print(f"Baseline model type: {type(baseline_model)}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 256)
        print(f"Input shape: {dummy_input.shape}")
        
        # Test forward pass
        with torch.no_grad():
            baseline_output = baseline_model(dummy_input)
            
        print(f"Baseline output type: {type(baseline_output)}")
        if isinstance(baseline_output, torch.Tensor):
            print(f"✅ Baseline returns TENSOR with shape: {baseline_output.shape}")
        elif isinstance(baseline_output, dict):
            print(f"❌ Baseline returns DICT with keys: {list(baseline_output.keys())}")
            for key, tensor in baseline_output.items():
                print(f"   - {key}: {tensor.shape}")
        else:
            print(f"❌ Baseline returns unknown type: {type(baseline_output)}")
            
        # Test what metrics calculator is selected
        baseline_metrics = baseline_factory.get_metrics_calculator()
        print(f"Baseline metrics calculator: {type(baseline_metrics)}")
        
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")
        return False
    
    # Test MTL model
    print("\n2. Testing MTL model...")
    mtl_config = "configs/baseline_comparisons/mtl_config.yaml"
    
    if not os.path.exists(mtl_config):
        print(f"Config not found: {mtl_config}")
        return False
        
    try:
        mtl_factory = ExperimentFactory(mtl_config)
        mtl_model = mtl_factory.get_model()
        
        print(f"MTL model type: {type(mtl_model)}")
        
        # Test forward pass
        with torch.no_grad():
            mtl_output = mtl_model(dummy_input)
            
        print(f"MTL output type: {type(mtl_output)}")
        if isinstance(mtl_output, dict):
            print(f"✅ MTL returns DICT with keys: {list(mtl_output.keys())}")
            for key, tensor in mtl_output.items():
                print(f"   - {key}: {tensor.shape}")
        elif isinstance(mtl_output, torch.Tensor):
            print(f"❌ MTL returns TENSOR with shape: {mtl_output.shape}")
        else:
            print(f"❌ MTL returns unknown type: {type(mtl_output)}")
            
        # Test what metrics calculator is selected
        mtl_metrics = mtl_factory.get_metrics_calculator()
        print(f"MTL metrics calculator: {type(mtl_metrics)}")
        
        # Check if 'background' task exists
        task_definitions = mtl_factory.task_splitter.hierarchical_definitions
        print(f"Available MTL tasks: {list(task_definitions.keys())}")
        if 'background' in task_definitions:
            print(f"✅ 'background' task found in definitions")
        else:
            print(f"❌ 'background' task NOT found in definitions")
        
    except Exception as e:
        print(f"❌ MTL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== Debug Test Complete ===")
    return True

if __name__ == "__main__":
    success = test_model_outputs()
    sys.exit(0 if success else 1)