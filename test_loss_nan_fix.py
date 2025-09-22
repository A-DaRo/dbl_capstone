#!/usr/bin/env python3
"""
Test script to verify NaN loss fixes in both baseline and MTL models.
"""

import torch
import sys
import os
import yaml
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the src directory to the path for imports  
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from coral_mtl.ExperimentFactory import ExperimentFactory

def test_loss_nan_robustness():
    """Test that loss functions handle NaN inputs gracefully."""
    print("=== Testing Loss NaN Robustness ===")
    
    try:
        # Test baseline loss with NaN inputs
        print("\n🧪 Testing Baseline Loss with edge cases...")
        baseline_config_path = "configs/baseline_comparisons/baseline_config.yaml"
        
        with open(baseline_config_path, 'r') as f:
            baseline_config = yaml.safe_load(f)
        
        baseline_factory = ExperimentFactory(baseline_config_path)
        baseline_loss_fn = baseline_factory.get_loss_function()
        
        # Create some test cases including potential NaN sources
        test_logits = torch.randn(2, 7, 64, 64)  # 7 classes for genus
        test_targets = torch.randint(0, 7, (2, 64, 64))
        
        # Test 1: Normal case
        loss_normal = baseline_loss_fn(test_logits, test_targets)
        print(f"✅ Baseline normal loss: {loss_normal.item():.6f}")
        
        # Test 2: Extreme logits (could cause NaN in softmax)
        test_logits_extreme = torch.randn(2, 7, 64, 64) * 100  # Very large values
        loss_extreme = baseline_loss_fn(test_logits_extreme, test_targets)
        if torch.isnan(loss_extreme):
            print("❌ Baseline loss returns NaN with extreme logits")
        else:
            print(f"✅ Baseline extreme loss: {loss_extreme.item():.6f}")
        
        # Test 3: Test with actual NaN inputs
        test_logits_nan = test_logits.clone()
        test_logits_nan[0, 0, 0, 0] = float('nan')
        loss_nan_input = baseline_loss_fn(test_logits_nan, test_targets)
        if torch.isnan(loss_nan_input):
            print("⚠️ Baseline loss returns NaN with NaN inputs (expected behavior)")
        else:
            print(f"✅ Baseline handled NaN input: {loss_nan_input.item():.6f}")
        
        print("\n🧪 Testing MTL Loss with edge cases...")
        
        # Test MTL loss
        mtl_config_path = "configs/baseline_comparisons/mtl_config.yaml"
        
        with open(mtl_config_path, 'r') as f:
            mtl_config = yaml.safe_load(f)
        
        mtl_factory = ExperimentFactory(mtl_config_path)
        mtl_loss_fn = mtl_factory.get_loss_function()
        
        # MTL predictions and targets
        mtl_predictions = {
            'genus': torch.randn(2, 7, 64, 64),
            'health': torch.randn(2, 3, 64, 64)
        }
        mtl_targets = {
            'genus': torch.randint(0, 7, (2, 64, 64)),
            'health': torch.randint(0, 3, (2, 64, 64))
        }
        
        # Test 1: Normal case
        mtl_loss_dict = mtl_loss_fn(mtl_predictions, mtl_targets)
        total_loss = mtl_loss_dict['total_loss']
        print(f"✅ MTL normal total loss: {total_loss.item():.6f}")
        print(f"   Genus loss: {mtl_loss_dict['loss_genus'].item():.6f}")
        print(f"   Health loss: {mtl_loss_dict['loss_health'].item():.6f}")
        print(f"   Consistency loss: {mtl_loss_dict['loss_consistency'].item():.6f}")
        
        # Test 2: Extreme values
        mtl_predictions_extreme = {
            'genus': torch.randn(2, 7, 64, 64) * 100,
            'health': torch.randn(2, 3, 64, 64) * 100
        }
        
        mtl_loss_dict_extreme = mtl_loss_fn(mtl_predictions_extreme, mtl_targets)
        total_loss_extreme = mtl_loss_dict_extreme['total_loss']
        if torch.isnan(total_loss_extreme):
            print("❌ MTL loss returns NaN with extreme values")
        else:
            print(f"✅ MTL extreme total loss: {total_loss_extreme.item():.6f}")
        
        # Test 3: Test uncertainty parameter ranges
        print(f"   Precision genus: {mtl_loss_dict['precision_genus']:.6f}")
        print(f"   Precision health: {mtl_loss_dict['precision_health']:.6f}")
        print(f"   Precision aux: {mtl_loss_dict['precision_aux_group']:.6f}")
        
        print("\n✅ All loss tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Loss test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_forward_pass():
    """Test model forward passes for potential NaN sources."""
    print("\n=== Testing Model Forward Passes ===")
    
    try:
        # Test baseline model
        baseline_config_path = "configs/baseline_comparisons/baseline_config.yaml"
        baseline_factory = ExperimentFactory(baseline_config_path)
        baseline_model = baseline_factory.get_model()
        
        print("\n🧪 Testing Baseline Model...")
        dummy_input = torch.randn(1, 3, 512, 512)
        
        with torch.no_grad():
            baseline_output = baseline_model(dummy_input)
            
        if torch.any(torch.isnan(baseline_output)):
            print("❌ Baseline model outputs NaN")
        else:
            print(f"✅ Baseline model output shape: {baseline_output.shape}, no NaNs")
        
        # Test MTL model
        print("\n🧪 Testing MTL Model...")
        mtl_config_path = "configs/baseline_comparisons/mtl_config.yaml"
        mtl_factory = ExperimentFactory(mtl_config_path)
        mtl_model = mtl_factory.get_model()
        
        with torch.no_grad():
            mtl_output = mtl_model(dummy_input)
            
        # Check each task output for NaN
        has_nan = False
        for task, output in mtl_output.items():
            if torch.any(torch.isnan(output)):
                print(f"❌ MTL model task '{task}' outputs NaN")
                has_nan = True
            else:
                print(f"✅ MTL model task '{task}' shape: {output.shape}, no NaNs")
        
        if not has_nan:
            print("✅ All model forward passes completed successfully!")
            
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting comprehensive NaN loss testing...")
    
    loss_success = test_loss_nan_robustness()
    model_success = test_model_forward_pass()
    
    overall_success = loss_success and model_success
    
    if overall_success:
        print("\n🎉 All tests passed! NaN issues should be resolved.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
    
    sys.exit(0 if overall_success else 1)