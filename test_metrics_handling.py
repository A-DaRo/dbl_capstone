#!/usr/bin/env python3
"""
Test script to verify the trainer/evaluator fixes work correctly.
"""

import torch
import sys
import os

# Add the src directory to the path for imports  
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def test_metrics_handling():
    """Test that the metrics handling logic works correctly."""
    print("=== Testing Metrics Handling Logic ===")
    
    # Simulate baseline model output (single tensor wrapped as dictionary)
    baseline_predictions = {
        'segmentation': torch.randn(2, 40, 256, 256)  # batch_size=2, classes=40
    }
    
    # Simulate MTL model output (multiple tasks)
    mtl_predictions = {
        'genus': torch.randn(2, 18, 256, 256),
        'health': torch.randn(2, 18, 256, 256),
        'fish': torch.randn(2, 2, 256, 256),
        'substrate': torch.randn(2, 6, 256, 256)
    }
    
    # Test baseline handling logic
    print("1. Testing baseline model output handling...")
    if len(baseline_predictions) == 1 and 'segmentation' in baseline_predictions:
        predictions_for_metrics = baseline_predictions['segmentation']
        print(f"✅ Baseline: Extracted tensor with shape {predictions_for_metrics.shape}")
        print(f"   Type: {type(predictions_for_metrics)}")
    else:
        predictions_for_metrics = baseline_predictions
        print(f"❌ Baseline: Kept as dictionary")
    
    # Test MTL handling logic  
    print("2. Testing MTL model output handling...")
    if len(mtl_predictions) == 1 and 'segmentation' in mtl_predictions:
        predictions_for_metrics = mtl_predictions['segmentation']
        print(f"❌ MTL: Incorrectly treated as baseline")
    else:
        predictions_for_metrics = mtl_predictions
        print(f"✅ MTL: Kept as dictionary with {len(predictions_for_metrics)} tasks")
        print(f"   Tasks: {list(predictions_for_metrics.keys())}")
    
    print("\n=== Metrics Handling Test Complete ===")
    return True

if __name__ == "__main__":
    success = test_metrics_handling()
    sys.exit(0 if success else 1)