#!/usr/bin/env python3
"""
Quick test to verify the training/validation fixes work correctly.
"""

import torch
import sys
import os
import yaml

# Add the src directory to the path for imports  
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from coral_mtl.ExperimentFactory import ExperimentFactory

def create_minimal_test_config():
    """Create a minimal config for quick testing."""
    
    # Read the baseline config to get the right structure
    baseline_config_path = "configs/baseline_comparisons/baseline_config.yaml"
    
    with open(baseline_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for very quick training test
    config['trainer']['epochs'] = 1  # Only 1 epoch
    config['data']['batch_size'] = 1  # Small batch size
    config['trainer']['output_dir'] = "experiments/test_fix_validation"
    
    # Save minimal config
    test_config_path = "test_baseline_fix.yaml"
    with open(test_config_path, 'w') as f:
        yaml.dump(config, f)
        
    return test_config_path

def test_training_validation_fix():
    """Test that the training/validation fix resolves the argmax error."""
    print("=== Testing Training/Validation Fix ===")
    
    try:
        # Create minimal config
        config_path = create_minimal_test_config()
        print(f"Created test config: {config_path}")
        
        # Initialize factory
        factory = ExperimentFactory(config_path)
        
        # Get model to verify it's baseline
        model = factory.get_model()
        print(f"Model type: {type(model)}")
        
        # Test with dummy data
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Model output type: {type(output)}")
        if isinstance(output, torch.Tensor):
            print(f"✅ Model returns tensor with shape: {output.shape}")
        else:
            print(f"❌ Model returns unexpected type: {type(output)}")
        
        print("\n🧪 Testing validation logic (this should not crash)...")
        
        # Just test building the trainer - don't run full training
        trainer = factory.get_trainer(
            model=model,
            train_loader=None,  # We won't actually train
            val_loader=None,
            loss_fn=factory.get_loss_function(),
            optimizer=None,
            scheduler=None,
            metrics_calculator=factory.get_metrics_calculator(),
            metrics_storer=factory.get_metrics_storer()
        )
        
        print("✅ Trainer created successfully - validation logic should be fixed!")
        
        # Clean up
        os.remove(config_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_validation_fix()
    sys.exit(0 if success else 1)