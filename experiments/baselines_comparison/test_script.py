#!/usr/bin/env python
"""
Simple test script to validate the train_val_test_script functionality.
"""

import sys
import os

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from train_val_test_script import BaselineComparison

def test_initialization():
    """Test that BaselineComparison can be initialized."""
    print("Testing BaselineComparison initialization...")
    comparison = BaselineComparison()
    print("SUCCESS: BaselineComparison initialized")
    print("Config paths found:")
    for name, path in comparison.config_paths.items():
        print(f"  - {name}: {path}")
        print(f"    Exists: {os.path.exists(path)}")

def test_config_loading():
    """Test that ExperimentFactory can load configurations."""
    print("\nTesting ExperimentFactory configuration loading...")
    
    try:
        from coral_mtl.ExperimentFactory import ExperimentFactory
        
        comparison = BaselineComparison()
        
        # Test baseline config
        print("Testing baseline config...")
        baseline_factory = ExperimentFactory(config_path=comparison.config_paths['baseline'])
        print("SUCCESS: Baseline config loaded")
        
        # Test MTL config  
        print("Testing MTL config...")
        mtl_factory = ExperimentFactory(config_path=comparison.config_paths['mtl'])
        print("SUCCESS: MTL config loaded")
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("="*50)
    print("BASELINE COMPARISON SCRIPT TEST")
    print("="*50)
    
    try:
        test_initialization()
        success = test_config_loading()
        
        if success:
            print("\n" + "="*50)
            print("ALL TESTS PASSED!")
            print("The train_val_test_script.py is ready to use.")
            print("="*50)
        else:
            print("\nSome tests failed. Check the configuration.")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()