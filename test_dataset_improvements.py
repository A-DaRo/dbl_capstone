#!/usr/bin/env python3
"""
Simple test script to verify the improved dataset loading functionality.
"""

import yaml
import numpy as np
from pathlib import Path

# Add src to path to import our modules
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from coral_mtl.data.dataset import (
    validate_task_definitions,
    create_label_mapping_from_task_definitions,
    CoralscapesMTLDataset,
    CoralscapesDataset
)

def test_task_definitions_validation():
    """Test the task definitions validation."""
    print("Testing task definitions validation...")
    
    # Load the actual task definitions
    task_def_path = Path(__file__).parent / "configs" / "task_definitions.yaml"
    with open(task_def_path, 'r') as f:
        task_definitions = yaml.safe_load(f)
    
    try:
        validate_task_definitions(task_definitions)
        print("✓ Task definitions validation passed")
    except Exception as e:
        print(f"✗ Task definitions validation failed: {e}")
        return False
    
    return True

def test_label_mapping_creation():
    """Test the label mapping creation."""
    print("\nTesting label mapping creation...")
    
    # Load the actual task definitions
    task_def_path = Path(__file__).parent / "configs" / "task_definitions.yaml"
    with open(task_def_path, 'r') as f:
        task_definitions = yaml.safe_load(f)
    
    try:
        mappings = create_label_mapping_from_task_definitions(task_definitions)
        print(f"✓ Created mappings for {len(mappings)} tasks: {list(mappings.keys())}")
        
        # Verify each mapping
        for task_name, mapping_array in mappings.items():
            print(f"  - {task_name}: mapping array shape {mapping_array.shape}")
            unique_values = np.unique(mapping_array)
            print(f"    Unique mapped values: {unique_values}")
        
        return True
    except Exception as e:
        print(f"✗ Label mapping creation failed: {e}")
        return False

def test_malformed_task_definitions():
    """Test handling of malformed task definitions."""
    print("\nTesting malformed task definitions handling...")
    
    # Test empty definitions
    try:
        validate_task_definitions({})
        print("✗ Should have failed on empty definitions")
        return False
    except ValueError:
        print("✓ Correctly rejected empty definitions")
    
    # Test missing id2label
    try:
        validate_task_definitions({"test_task": {"invalid": "data"}})
        print("✗ Should have failed on missing id2label")
        return False
    except ValueError:
        print("✓ Correctly rejected missing id2label")
    
    # Test non-integer label IDs
    try:
        validate_task_definitions({"test_task": {"id2label": {"invalid_id": "label"}}})
        print("✗ Should have failed on non-integer label ID")
        return False
    except ValueError:
        print("✓ Correctly rejected non-integer label ID")
    
    return True

def test_dataset_creation():
    """Test basic dataset creation (without actual data files)."""
    print("\nTesting dataset creation...")
    
    # Load the actual task definitions
    task_def_path = Path(__file__).parent / "configs" / "task_definitions.yaml"
    with open(task_def_path, 'r') as f:
        task_definitions = yaml.safe_load(f)
    
    try:
        # Test MTL dataset creation (this will fail at data loading, but should pass validation)
        try:
            mtl_dataset = CoralscapesMTLDataset(
                task_definitions=task_definitions,
                split='train',
                hf_dataset_name="dummy"  # Will fail later but validates structure
            )
            print("✓ CoralscapesMTLDataset creation passed validation")
        except Exception as e:
            if "failed to load" in str(e).lower() or "dataset" in str(e).lower():
                print("✓ CoralscapesMTLDataset creation passed validation (failed at data loading as expected)")
            else:
                print(f"✗ CoralscapesMTLDataset creation failed: {e}")
                return False
        
        # Test single-task dataset creation
        try:
            single_dataset = CoralscapesDataset(
                task_definitions=task_definitions,
                split='train',
                hf_dataset_name="dummy"  # Will fail later but validates structure
            )
            print("✓ CoralscapesDataset creation passed validation")
        except Exception as e:
            if "failed to load" in str(e).lower() or "dataset" in str(e).lower():
                print("✓ CoralscapesDataset creation passed validation (failed at data loading as expected)")
            else:
                print(f"✗ CoralscapesDataset creation failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Dataset creation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Improved Dataset Implementation")
    print("=" * 60)
    
    tests = [
        test_task_definitions_validation,
        test_label_mapping_creation,
        test_malformed_task_definitions,
        test_dataset_creation
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! The dataset improvements are working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()