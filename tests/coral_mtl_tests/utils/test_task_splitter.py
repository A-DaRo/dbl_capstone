# Edit file: tests/coral_mtl/utils/task_splitter/test_task_splitter.py
"""
Robust tests for TaskSplitter variants using parametrized configurations.

This test module validates the core logic of TaskSplitter, MTLTaskSplitter,
and BaseTaskSplitter against a variety of complex and edge-case task
definition configurations provided by fixtures from conftest.py.

Tests cover:
- Successful instantiation across all variants.
- Correct parsing of hierarchical and grouped structures.
- Correct creation of the unified global label space.
- Logical consistency of generated mapping arrays.
- Round-trip transformation integrity for the BaseTaskSplitter.
"""
import pytest
import numpy as np
from coral_mtl.utils.task_splitter import TaskSplitter, MTLTaskSplitter, BaseTaskSplitter


# --- Test Group 1: Instantiation, Validation, and Error Handling ---

def test_splitter_instantiation_with_variants(task_definitions):
    """
    Verify that both splitter types instantiate successfully with every
    parametrized task definition variant (default, extreme, all_in_one, etc.).
    This is a primary smoke test for configuration parsing.
    """
    # Test MTLTaskSplitter
    mtl_splitter = MTLTaskSplitter(task_definitions)
    assert isinstance(mtl_splitter, TaskSplitter)
    assert len(mtl_splitter.hierarchical_definitions) > 0
    assert mtl_splitter.num_global_classes > 1

    # Test BaseTaskSplitter
    base_splitter = BaseTaskSplitter(task_definitions)
    assert isinstance(base_splitter, BaseTaskSplitter)
    assert hasattr(base_splitter, 'flat_mapping_array')
    assert hasattr(base_splitter, 'flat_to_original_mapping_array')
    assert len(base_splitter.hierarchical_definitions) > 0


@pytest.mark.parametrize("invalid_input, error_msg", [
    (None, "Task definitions must be a non-empty dictionary."),
    ({}, "Task definitions must be a non-empty dictionary."),
    ({'task_a': 'not_a_dict'}, "Task 'task_a' info must be a dictionary."),
    ({'task_a': {'wrong_key': 'value'}}, "Task 'task_a' missing required 'id2label' field."),
    ({'task_a': {'id2label': {}, 'groupby': {'missing_keys': True}}}, "Task 'task_a' has incomplete 'groupby' section.")
])
def test_task_splitter_invalid_input_raises_value_error(invalid_input, error_msg):
    """Test that TaskSplitter raises ValueError for malformed inputs."""
    with pytest.raises(ValueError, match=error_msg):
        MTLTaskSplitter(invalid_input)


# --- Test Group 2: Core `TaskSplitter` Property Correctness (using fixtures) ---

def test_splitter_finds_max_original_id(task_definitions):
    """Verify _find_max_original_id is correct for all variants."""
    splitter = MTLTaskSplitter(task_definitions)
    
    # Calculate expected max_id dynamically from the input fixture
    expected_max_id = 0
    for task_info in task_definitions.values():
        keys = [int(k) for k in task_info['id2label'].keys()]
        if keys:
            expected_max_id = max(expected_max_id, max(keys))
            
    assert splitter.max_original_id == expected_max_id


def test_splitter_parses_hierarchical_definitions_structurally(task_definitions):
    """
    Verify the structural integrity of `hierarchical_definitions` for all variants.
    This test checks for logical consistency rather than specific values.
    """
    splitter = MTLTaskSplitter(task_definitions)
    
    for task_name, task_data in splitter.hierarchical_definitions.items():
        assert 'is_grouped' in task_data
        assert 'ungrouped' in task_data
        
        ungrouped_info = task_data['ungrouped']
        assert 'id2label' in ungrouped_info
        assert 'class_names' in ungrouped_info
        assert 'mapping_array' in ungrouped_info
        assert isinstance(ungrouped_info['mapping_array'], np.ndarray)
        assert len(ungrouped_info['id2label']) == len(ungrouped_info['class_names'])

        if task_data['is_grouped']:
            assert 'grouped' in task_data
            grouped_info = task_data['grouped']
            assert 'id2label' in grouped_info
            assert 'class_names' in grouped_info
            assert 'mapping_array' in grouped_info
            assert isinstance(grouped_info['mapping_array'], np.ndarray)
            assert len(grouped_info['id2label']) == len(grouped_info['class_names'])
            
            # Add assertion for grouped mapping_array LUT semantics
            assert len(grouped_info['mapping_array']) == splitter.max_original_id + 1, \
                f"Task '{task_name}' grouped mapping_array should have length max_original_id+1 for LUT semantics"
            
            assert 'ungrouped_to_grouped_map' in task_data
            group_map = task_data['ungrouped_to_grouped_map']
            assert isinstance(group_map, np.ndarray)
            assert len(group_map) == len(ungrouped_info['id2label'])
            if len(group_map) > 0:
                 assert group_map.max() < len(grouped_info['id2label'])


def test_splitter_creates_valid_global_space(task_definitions):
    """Verify the properties of the unified global label space for all variants."""
    splitter = MTLTaskSplitter(task_definitions)
    
    assert splitter.num_global_classes == len(splitter.global_id2label)
    assert splitter.global_mapping_array.shape == (splitter.max_original_id + 1,)
    assert splitter.global_id2label[0] == "background"
    
    # Check for consistency: every original ID mentioned in any task must
    # map to a valid global ID.
    all_original_ids = set()
    for task_info in task_definitions.values():
        all_original_ids.update(int(k) for k in task_info['id2label'].keys())

    for original_id in all_original_ids:
        if original_id != 0:
            global_id = splitter.global_mapping_array[original_id]
            assert global_id > 0
            assert global_id in splitter.global_id2label


# --- Test Group 3: Transformation Logic ---

def test_splitter_mtl_mask_transformation(splitter_mtl: MTLTaskSplitter):
    """
    Verify that an original mask is correctly transformed into per-task masks
    using the mapping arrays from a fixture-provided splitter.
    This test is crucial for ensuring dynamic behavior.
    """
    # Dynamically select a few original IDs that exist in the provided task definitions
    # to make the test resilient to changes in test configs.
    sample_ids = []
    for task_info in splitter_mtl.raw_definitions.values():
        ids = [int(k) for k in task_info['id2label'].keys() if int(k) != 0]
        if ids:
            sample_ids.append(ids[0])
        if len(sample_ids) >= 4:
            break
    
    if not sample_ids:
        pytest.skip("No non-background class IDs found in task definitions to test transformation.")

    # Create a synthetic mask with these dynamically chosen original IDs
    h, w = 2, (len(sample_ids) + 1) // 2
    original_mask = np.array(sample_ids + [0] * (h * w - len(sample_ids)), dtype=np.int64).reshape(h, w)
    
    # Test ungrouped transformations for each defined task
    for task_name, task_data in splitter_mtl.hierarchical_definitions.items():
        mapping_array = task_data['ungrouped']['mapping_array']
        task_mask = mapping_array[original_mask]
        
        # An original ID should map to a non-zero value in the task mask IF AND ONLY IF
        # that original ID is part of the task's own id2label definition.
        task_original_ids = {int(k) for k in splitter_mtl.raw_definitions[task_name]['id2label'].keys()}
        
        for orig_id in sample_ids:
            is_in_task = orig_id in task_original_ids
            mask_val_at_id = task_mask[original_mask == orig_id][0]
            
            if is_in_task:
                assert mask_val_at_id > 0, f"ID {orig_id} should be in task '{task_name}' but was mapped to 0"
            else:
                assert mask_val_at_id == 0, f"ID {orig_id} should NOT be in task '{task_name}' but was mapped to {mask_val_at_id}"
        
        # Test grouped transformations when is_grouped is True
        if task_data.get('is_grouped', False):
            grouped_mapping_array = task_data['grouped']['mapping_array']
            grouped_task_mask = grouped_mapping_array[original_mask]
            
            # Same logic: original ID should map to non-zero in grouped space IF AND ONLY IF in task
            for orig_id in sample_ids:
                is_in_task = orig_id in task_original_ids
                grouped_mask_val_at_id = grouped_task_mask[original_mask == orig_id][0]
                
                if is_in_task:
                    assert grouped_mask_val_at_id > 0, f"ID {orig_id} should be in grouped task '{task_name}' but was mapped to 0"
                else:
                    assert grouped_mask_val_at_id == 0, f"ID {orig_id} should NOT be in grouped task '{task_name}' but was mapped to {grouped_mask_val_at_id}"


# --- Test Group 4: `BaseTaskSplitter` Specifics ---

def test_splitter_base_flat_space_is_global(splitter_base: BaseTaskSplitter):
    """Verify that the flattened training space is identical to the global space."""
    assert np.array_equal(splitter_base.flat_mapping_array, splitter_base.global_mapping_array)
    assert splitter_base.flat_id2label == splitter_base.global_id2label


def test_splitter_base_round_trip_transformation(splitter_base: BaseTaskSplitter):
    """
    CRITICAL: Verify that transforming a mask to the flat space and back again
    correctly reconstructs masks for original IDs that exist in task definitions.
    Unused original IDs should map to background (0) and remain background.
    """
    # Collect all original IDs that actually exist in the task definitions
    existing_original_ids = set()
    for task_info in splitter_base.raw_definitions.values():
        existing_original_ids.update(int(k) for k in task_info['id2label'].keys())
    
    # Create a mask containing only these existing original IDs
    existing_ids_list = sorted(list(existing_original_ids))
    original_mask = np.array(existing_ids_list, dtype=np.int64).reshape(1, -1)

    # --- Forward pass: original -> flat ---
    flat_mask = splitter_base.flat_mapping_array[original_mask]

    # --- Backward pass: flat -> original ---
    reconstructed_mask = splitter_base.flat_to_original_mapping_array[flat_mask]

    # --- Validation ---
    # The reconstruction must be perfect for existing original IDs
    assert np.array_equal(original_mask, reconstructed_mask), \
        f"Round-trip transformation failed. Original: {original_mask}, Reconstructed: {reconstructed_mask}"
    
    # Additional test: unused original IDs should map to background (0) and stay background
    if splitter_base.max_original_id > max(existing_original_ids):
        # Test with a non-existent original ID (gap in the ID space)
        unused_ids = []
        for i in range(splitter_base.max_original_id + 1):
            if i not in existing_original_ids:
                unused_ids.append(i)
        
        if unused_ids:
            unused_mask = np.array(unused_ids[:3], dtype=np.int64).reshape(1, -1)  # Test first few unused IDs
            flat_unused = splitter_base.flat_mapping_array[unused_mask]
            reconstructed_unused = splitter_base.flat_to_original_mapping_array[flat_unused]
            
            # Unused IDs should map to 0 (background) and stay as 0
            assert np.all(flat_unused == 0), f"Unused original IDs should map to background (0), got: {flat_unused}"
            assert np.all(reconstructed_unused == 0), f"Background should reconstruct to background (0), got: {reconstructed_unused}"