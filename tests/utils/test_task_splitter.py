import pytest
import numpy as np
import torch
from coral_mtl.utils.task_splitter import TaskSplitter, MTLTaskSplitter, BaseTaskSplitter


@pytest.fixture
def simple_task_definitions():
    """Simple task definitions for basic testing."""
    return {
        'task_a': {
            'id2label': {0: 'background', 1: 'class_1', 2: 'class_2'}
        },
        'task_b': {
            'id2label': {0: 'background', 3: 'class_3', 4: 'class_4'}
        }
    }


@pytest.fixture
def complex_task_definitions():
    """Complex task definitions with groupby for advanced testing."""
    return {
        'genus': {
            'id2label': {
                0: 'background', 1: 'acropora_1', 2: 'acropora_2', 
                3: 'pocillopora', 4: 'porites'
            },
            'groupby': {
                'id2label': {0: 'background', 1: 'acropora', 2: 'pocillopora', 3: 'porites'},
                'mapping': {0: [0], 1: [1, 2], 2: [3], 3: [4]}
            }
        },
        'health': {
            'id2label': {0: 'background', 5: 'healthy', 6: 'bleached'},
            'groupby': {
                'id2label': {0: 'background', 1: 'alive', 2: 'bleached'},
                'mapping': {0: [0], 1: [5], 2: [6]}
            }
        }
    }


class TestTaskSplitterBase:
    """Tests the base TaskSplitter functionality."""

    def test_init_with_valid_definitions(self, simple_task_definitions):
        """Test initialization with valid task definitions."""
        splitter = MTLTaskSplitter(simple_task_definitions)
        assert splitter.raw_definitions == simple_task_definitions
        assert hasattr(splitter, 'hierarchical_definitions')
        assert hasattr(splitter, 'global_mapping_array')
        assert hasattr(splitter, 'global_id2label')

    def test_init_with_invalid_definitions(self):
        """Test initialization with invalid task definitions."""
        with pytest.raises(ValueError, match="Task definitions must be a non-empty dictionary"):
            MTLTaskSplitter({})
        
        with pytest.raises(ValueError, match="Task definitions must be a non-empty dictionary"):
            MTLTaskSplitter(None)
        
        # Missing required fields
        invalid_definitions = {'task_a': {'invalid_field': {}}}
        with pytest.raises(ValueError, match="missing required 'id2label' field"):
            MTLTaskSplitter(invalid_definitions)

    def test_max_original_id_calculation(self, simple_task_definitions):
        """Test calculation of maximum original ID."""
        splitter = MTLTaskSplitter(simple_task_definitions)
        # Max ID should be 4 (from task_b)
        assert splitter.max_original_id == 4

    def test_global_space_creation(self, simple_task_definitions):
        """Test creation of global non-overlapping label space."""
        splitter = MTLTaskSplitter(simple_task_definitions)
        
        # Check global mapping properties
        assert isinstance(splitter.global_mapping_array, np.ndarray)
        assert isinstance(splitter.global_id2label, dict)
        assert 0 in splitter.global_id2label  # Background should always be 0
        assert splitter.global_id2label[0] == 'background'
        
        # Check that all non-background classes are mapped
        expected_classes = {'class_1', 'class_2', 'class_3', 'class_4'}
        global_classes = set(splitter.global_id2label.values()) - {'background'}
        assert global_classes == expected_classes

    def test_hierarchical_definitions_simple(self, simple_task_definitions):
        """Test hierarchical definitions parsing for simple tasks."""
        splitter = MTLTaskSplitter(simple_task_definitions)
        
        for task_name in ['task_a', 'task_b']:
            task_def = splitter.hierarchical_definitions[task_name]
            
            # Should have ungrouped section
            assert 'ungrouped' in task_def
            assert 'is_grouped' in task_def
            assert task_def['is_grouped'] == False
            
            # Check ungrouped structure
            ungrouped = task_def['ungrouped']
            assert 'id2label' in ungrouped
            assert 'class_names' in ungrouped
            assert 'mapping_array' in ungrouped

    def test_hierarchical_definitions_complex(self, complex_task_definitions):
        """Test hierarchical definitions parsing for grouped tasks."""
        splitter = MTLTaskSplitter(complex_task_definitions)
        
        for task_name in ['genus', 'health']:
            task_def = splitter.hierarchical_definitions[task_name]
            
            # Should have both ungrouped and grouped sections
            assert 'ungrouped' in task_def
            assert 'grouped' in task_def
            assert task_def['is_grouped'] == True
            
            # Check grouped structure
            grouped = task_def['grouped']
            assert 'id2label' in grouped
            assert 'class_names' in grouped
            
            # Check ungrouped to grouped mapping
            assert 'ungrouped_to_grouped_map' in task_def
            assert isinstance(task_def['ungrouped_to_grouped_map'], np.ndarray)


class TestMTLTaskSplitter:
    """Tests specific to MTLTaskSplitter."""

    def test_inheritance(self, simple_task_definitions):
        """Test that MTLTaskSplitter properly inherits from TaskSplitter."""
        splitter = MTLTaskSplitter(simple_task_definitions)
        assert isinstance(splitter, TaskSplitter)
        
        # Should have all the base functionality
        assert hasattr(splitter, 'hierarchical_definitions')
        assert hasattr(splitter, 'global_mapping_array')
        assert hasattr(splitter, 'global_id2label')

    def test_mtl_specific_functionality(self, complex_task_definitions):
        """Test MTL-specific functionality."""
        splitter = MTLTaskSplitter(complex_task_definitions)
        
        # Should preserve all task-specific information for MTL training
        assert len(splitter.hierarchical_definitions) == 2
        
        # Each task should have proper ungrouped definitions for multi-head outputs
        for task_name, task_def in splitter.hierarchical_definitions.items():
            ungrouped = task_def['ungrouped']
            assert len(ungrouped['id2label']) > 0
            assert len(ungrouped['class_names']) == len(ungrouped['id2label'])


class TestBaseTaskSplitter:
    """Tests specific to BaseTaskSplitter."""

    def test_inheritance_and_additional_attributes(self, simple_task_definitions):
        """Test BaseTaskSplitter inheritance and additional attributes."""
        splitter = BaseTaskSplitter(simple_task_definitions)
        assert isinstance(splitter, TaskSplitter)
        
        # Should have additional flattened space attributes
        assert hasattr(splitter, 'flat_mapping_array')
        assert hasattr(splitter, 'flat_id2label')
        assert hasattr(splitter, 'flat_to_original_mapping_array')
        assert hasattr(splitter, 'flat_to_original_mapping_torch')

    def test_flattened_space_creation(self, simple_task_definitions):
        """Test creation of flattened label space for baseline models."""
        splitter = BaseTaskSplitter(simple_task_definitions)
        
        # Flattened space should be identical to global space
        assert np.array_equal(splitter.flat_mapping_array, splitter.global_mapping_array)
        assert splitter.flat_id2label == splitter.global_id2label

    def test_inverse_mapping(self, simple_task_definitions):
        """Test creation of inverse mapping from flat to original space."""
        splitter = BaseTaskSplitter(simple_task_definitions)
        
        # Test round-trip mapping
        original_ids = [0, 1, 2, 3, 4]  # All original IDs in test data
        for orig_id in original_ids:
            if orig_id <= splitter.max_original_id:
                flat_id = splitter.flat_mapping_array[orig_id]
                if flat_id > 0:  # Skip background/unmapped
                    recovered_orig_id = splitter.flat_to_original_mapping_array[flat_id]
                    # The recovered ID should be an original ID that maps to this flat ID
                    assert splitter.flat_mapping_array[recovered_orig_id] == flat_id

    def test_torch_tensor_conversion(self, simple_task_definitions):
        """Test that torch tensor versions are created properly."""
        splitter = BaseTaskSplitter(simple_task_definitions)
        
        # Check torch tensor attributes
        assert isinstance(splitter.global_mapping_torch, torch.Tensor)
        assert isinstance(splitter.flat_to_original_mapping_torch, torch.Tensor)
        
        # Check that they match numpy versions
        assert torch.equal(splitter.global_mapping_torch, 
                          torch.from_numpy(splitter.global_mapping_array))
        assert torch.equal(splitter.flat_to_original_mapping_torch, 
                          torch.from_numpy(splitter.flat_to_original_mapping_array))


class TestTaskSplitterEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_id2label(self):
        """Test handling of empty id2label."""
        invalid_def = {'task': {'id2label': {}}}
        # This should still work but create minimal structures
        splitter = MTLTaskSplitter(invalid_def)
        assert len(splitter.global_id2label) == 1  # Only background
        assert splitter.global_id2label[0] == 'background'

    def test_large_id_values(self):
        """Test handling of large ID values."""
        large_id_def = {
            'task': {'id2label': {0: 'background', 1000: 'large_class'}}
        }
        splitter = MTLTaskSplitter(large_id_def)
        assert splitter.max_original_id == 1000
        assert len(splitter.global_mapping_array) == 1001  # 0 to 1000 inclusive

    def test_inconsistent_groupby_mapping(self):
        """Test handling of inconsistent groupby mappings."""
        inconsistent_def = {
            'task': {
                'id2label': {0: 'background', 1: 'class1'},
                'groupby': {
                    'id2label': {0: 'background', 1: 'group1'},
                    'mapping': {1: [999]}  # Maps to non-existent original ID
                }
            }
        }
        # Should not crash, but mapping for non-existent ID should be ignored
        splitter = MTLTaskSplitter(inconsistent_def)
        task_def = splitter.hierarchical_definitions['task']
        # The mapping array should handle this gracefully
        assert isinstance(task_def['ungrouped_to_grouped_map'], np.ndarray)