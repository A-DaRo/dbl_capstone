"""
Unit tests for TaskSplitter classes and hierarchical task mapping.
"""
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestTaskSplitterBasic(unittest.TestCase):
    """Test TaskSplitter basic functionality without external dependencies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dummy_task_definitions = {
            'genus': {
                'id2label': {0: 'unlabeled', 1: 'coral_a', 2: 'coral_b', 3: 'coral_c'},
                'groupby': {
                    'mapping': {0: 0, 1: 1, 2: 1, 3: 2},  # Group corals
                    'id2label': {0: 'unlabeled', 1: 'coral_group1', 2: 'coral_group2'}
                }
            },
            'health': {
                'id2label': {0: 'unlabeled', 1: 'healthy', 2: 'bleached', 3: 'dead'},
                'groupby': None  # No grouping for health
            },
            'morphology': {
                'id2label': {0: 'unlabeled', 1: 'branching', 2: 'massive'},
                'groupby': None
            }
        }
    
    def test_task_definitions_structure(self):
        """Test that task definitions have expected structure."""
        # Verify basic structure
        self.assertIn('genus', self.dummy_task_definitions)
        self.assertIn('health', self.dummy_task_definitions) 
        self.assertIn('morphology', self.dummy_task_definitions)
        
        # Verify each task has id2label
        for task_name, task_def in self.dummy_task_definitions.items():
            self.assertIn('id2label', task_def)
            self.assertIsInstance(task_def['id2label'], dict)
    
    def test_task_splitter_import(self):
        """Test that TaskSplitter classes can be imported."""
        try:
            from coral_mtl.utils.task_splitter import TaskSplitter, MTLTaskSplitter, BaseTaskSplitter
            # If we get here, imports work
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Could not import TaskSplitter classes: {e}")
    
    def test_mtl_task_splitter_basic_init(self):
        """Test MTLTaskSplitter basic initialization."""
        try:
            from coral_mtl.utils.task_splitter import MTLTaskSplitter
            splitter = MTLTaskSplitter(self.dummy_task_definitions)
            self.assertIsNotNone(splitter)
        except Exception as e:
            # If it fails due to missing dependencies, that's expected
            if "numpy" in str(e) or "torch" in str(e):
                self.skipTest(f"Skipping due to missing dependency: {e}")
            else:
                self.fail(f"Unexpected error: {e}")
    
    def test_base_task_splitter_basic_init(self):
        """Test BaseTaskSplitter basic initialization."""
        try:
            from coral_mtl.utils.task_splitter import BaseTaskSplitter
            splitter = BaseTaskSplitter(self.dummy_task_definitions)
            self.assertIsNotNone(splitter)
        except Exception as e:
            # If it fails due to missing dependencies, that's expected
            if "numpy" in str(e) or "torch" in str(e):
                self.skipTest(f"Skipping due to missing dependency: {e}")
            else:
                self.fail(f"Unexpected error: {e}")
    
    def test_invalid_task_definitions(self):
        """Test splitter handles invalid task definitions appropriately."""
        try:
            from coral_mtl.utils.task_splitter import MTLTaskSplitter
            
            invalid_definitions = [
                # Empty definitions
                {},
                # Missing required keys
                {'genus': {'id2label': {}}},
            ]
            
            for invalid_def in invalid_definitions:
                with self.assertRaises((KeyError, ValueError, AttributeError)):
                    MTLTaskSplitter(invalid_def)
        except ImportError as e:
            if "numpy" in str(e) or "torch" in str(e):
                self.skipTest(f"Skipping due to missing dependency: {e}")
            else:
                raise


if __name__ == '__main__':
    unittest.main()