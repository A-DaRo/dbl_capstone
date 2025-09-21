"""Shared fixtures for the test suite."""
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock
from coral_mtl.utils.task_splitter import MTLTaskSplitter, BaseTaskSplitter


@pytest.fixture
def real_task_definitions():
    """Load the real task definitions from configs/task_definitions.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "task_definitions.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_mtl_task_splitter(real_task_definitions):
    """Create a properly mocked MTLTaskSplitter with real task definitions."""
    mock_splitter = MagicMock(spec=MTLTaskSplitter)
    
    # Initialize the real splitter to get proper attributes
    real_splitter = MTLTaskSplitter(real_task_definitions)
    
    # Copy important attributes from real splitter to mock
    mock_splitter.raw_definitions = real_splitter.raw_definitions
    mock_splitter.hierarchical_definitions = real_splitter.hierarchical_definitions
    mock_splitter.max_original_id = real_splitter.max_original_id
    mock_splitter.global_mapping = real_splitter.global_mapping
    mock_splitter.flat_mapping_array = real_splitter.flat_mapping_array
    mock_splitter.flat_id2label = real_splitter.flat_id2label
    
    return mock_splitter


@pytest.fixture
def mock_base_task_splitter(real_task_definitions):
    """Create a properly mocked BaseTaskSplitter with real task definitions."""
    mock_splitter = MagicMock(spec=BaseTaskSplitter)
    
    # Initialize the real splitter to get proper attributes
    real_splitter = BaseTaskSplitter(real_task_definitions)
    
    # Copy important attributes from real splitter to mock
    mock_splitter.raw_definitions = real_splitter.raw_definitions
    mock_splitter.hierarchical_definitions = real_splitter.hierarchical_definitions
    mock_splitter.max_original_id = real_splitter.max_original_id
    mock_splitter.flat_mapping_array = real_splitter.flat_mapping_array
    mock_splitter.flat_id2label = real_splitter.flat_id2label
    
    return mock_splitter