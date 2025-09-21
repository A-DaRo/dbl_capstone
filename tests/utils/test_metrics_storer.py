import pytest
import json
import os
import numpy as np
from pathlib import Path
from coral_mtl.utils.metrics_storer import MetricsStorer


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    return str(tmp_path / "test_metrics")


@pytest.fixture
def sample_metrics_report():
    """Create a sample metrics report for testing."""
    return {
        'optimization_metrics': {
            'loss': 0.5,
            'accuracy': 0.85,
            'nested': {
                'precision': 0.80,
                'recall': 0.75
            }
        },
        'other_data': {
            'learning_rate': 1e-4
        }
    }


@pytest.fixture
def sample_confusion_matrices():
    """Create sample confusion matrices for testing."""
    return {
        'genus': np.array([[100, 5, 2], [3, 80, 4], [1, 6, 90]], dtype=np.int64),
        'health': np.array([[150, 10], [8, 120]], dtype=np.int64)
    }


class TestMetricsStorerInit:
    """Test MetricsStorer initialization."""

    def test_init_creates_directory(self, temp_output_dir):
        """Test that initialization creates the output directory."""
        storer = MetricsStorer(temp_output_dir)
        assert os.path.exists(temp_output_dir)
        assert storer.output_dir == temp_output_dir

    def test_init_creates_paths(self, temp_output_dir):
        """Test that initialization sets up correct file paths."""
        storer = MetricsStorer(temp_output_dir)
        
        expected_history_path = os.path.join(temp_output_dir, "history.json")
        expected_val_cm_path = os.path.join(temp_output_dir, "validation_cms.jsonl")
        expected_test_cm_path = os.path.join(temp_output_dir, "test_cms.jsonl")
        
        assert storer.history_path == expected_history_path
        assert storer.val_cm_path == expected_val_cm_path
        assert storer.test_cm_path == expected_test_cm_path

    def test_init_state(self, temp_output_dir):
        """Test initial state of MetricsStorer."""
        storer = MetricsStorer(temp_output_dir)
        
        assert storer._history_data == {}
        assert storer._val_cm_file is None
        assert storer._test_cm_file is None


class TestMetricsStorerFlattenDict:
    """Test the _flatten_dict static method."""

    def test_flatten_simple_dict(self):
        """Test flattening of a simple dictionary."""
        simple_dict = {'a': 1, 'b': 2}
        flattened = MetricsStorer._flatten_dict(simple_dict)
        assert flattened == {'a': 1, 'b': 2}

    def test_flatten_nested_dict(self):
        """Test flattening of a nested dictionary."""
        nested_dict = {
            'level1': {
                'level2': {
                    'value': 42
                },
                'direct': 10
            },
            'root': 5
        }
        flattened = MetricsStorer._flatten_dict(nested_dict)
        expected = {
            'level1.level2.value': 42,
            'level1.direct': 10,
            'root': 5
        }
        assert flattened == expected

    def test_flatten_with_custom_separator(self):
        """Test flattening with custom separator."""
        nested_dict = {'a': {'b': {'c': 1}}}
        flattened = MetricsStorer._flatten_dict(nested_dict, sep='_')
        assert flattened == {'a_b_c': 1}

    def test_flatten_with_parent_key(self):
        """Test flattening with initial parent key."""
        nested_dict = {'b': {'c': 1}}
        flattened = MetricsStorer._flatten_dict(nested_dict, parent_key='a')
        assert flattened == {'a.b.c': 1}


class TestMetricsStorerEpochHistory:
    """Test epoch history storage functionality."""

    def test_store_first_epoch(self, temp_output_dir, sample_metrics_report):
        """Test storing the first epoch initializes the history correctly."""
        storer = MetricsStorer(temp_output_dir)
        storer.store_epoch_history(sample_metrics_report, epoch=1)
        
        # Check that history file was created
        assert os.path.exists(storer.history_path)
        
        # Check internal state
        assert 'epoch' in storer._history_data
        assert storer._history_data['epoch'] == [1]
        assert 'loss' in storer._history_data
        assert storer._history_data['loss'] == [0.5]
        assert 'accuracy' in storer._history_data
        assert storer._history_data['accuracy'] == [0.85]
        assert 'nested.precision' in storer._history_data
        assert storer._history_data['nested.precision'] == [0.80]

    def test_store_multiple_epochs(self, temp_output_dir, sample_metrics_report):
        """Test storing multiple epochs accumulates history correctly."""
        storer = MetricsStorer(temp_output_dir)
        
        # Store first epoch
        storer.store_epoch_history(sample_metrics_report, epoch=1)
        
        # Modify metrics for second epoch
        modified_report = sample_metrics_report.copy()
        modified_report['optimization_metrics']['loss'] = 0.3
        modified_report['optimization_metrics']['accuracy'] = 0.90
        
        # Store second epoch
        storer.store_epoch_history(modified_report, epoch=2)
        
        # Check accumulated history
        assert storer._history_data['epoch'] == [1, 2]
        assert storer._history_data['loss'] == [0.5, 0.3]
        assert storer._history_data['accuracy'] == [0.85, 0.90]

    def test_history_file_persistence(self, temp_output_dir, sample_metrics_report):
        """Test that history is persisted to file correctly."""
        storer = MetricsStorer(temp_output_dir)
        storer.store_epoch_history(sample_metrics_report, epoch=1)
        
        # Read the file directly
        with open(storer.history_path, 'r') as f:
            saved_data = json.load(f)
        
        # Check file contents
        assert 'epoch' in saved_data
        assert saved_data['epoch'] == [1]
        assert 'loss' in saved_data
        assert saved_data['loss'] == [0.5]

    def test_atomic_write(self, temp_output_dir, sample_metrics_report):
        """Test that history writing is atomic (uses temp file + rename)."""
        storer = MetricsStorer(temp_output_dir)
        storer.store_epoch_history(sample_metrics_report, epoch=1)
        
        # Check that temp file doesn't exist after writing
        temp_path = storer.history_path + ".tmp"
        assert not os.path.exists(temp_path)
        
        # Check that main file exists
        assert os.path.exists(storer.history_path)


class TestMetricsStorerPerImageCMs:
    """Test per-image confusion matrix storage functionality."""

    def test_open_close_validation_file(self, temp_output_dir):
        """Test opening and closing validation CM file."""
        storer = MetricsStorer(temp_output_dir)
        
        # Initially, no file should be open
        assert storer._val_cm_file is None
        
        # Open for validation run
        storer.open_for_run(is_testing=False)
        assert storer._val_cm_file is not None
        
        # Close
        storer.close()
        assert storer._val_cm_file is None

    def test_open_close_test_file(self, temp_output_dir):
        """Test opening and closing test CM file."""
        storer = MetricsStorer(temp_output_dir)
        
        # Open for test run
        storer.open_for_run(is_testing=True)
        assert storer._test_cm_file is not None
        
        # Close
        storer.close()
        assert storer._test_cm_file is None

    def test_file_clearing(self, temp_output_dir):
        """Test that opening a file clears previous content."""
        storer = MetricsStorer(temp_output_dir)
        
        # Create a file with some content
        with open(storer.val_cm_path, 'w') as f:
            f.write("old content\n")
        
        # Open for run should clear the file
        storer.open_for_run(is_testing=False)
        storer.close()
        
        # File should be empty now
        with open(storer.val_cm_path, 'r') as f:
            content = f.read()
        assert content == ""

    def test_store_validation_cms(self, temp_output_dir, sample_confusion_matrices):
        """Test storing validation confusion matrices."""
        storer = MetricsStorer(temp_output_dir)
        storer.open_for_run(is_testing=False)
        
        # Store CMs for an image
        storer.store_per_image_cms("image_001", sample_confusion_matrices, is_testing=False)
        
        storer.close()
        
        # Check that file was created and contains correct data
        assert os.path.exists(storer.val_cm_path)
        
        with open(storer.val_cm_path, 'r') as f:
            line = f.readline().strip()
            data = json.loads(line)
        
        assert data['image_id'] == "image_001"
        assert 'confusion_matrices' in data
        assert 'genus' in data['confusion_matrices']
        assert 'health' in data['confusion_matrices']
        
        # Check that numpy arrays were converted to lists
        assert isinstance(data['confusion_matrices']['genus'], list)
        assert len(data['confusion_matrices']['genus']) == 3  # 3x3 matrix

    def test_store_test_cms(self, temp_output_dir, sample_confusion_matrices):
        """Test storing test confusion matrices."""
        storer = MetricsStorer(temp_output_dir)
        storer.open_for_run(is_testing=True)
        
        # Store CMs for an image
        storer.store_per_image_cms("test_image_001", sample_confusion_matrices, is_testing=True)
        
        storer.close()
        
        # Check that test file was created
        assert os.path.exists(storer.test_cm_path)
        
        with open(storer.test_cm_path, 'r') as f:
            line = f.readline().strip()
            data = json.loads(line)
        
        assert data['image_id'] == "test_image_001"

    def test_store_multiple_images(self, temp_output_dir, sample_confusion_matrices):
        """Test storing CMs for multiple images."""
        storer = MetricsStorer(temp_output_dir)
        storer.open_for_run(is_testing=False)
        
        # Store CMs for multiple images
        for i in range(3):
            image_id = f"image_{i:03d}"
            storer.store_per_image_cms(image_id, sample_confusion_matrices, is_testing=False)
        
        storer.close()
        
        # Check that all images were stored
        with open(storer.val_cm_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3
        
        # Check each line
        for i, line in enumerate(lines):
            data = json.loads(line.strip())
            expected_id = f"image_{i:03d}"
            assert data['image_id'] == expected_id

    def test_error_without_open_file(self, temp_output_dir, sample_confusion_matrices):
        """Test that storing CMs without opening file raises error."""
        storer = MetricsStorer(temp_output_dir)
        
        with pytest.raises(IOError, match="File handle is not open"):
            storer.store_per_image_cms("image_001", sample_confusion_matrices, is_testing=False)

    def test_jsonl_format(self, temp_output_dir, sample_confusion_matrices):
        """Test that output follows JSONL format (one JSON object per line)."""
        storer = MetricsStorer(temp_output_dir)
        storer.open_for_run(is_testing=False)
        
        # Store CMs for two images
        storer.store_per_image_cms("image_001", sample_confusion_matrices, is_testing=False)
        storer.store_per_image_cms("image_002", sample_confusion_matrices, is_testing=False)
        
        storer.close()
        
        # Read file and verify JSONL format
        with open(storer.val_cm_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line.strip())
            assert 'image_id' in data
            assert 'confusion_matrices' in data


class TestMetricsStorerIntegration:
    """Test integration scenarios."""

    def test_full_workflow(self, temp_output_dir, sample_metrics_report, sample_confusion_matrices):
        """Test a complete workflow with both history and CM storage."""
        storer = MetricsStorer(temp_output_dir)
        
        # Simulate training workflow
        for epoch in range(1, 4):  # 3 epochs
            # Store epoch history
            modified_report = sample_metrics_report.copy()
            modified_report['optimization_metrics']['loss'] = 0.8 - (epoch * 0.1)
            storer.store_epoch_history(modified_report, epoch=epoch)
            
            # Simulate validation run
            storer.open_for_run(is_testing=False)
            for img_id in [f"val_img_{epoch}_{i}" for i in range(2)]:
                storer.store_per_image_cms(img_id, sample_confusion_matrices, is_testing=False)
            storer.close()
        
        # Check that all files exist
        assert os.path.exists(storer.history_path)
        assert os.path.exists(storer.val_cm_path)
        
        # Check history contains all epochs
        with open(storer.history_path, 'r') as f:
            history = json.load(f)
        assert history['epoch'] == [1, 2, 3]
        assert len(history['loss']) == 3
        
        # Check validation CMs file contains all images
        with open(storer.val_cm_path, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 6  # 3 epochs × 2 images per epoch

    def test_concurrent_file_access(self, temp_output_dir, sample_confusion_matrices):
        """Test behavior with multiple file handles."""
        storer = MetricsStorer(temp_output_dir)
        
        # Open both validation and test files
        storer.open_for_run(is_testing=False)
        storer.open_for_run(is_testing=True)
        
        # Store to both
        storer.store_per_image_cms("val_img", sample_confusion_matrices, is_testing=False)
        storer.store_per_image_cms("test_img", sample_confusion_matrices, is_testing=True)
        
        # Close both
        storer.close()
        
        # Check that both files exist and contain correct data
        assert os.path.exists(storer.val_cm_path)
        assert os.path.exists(storer.test_cm_path)
        
        with open(storer.val_cm_path, 'r') as f:
            val_data = json.loads(f.readline().strip())
        assert val_data['image_id'] == "val_img"
        
        with open(storer.test_cm_path, 'r') as f:
            test_data = json.loads(f.readline().strip())
        assert test_data['image_id'] == "test_img"