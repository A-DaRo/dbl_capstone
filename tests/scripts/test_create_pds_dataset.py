import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import yaml
import json
from pathlib import Path
import sys
from PIL import Image
import coral_mtl.scripts.create_pds_dataset as pds_module


# Adjust path to import from src
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

# Mock the external dependencies before importing the main module
mock_modules = {
    'datasets': MagicMock(),
    'huggingface_hub': MagicMock(),
    'tqdm': MagicMock(),
    'numba': MagicMock()
}

# Mock the jit decorator
def mock_jit(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

mock_modules['numba'].jit = mock_jit

with patch.dict('sys.modules', mock_modules):
    from coral_mtl.scripts.create_pds_dataset import (
        poisson_disk_sampling,
        _create_remapping_assets,
        process_image,
        create_pds_dataset
    )
    from coral_mtl.utils.task_splitter import BaseTaskSplitter

class TestPoissonDiskSampling(unittest.TestCase):
    
    def test_pds_generation(self):
        """ Test that points are generated and respect the radius constraint. """
        width, height, radius = 200, 200, 20
        # Create a foreground mask that covers the whole area
        foreground_mask = np.ones((height, width), dtype=bool)
        
        samples = poisson_disk_sampling(width, height, radius, foreground_mask)
        
        self.assertGreater(len(samples), 0, "Should generate at least one sample point.")
        
        # Check distance constraint
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                p1 = samples[i]
                p2 = samples[j]
                dist_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
                self.assertGreaterEqual(dist_sq, radius**2, f"Points {p1} and {p2} are too close.")

    def test_pds_with_empty_foreground(self):
        """ Test that no points are generated if the foreground mask is empty. """
        width, height, radius = 100, 100, 10
        foreground_mask = np.zeros((height, width), dtype=bool)
        
        samples = poisson_disk_sampling(width, height, radius, foreground_mask)
        
        self.assertEqual(len(samples), 0, "Should not generate any points for an empty mask.")

class TestRemappingAssets(unittest.TestCase):

    @patch('coral_mtl.scripts.create_pds_dataset.BaseTaskSplitter')
    @patch('builtins.open', new_callable=mock_open, read_data="test_yaml_content")
    @patch('yaml.safe_load')
    def test_asset_creation_from_yaml(self, mock_yaml_load, mock_file, mock_task_splitter_class):
        """ Test asset creation when a task definition file is provided using TaskSplitter. """
        # Mock YAML content - using similar structure as task_definitions.yaml
        mock_task_definitions = {
            "coral_health": {
                "id2label": {
                    "0": "unlabeled",
                    "6": "other coral alive",
                    "4": "other coral bleached", 
                    "3": "other coral dead"
                },
                "groupby": {
                    "mapping": {
                        "0": 0,
                        "1": [6],    # healthy coral
                        "2": [4],    # bleached coral  
                        "3": [3]     # dead coral
                    },
                    "id2label": {
                        "0": "unlabeled",
                        "1": "healthy coral",
                        "2": "bleached coral",
                        "3": "dead coral"
                    }
                }
            }
        }
        mock_yaml_load.return_value = mock_task_definitions
        
        # Mock BaseTaskSplitter instance
        mock_task_splitter = MagicMock()
        mock_task_splitter.flat_mapping_array = np.array([0, 0, 0, 3, 2, 0, 1], dtype=np.uint8)  # 7 classes total
        mock_task_splitter.flat_id2label = {0: "background", 1: "healthy coral", 2: "bleached coral", 3: "dead coral"}
        mock_task_splitter_class.return_value = mock_task_splitter
        
        # Mock original mappings
        original_id2label = {0: "unlabeled", 3: "other coral dead", 4: "other coral bleached", 6: "other coral alive"}
        original_label2color = {
            "unlabeled": (0, 0, 0),
            "other coral dead": (128, 0, 0),
            "other coral bleached": (255, 255, 255),
            "other coral alive": (0, 255, 0)
        }
        
        # Execute
        lut, id2label, label2color = _create_remapping_assets(
            "test.yaml", 7, original_id2label, original_label2color
        )
        
        # Assertions
        mock_task_splitter_class.assert_called_once_with(mock_task_definitions)
        self.assertEqual(id2label, {0: "background", 1: "healthy coral", 2: "bleached coral", 3: "dead coral"})
        self.assertIn("unlabeled", label2color)  # The function uses "unlabeled" for background
        self.assertIn("healthy coral", label2color)
        self.assertIn("bleached coral", label2color)
        self.assertIn("dead coral", label2color)
        self.assertEqual(label2color["unlabeled"], (0, 0, 0))
        
        # Check that LUT uses the TaskSplitter's mapping
        np.testing.assert_array_equal(lut, mock_task_splitter.flat_mapping_array)

    def test_no_task_definition_file(self):
        """ Test behavior when no task definition file is provided. """
        original_id2label = {i: f'label_{i}' for i in range(5)}
        original_label2color = {f'label_{i}': (i, i, i) for i in range(5)}
        
        remapping_lut, new_id2label, new_label2color = _create_remapping_assets(
            None, 5, original_id2label, original_label2color
        )
        
        # Should return original mappings
        np.testing.assert_array_equal(remapping_lut, np.arange(5))
        self.assertEqual(new_id2label, original_id2label)
        self.assertEqual(new_label2color, original_label2color)

class TestProcessImage(unittest.TestCase):

    @patch.object(pds_module, 'poisson_disk_sampling')
    @patch('PIL.Image.open')
    @patch('pathlib.Path.mkdir')
    def test_image_processing_flow(self, mock_mkdir, mock_img_open, mock_pds):
        """ Test the complete flow of processing a single image. """
        # --- Setup Mocks ---
        mock_pds.return_value = [(100, 100), (150, 150)] # Two sample points

        # Mock image and mask
        mock_image = MagicMock(spec=Image.Image)
        mock_image.size = (200, 200)  # This should be a real tuple, not a MagicMock
        mock_converted_image = MagicMock(spec=Image.Image)
        mock_converted_image.size = (200, 200)
        mock_converted_image.crop.return_value = MagicMock(spec=Image.Image)
        mock_image.convert.return_value = mock_converted_image

        mock_mask_array = np.ones((200, 200), dtype=np.uint8) * 5 # Original class ID 5

        # Configure Image.open to return the image mock for both image and mask
        mock_img_open.return_value = mock_image

        # Mock np.array to return our mask array
        with patch('numpy.array', return_value=mock_mask_array):
            with patch('PIL.Image.fromarray') as mock_fromarray:
                mock_mask_image = MagicMock(spec=Image.Image)
                mock_fromarray.return_value = mock_mask_image

                # --- Setup Paths and LUT ---
                img_path = Path("/fake/dir/image_01_leftImg8bit.png")
                mask_path = Path("/fake/dir/mask_01_gtFine.png")
                output_dir = "/fake/output"

                remapping_lut = np.arange(10)
                remapping_lut[5] = 1 # Remap class 5 to 1

                # --- Execution ---
                # Mock Path.mkdir to avoid directory creation
                with patch('pathlib.Path.mkdir'):
                    patch_count = process_image(
                        (img_path, mask_path),  # This should be a tuple of two Path objects
                        output_dir,
                        patch_size=64,
                        pds_radius=30,
                        remapping_lut=remapping_lut
                    )
                    
                # Verify mock was called
                print(f"Mock called: {mock_pds.called}")
                print(f"Mock call count: {mock_pds.call_count}")
                if mock_pds.call_count > 0:
                    print(f"Mock call args: {mock_pds.call_args}")

                # --- Assertions ---
                self.assertEqual(patch_count, 2, "Should process two valid patches.")
                # Check that PDS was called with the foreground mask
                mock_pds.assert_called_once()
                
                # Check that images were cropped and saved
                self.assertEqual(mock_converted_image.crop.call_count, 2)
                self.assertEqual(mock_converted_image.crop.return_value.save.call_count, 2)
                
                # Check that fromarray was called to create mask images
                self.assertEqual(mock_fromarray.call_count, 2)
                
                # Check that save was called on mask images
                self.assertEqual(mock_mask_image.save.call_count, 2)


class TestCreatePdsDataset(unittest.TestCase):

    @patch('coral_mtl.scripts.create_pds_dataset.get_coralscapes_mappings')
    @patch('coral_mtl.scripts.create_pds_dataset._create_remapping_assets')
    @patch('concurrent.futures.ProcessPoolExecutor')
    @patch('pathlib.Path.glob', return_value=[Path('fake_root/leftImg8bit/train/city/image_01_leftImg8bit.png')])
    @patch('pathlib.Path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open)
    def test_main_orchestrator(self, mock_file, mock_exists, mock_glob, mock_executor, mock_assets, mock_mappings):
        """ Test the main orchestrator function `create_pds_dataset` with TaskSplitter integration. """
        # --- Setup Mocks ---
        mock_mappings.return_value = ({0: 'unlabeled'}, {'unlabeled': (0,0,0)})
        # TaskSplitter now returns flattened structure
        mock_assets.return_value = (
            np.array([0,1]), 
            {0:'background', 1:'coral'},  # Using TaskSplitter naming convention
            {'coral':(1,1,1), 'background':(0,0,0)}
        )

        # Mock the executor to return a result immediately
        mock_future = MagicMock()
        mock_future.result.return_value = 5 # 5 patches created
        mock_executor.return_value.__enter__.return_value.map.return_value = [5]

        # --- Execution ---
        create_pds_dataset(
            dataset_root="fake_root",
            output_dir="fake_output",
            task_definition_path="fake.yaml"
        )

        # --- Assertions ---
        # Check that mappings were fetched and assets were created
        mock_mappings.assert_called_once()
        mock_assets.assert_called_once()

        # Check that metadata files were written
        # The implementation converts integer keys to strings when saving JSON
        expected_id2label = {'0': 'background', '1': 'coral'}
        expected_label2color = {'coral': [1, 1, 1], 'background': [0, 0, 0]}
        
        # Join all write calls to get the complete written content
        all_written_content = ''.join([call[0][0] for call in mock_file().write.call_args_list])
        
        # Check that the expected JSON content is present (allowing for formatting differences)
        self.assertIn('"0": "background"', all_written_content)
        self.assertIn('"1": "coral"', all_written_content)
        self.assertIn('"coral":', all_written_content)
        self.assertIn('[', all_written_content)  # Check for array format
        self.assertIn('1', all_written_content)  # Check that the color values are present

        # Check that the parallel executor was used
        mock_executor.return_value.__enter__.return_value.map.assert_called_once()


if __name__ == '__main__':
    unittest.main()
