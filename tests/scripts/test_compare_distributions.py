import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import pandas as pd
import yaml
import json
from pathlib import Path
import sys

# Adjust path to import from src
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from coral_mtl.scripts.compare_distributions import (
    _create_remapping_lut_from_yaml,
    _calculate_pixel_counts,
    compare_and_visualize
)

class TestCompareDistributionsHelpers(unittest.TestCase):

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=yaml.dump({
        'task1': {'mapping': {'1': [10, 11], '2': [20]}}
    }))
    def test_create_remapping_lut(self, mock_file, mock_exists):
        """ Test creation of a remapping LUT from a YAML file. """
        lut = _create_remapping_lut_from_yaml(Path("dummy.yaml"), num_original_classes=30)
        self.assertIsNotNone(lut)
        self.assertEqual(lut[10], 1)
        self.assertEqual(lut[11], 1)
        self.assertEqual(lut[20], 2)
        self.assertEqual(lut[5], 0)

    @patch('PIL.Image.open')
    @patch('pathlib.Path.glob')
    def test_calculate_pixel_counts(self, mock_glob, mock_img_open):
        """ Test the pixel counting logic with and without a LUT. """
        mock_glob.return_value = [Path("mask1.png"), Path("mask2.png")]
        
        mask1_array = np.array([[10, 10], [20, 30]])
        mask2_array = np.array([[10, 30], [30, 30]])
        
        with patch('numpy.array', side_effect=[mask1_array, mask2_array]):
            mock_img_open.side_effect = [MagicMock(), MagicMock()]
            counts_no_lut = _calculate_pixel_counts(Path("/fake"))
            self.assertEqual(counts_no_lut, {10: 3, 20: 1, 30: 4})

        with patch('numpy.array', side_effect=[mask1_array, mask2_array]):
            mock_img_open.side_effect = [MagicMock(), MagicMock()]
            remapping_lut = np.zeros(40, dtype=np.uint8)
            remapping_lut[10] = 1
            remapping_lut[20] = 2
            remapping_lut[30] = 1
            counts_with_lut = _calculate_pixel_counts(Path("/fake"), remapping_lut)
            self.assertEqual(counts_with_lut, {1: 7, 2: 1})

class TestCompareAndVisualize(unittest.TestCase):

    @patch('coral_mtl.scripts.compare_distributions._calculate_pixel_counts')
    @patch('coral_mtl.scripts.compare_distributions._create_remapping_lut_from_yaml')
    @patch('coral_mtl.scripts.compare_distributions._generate_plot_chunk')
    @patch('pathlib.Path.exists', return_value=True)
    def test_main_visualization_flow(self, mock_exists, mock_generate_plot, mock_create_lut, mock_calculate_counts):
        """
        Test the main `compare_and_visualize` function's orchestration.
        """
        original_counts = {1: 1000, 2: 500, 0: 10000}
        pds_counts = {1: 200, 2: 400, 0: 1000}
        mock_calculate_counts.side_effect = [pds_counts, original_counts]
        
        mock_create_lut.return_value = np.zeros(5, dtype=np.uint8)
        
        id2label_content = json.dumps({
            "0": "unlabeled",
            "1": "Coral",
            "2": "Rock"
        })
        
        m_open = mock_open(read_data=id2label_content)
        with patch("builtins.open", m_open):
            compare_and_visualize(
                original_dataset_root=Path("/fake/original"),
                pds_patch_dir=Path("/fake/pds"),
                output_dir=Path("/fake/output"),
                task_definition_path=Path("dummy.yaml")
            )

            self.assertEqual(mock_calculate_counts.call_count, 2)
            mock_create_lut.assert_called_once()
            
            mock_generate_plot.assert_called_once()
            
            call_args, _ = mock_generate_plot.call_args
            df_chunk = call_args[0]
            
            self.assertIsInstance(df_chunk, pd.DataFrame)
            self.assertNotIn("unlabeled", df_chunk.index)
            self.assertEqual(df_chunk.loc['Coral']['Original'], 1000)
            self.assertEqual(df_chunk.loc['Coral']['PDS'], 200)
            self.assertEqual(df_chunk.loc['Rock']['Original'], 500)
            self.assertEqual(df_chunk.loc['Rock']['PDS'], 400)
            
            handle = m_open()
            written_content = "".join(c[0][0] for c in handle.write.call_args_list)
            
            self.assertIn("Dataset Distribution Comparison Report", written_content)
            self.assertTrue(written_content.find("Rock") < written_content.find("Coral"))
            self.assertIn("Rock", written_content)
            self.assertIn("400", written_content)

if __name__ == '__main__':
    unittest.main()