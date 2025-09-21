import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
import json
import numpy as np
from PIL import Image
import sys
import os

# Adjust path to import from src
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from coral_mtl.scripts.analyze_patch_distribution import analyze_distribution

class TestAnalyzePatchDistribution(unittest.TestCase):

    @patch('pathlib.Path.mkdir')
    @patch('PIL.Image.open')
    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.exists')
    def test_distribution_analysis(self, mock_exists, mock_glob, mock_img_open, mock_mkdir):
        """
        Tests the core logic of analyzing mask distributions and generating a report.
        """
        # --- Setup Mocks ---
        # 1. Mock the filesystem and metadata
        mock_exists.return_value = True  # Mock that paths exist
        mock_patch_dir = Path("/fake/patch_dir")
        mock_output_dir = Path("/fake/output_dir")
        
        # Mock id2label.json
        id2label_content = json.dumps({
            "0": "unlabeled",
            "1": "coral",
            "2": "rock"
        })
        
        # Mock mask files found by glob
        mock_mask_files = [
            mock_patch_dir / "masks" / "mask1.png",
            mock_patch_dir / "masks" / "mask2.png"
        ]
        mock_glob.return_value = mock_mask_files
        
        # 2. Mock the content of the mask images
        mask1_array = np.array([[1]*10 + [2]*5] * 10, dtype=np.uint8)
        mask2_array = np.array([[1]*20 + [0]*30] * 10, dtype=np.uint8)
        
        mock_images = [Image.fromarray(mask1_array), Image.fromarray(mask2_array)]
        mock_img_open.side_effect = mock_images

        # 3. Use mock_open to handle both reading and writing
        m_open = mock_open(read_data=id2label_content)
        with patch("builtins.open", m_open):
        
            # --- Execution ---
            analyze_distribution(mock_patch_dir, mock_output_dir)

            # --- Assertions ---
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            
            m_open.assert_any_call(mock_patch_dir / "id2label.json", 'r')
            m_open.assert_any_call(mock_output_dir / "distribution_report.md", "w")
            
            handle = m_open()
            written_content = "".join(c[0][0] for c in handle.write.call_args_list)
            
            self.assertIn("| 0   | unlabeled                    | 300                  | 46.154%   |", written_content)
            self.assertIn("| 1   | coral                        | 300                  | 46.154%   |", written_content)
            self.assertIn("| 2   | rock                         | 50                   | 7.692%   |", written_content)
            self.assertIn("# Patch Dataset Class Distribution Report", written_content)

    @patch('builtins.print')
    def test_no_masks_found(self, mock_print):
        """
        Tests that an error is printed if no mask files are found.
        """
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.glob', return_value=[]): # No files found
                with patch('builtins.open', mock_open(read_data='{}')):
                    
                    analyze_distribution(Path("/fake/patch"), Path("/fake/out"))
                    
                    # Construct expected path with os-specific separator
                    expected_path = os.path.join('/fake/patch', 'masks')
                    mock_print.assert_any_call(f"Error: No .png mask files found in '{Path(expected_path)}'.")

if __name__ == '__main__':
    unittest.main()