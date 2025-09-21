import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
from pathlib import Path

# Adjust path to import from src
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from coral_mtl.scripts.id2labels_labels2colors_coralscapes import (
    get_coralscapes_mappings,
    display_mappings,
    decode_and_display_labels,
    create_segmentation_legend
)

class TestCoralScapesMappings(unittest.TestCase):

    @patch('coral_mtl.scripts.id2labels_labels2colors_coralscapes.hf_hub_download')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"1": "coral"}')
    def test_get_mappings_success(self, mock_open, mock_hf_download):
        """
        Tests successful fetching and processing of mappings.
        """
        # --- Setup ---
        # Mock hf_hub_download to return dummy paths
        mock_hf_download.side_effect = ['/fake/id2label.json', '/fake/label2color.json']
        
        # Configure mock_open for two different files
        id2label_data = '{"1": "coral", "2": "rock"}'
        label2color_data = '{"coral": [255, 0, 0], "rock": [128, 128, 128]}'
        
        # The mock needs to handle two separate `open` calls with different content
        m = unittest.mock.mock_open()
        m.side_effect = [
            unittest.mock.mock_open(read_data=id2label_data).return_value,
            unittest.mock.mock_open(read_data=label2color_data).return_value
        ]

        with patch('builtins.open', m):
            # --- Execution ---
            id2label, label2color = get_coralscapes_mappings()

            # --- Assertions ---
            # Check that keys are correctly converted to integers for id2label
            self.assertIn(1, id2label)
            self.assertEqual(id2label[1], "coral")
            self.assertNotIn("1", id2label) # Ensure string key is gone
            
            # Check that color lists are correctly converted to tuples
            self.assertIsInstance(label2color["coral"], tuple)
            self.assertEqual(label2color["coral"], (255, 0, 0))
            
            # Check that the 'unlabeled' class is added automatically
            self.assertIn(0, id2label)
            self.assertEqual(id2label[0], "unlabeled")
            self.assertIn("unlabeled", label2color)
            self.assertEqual(label2color["unlabeled"], (0, 0, 0))

    @patch('coral_mtl.scripts.id2labels_labels2colors_coralscapes.hf_hub_download', side_effect=Exception("Network error"))
    def test_get_mappings_failure(self, mock_hf_download):
        """
        Tests that the function returns (None, None) on failure.
        """
        id2label, label2color = get_coralscapes_mappings()
        self.assertIsNone(id2label)
        self.assertIsNone(label2color)

    @patch('builtins.print')
    def test_display_mappings(self, mock_print):
        """
        Tests the display function to ensure it prints formatted output.
        """
        id2label = {1: 'coral', 2: 'rock'}
        label2color = {'coral': (255,0,0), 'rock': (128,128,128)}
        
        display_mappings(id2label, label2color, num_examples=1)
        
        # Check that the print function was called with expected substrings
        mock_print.assert_any_call("ID 1: coral")
        mock_print.assert_any_call("Label 'coral': (255, 0, 0)")

    @patch('builtins.print')
    def test_decode_labels(self, mock_print):
        """
        Tests the decoding of a label mask.
        """
        label_mask = np.array([[1, 1], [2, 0]])
        id2label = {0: 'unlabeled', 1: 'coral', 2: 'rock'}
        label2color = {'unlabeled': (0,0,0), 'coral': (255,0,0), 'rock': (128,128,128)}
        
        decode_and_display_labels(label_mask, id2label, label2color)
        
        # Check for output related to each unique ID in the mask
        mock_print.assert_any_call("ID: 0   | Name: unlabeled                 | Color: (0, 0, 0)")
        mock_print.assert_any_call("ID: 1   | Name: coral                     | Color: (255, 0, 0)")
        mock_print.assert_any_call("ID: 2   | Name: rock                      | Color: (128, 128, 128)")

    @patch('PIL.Image.new')
    @patch('PIL.ImageDraw.Draw')
    @patch('PIL.ImageFont.load_default')
    def test_create_legend(self, mock_font, mock_draw, mock_img_new):
        """
        Tests the creation of a legend image.
        """
        id2label = {0: 'unlabeled', 1: 'coral'}
        label2color = {'unlabeled': (0,0,0), 'coral': (255,0,0)}
        
        # Mock the Image and Draw objects to avoid actual image creation
        mock_image_instance = MagicMock(spec=Image.Image)
        mock_draw_instance = MagicMock(spec=ImageDraw.ImageDraw)
        mock_img_new.return_value = mock_image_instance
        mock_draw.return_value = mock_draw_instance
        
        legend_img = create_segmentation_legend(id2label, label2color)
        
        self.assertIsNotNone(legend_img)
        # Check that a new image was created
        mock_img_new.assert_called_once()
        # Check that drawing functions were called for the 'coral' class
        # (unlabeled is excluded)
        mock_draw_instance.rectangle.assert_called_once()
        mock_draw_instance.text.assert_called_once_with(unittest.mock.ANY, "1: coral", fill="black", font=unittest.mock.ANY)

if __name__ == '__main__':
    unittest.main()
