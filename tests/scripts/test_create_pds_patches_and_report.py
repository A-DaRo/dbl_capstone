import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add the src directory to the Python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from coral_mtl.scripts.create_pds_patches_and_report import create_pds_patches_and_report

class TestCreatePdsPatchesAndReport(unittest.TestCase):

    @patch('coral_mtl.scripts.create_pds_patches_and_report.create_pds_dataset')
    @patch('coral_mtl.scripts.create_pds_patches_and_report.analyze_distribution')
    @patch('coral_mtl.scripts.create_pds_patches_and_report.compare_and_visualize')
    def test_pipeline_orchestration(self, mock_compare, mock_analyze, mock_create):
        """
        Tests that the main orchestration function calls the pipeline steps in sequence
        with the correct parameters.
        """
        # --- Setup ---
        # Using MagicMock for Path objects to avoid filesystem interactions
        mock_dataset_root = MagicMock(spec=Path)
        mock_pds_output_dir = MagicMock(spec=Path)
        mock_analysis_output_dir = MagicMock(spec=Path)
        
        # Define test parameters
        patch_size = 512
        pds_radius = 300
        num_workers = 8
        mock_task_def_path = MagicMock(spec=Path)

        # --- Execution ---
        # Call the function that orchestrates the pipeline
        create_pds_patches_and_report(
            dataset_root=mock_dataset_root,
            pds_output_dir=mock_pds_output_dir,
            analysis_output_dir=mock_analysis_output_dir,
            patch_size=patch_size,
            pds_radius=pds_radius,
            num_workers=num_workers,
            task_definition_path=mock_task_def_path
        )

        # --- Assertions ---
        # 1. Check if create_pds_dataset was called correctly
        mock_create.assert_called_once_with(
            dataset_root=str(mock_dataset_root),
            output_dir=str(mock_pds_output_dir),
            patch_size=patch_size,
            pds_radius=pds_radius,
            num_workers=num_workers,
            task_definition_path=str(mock_task_def_path)
        )
        
        # The following steps are currently commented out in the source code.
        # If they are re-enabled, these assertions will validate their behavior.
        
        # 2. Check if analyze_distribution was called correctly
        # The script constructs a subdirectory, so we need to check that
        # pds_analysis_out_dir = analysis_output_dir / "pds_analysis"
        # mock_analyze.assert_called_once()
        # call_args, call_kwargs = mock_analyze.call_args
        # self.assertEqual(call_kwargs['patch_dir'], mock_pds_output_dir)
        # self.assertEqual(call_kwargs['output_dir'], mock_analysis_output_dir / "pds_analysis")

        # 3. Check if compare_and_visualize was called correctly
        # comparison_out_dir = analysis_output_dir / "comparison_results"
        # mock_compare.assert_called_once_with(
        #     original_dataset_root=mock_dataset_root,
        #     pds_patch_dir=mock_pds_output_dir,
        #     output_dir=mock_analysis_output_dir / "comparison_results",
        #     task_definition_path=mock_task_def_path
        # )

    @patch('coral_mtl.scripts.create_pds_patches_and_report.create_pds_dataset')
    def test_no_task_definition(self, mock_create):
        """
        Tests that the pipeline runs correctly when task_definition_path is None.
        """
        mock_dataset_root = MagicMock(spec=Path)
        mock_pds_output_dir = MagicMock(spec=Path)
        
        create_pds_patches_and_report(
            dataset_root=mock_dataset_root,
            pds_output_dir=mock_pds_output_dir,
            analysis_output_dir=MagicMock(spec=Path),
            patch_size=256,
            pds_radius=150,
            num_workers=4,
            task_definition_path=None
        )
        
        mock_create.assert_called_once()
        # Verify that task_definition_path was passed as None
        self.assertIsNone(mock_create.call_args.kwargs['task_definition_path'])

if __name__ == '__main__':
    unittest.main()
