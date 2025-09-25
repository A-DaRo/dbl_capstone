# Edit file: tests/coral_mtl/metrics/metrics/test_coral_metrics.py
import pytest
import torch
from unittest.mock import MagicMock

from coral_mtl.metrics.metrics import CoralMetrics

# --- Test Suite ---

@pytest.mark.gpu
class TestCoralMetrics:
    """
    Test suite for the CoralMetrics (baseline) class, focusing on scientific correctness.
    """

    @pytest.fixture
    def metrics_calculator(self, splitter_base, device):
        """Provides a fresh CoralMetrics instance for each test."""
        mock_storer = MagicMock()
        metrics = CoralMetrics(
            splitter=splitter_base, storer=mock_storer, device=device, ignore_index=255
        )
        metrics.reset()
        return metrics

    def test_perfect_prediction(self, metrics_calculator, splitter_base, device):
        """
        Tests that with perfect predictions, all relevant metrics are optimal.
        """
        h, w, b = 16, 16, 2
        num_classes = splitter_base.num_global_classes
        
        # Target: a simple square of class 1 on a background of class 0
        target = torch.zeros(b, h, w, dtype=torch.long, device=device)
        target[:, h//4:h*3//4, w//4:w*3//4] = 1
        
        # Logits: perfectly confident and correct
        logits = torch.full((b, num_classes, h, w), -1000.0, device=device)
        logits.scatter_(1, target.unsqueeze(1), 1000.0)

        metrics_calculator.update(
            predictions=None, original_targets=target, image_ids=["im1", "im2"],
            epoch=0, predictions_logits=logits
        )
        report = metrics_calculator.compute()

        opt_metrics = report['optimization_metrics']
        assert opt_metrics['global.mIoU'] == pytest.approx(1.0)
        assert opt_metrics['global.BIoU'] == pytest.approx(1.0)
        assert opt_metrics['global.Boundary_F1'] == pytest.approx(1.0)
        assert opt_metrics['global.NLL'] < 1e-4
        assert opt_metrics['global.Brier_Score'] < 1e-4
        assert opt_metrics['global.ECE'] == pytest.approx(0.0)

    def test_completely_wrong_prediction(self, metrics_calculator, splitter_base, device):
        """
        Tests that with completely wrong predictions, scores are minimal.
        """
        h, w, b = 16, 16, 2
        num_classes = splitter_base.num_global_classes
        
        target = torch.ones(b, h, w, dtype=torch.long, device=device) # All class 1
        # Predict class 2 everywhere
        preds = torch.full((b, h, w), 2, dtype=torch.long, device=device)

        logits = torch.full((b, num_classes, h, w), -1000.0, device=device)
        logits.scatter_(1, preds.unsqueeze(1), 1000.0)

        metrics_calculator.update(
            predictions=None, original_targets=target, image_ids=["im1", "im2"],
            epoch=0, predictions_logits=logits
        )
        report = metrics_calculator.compute()

        opt_metrics = report['optimization_metrics']
        assert opt_metrics['global.mIoU'] == pytest.approx(0.0)
        assert opt_metrics['global.BIoU'] == pytest.approx(0.0)
        assert opt_metrics['global.Boundary_F1'] == pytest.approx(0.0)
        assert opt_metrics['global.NLL'] > 5.0 # Should be a large loss
        assert opt_metrics['global.Brier_Score'] > 1.0

    def test_ignore_index_handling(self, metrics_calculator, device):
        """
        Tests that a batch containing only the ignore_index is handled gracefully.
        """
        h, w, b = 16, 16, 2
        target = torch.full((b, h, w), 255, dtype=torch.long, device=device)
        logits = torch.randn(b, metrics_calculator.splitter.num_global_classes, h, w, device=device)

        metrics_calculator.update(
            predictions=None, original_targets=target, image_ids=["im1", "im2"],
            epoch=0, predictions_logits=logits
        )
        report = metrics_calculator.compute()

        assert report['optimization_metrics']['global.mIoU'] == 0.0
        assert report['optimization_metrics']['global.NLL'] == 0.0
        assert report['optimization_metrics']['global.ECE'] == 0.0
        assert metrics_calculator.total_pixels.item() == 0

    def test_calibration_uncertain_prediction(self, metrics_calculator, device):
        """
        Tests that uncertain (uniform) logits result in poor calibration scores.
        """
        h, w, b = 16, 16, 2
        num_classes = metrics_calculator.splitter.num_global_classes
        target = torch.ones(b, h, w, dtype=torch.long, device=device)
        # Uniform logits (all zeros) indicate maximum uncertainty
        logits = torch.zeros(b, num_classes, h, w, device=device)

        metrics_calculator.update(
            predictions=None, original_targets=target, image_ids=["im1", "im2"],
            epoch=0, predictions_logits=logits
        )
        report = metrics_calculator.compute()

        opt_metrics = report['optimization_metrics']
        # ECE should be high, as confidence is 1/N but accuracy is either 0 or 1.
        # The exact value is abs(accuracy - 1/N).
        expected_ece = abs(0.0 - (1/num_classes))
        assert opt_metrics['global.ECE'] == pytest.approx(expected_ece)
        # NLL for uniform prediction is log(num_classes)
        assert opt_metrics['global.NLL'] == pytest.approx(torch.log(torch.tensor(num_classes)).item())

    def test_no_foreground_in_target(self, metrics_calculator, splitter_base, device):
        """
        Tests the edge case where the ground truth contains only background pixels.
        """
        h, w, b = 16, 16, 2
        num_classes = splitter_base.num_global_classes
        
        # Target is all background
        target = torch.zeros(b, h, w, dtype=torch.long, device=device)
        # Prediction is also all background
        logits = torch.full((b, num_classes, h, w), -1000.0, device=device)
        logits[:, 0, :, :] = 1000.0

        metrics_calculator.update(
            predictions=None, original_targets=target, image_ids=["im1", "im2"],
            epoch=0, predictions_logits=logits
        )
        report = metrics_calculator.compute()

        opt_metrics = report['optimization_metrics']
        # mIoU is tricky here. Only background class has support. Most frameworks report 1.0.
        # Our nanmean will result in 1.0 because it computes IoU for class 0, which is 1.0, and NaNs for others.
        assert opt_metrics['global.mIoU'] == pytest.approx(1.0)
        # Boundary metrics should be 0 because there are no foreground boundaries
        assert opt_metrics['global.BIoU'] == pytest.approx(0.0)
        assert opt_metrics['global.Boundary_F1'] == pytest.approx(0.0)