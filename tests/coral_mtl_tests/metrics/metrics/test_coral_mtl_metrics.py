# Edit file: tests/coral_mtl/metrics/metrics/test_coral_mtl_metrics.py
import pytest
import torch
from unittest.mock import MagicMock

from coral_mtl.metrics.metrics import CoralMTLMetrics

@pytest.mark.gpu
class TestCoralMTLMetrics:
    """
    Test suite for CoralMTLMetrics, focusing on scientific correctness and
    hierarchical report structure.
    """

    @pytest.fixture
    def metrics_calculator(self, splitter_mtl, device):
        """Provides a fresh CoralMTLMetrics instance for each test."""
        mock_storer = MagicMock()
        metrics = CoralMTLMetrics(
            splitter=splitter_mtl, storer=mock_storer, device=device, ignore_index=255
        )
        metrics.reset()
        return metrics

    def test_perfect_prediction(self, metrics_calculator, splitter_mtl, device):
        """
        Tests that perfect predictions for all tasks result in optimal metrics.
        """
        h, w, b = 16, 16, 2
        original_target = torch.zeros(b, h, w, dtype=torch.long, device=device)
        genus_defs = splitter_mtl.hierarchical_definitions.get('genus')
        if genus_defs is None:
            genus_defs = next(iter(splitter_mtl.hierarchical_definitions.values()))
        available_raw_ids = [rid for rid in sorted(genus_defs['ungrouped']['id2label'].keys()) if rid != 0]
        sample_raw_id = available_raw_ids[0] if available_raw_ids else 0
        if sample_raw_id != 0:
            original_target[:, h//4:h*3//4, w//4:w*3//4] = sample_raw_id

        logits_dict = {}
        for task, details in splitter_mtl.hierarchical_definitions.items():
            num_classes = len(details['ungrouped']['id2label'])
            mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(device)
            task_target = mapping[original_target]
            
            task_logits = torch.full((b, num_classes, h, w), -1000.0, device=device)
            task_logits.scatter_(1, task_target.unsqueeze(1), 1000.0)
            logits_dict[task] = task_logits

        metrics_calculator.update(
            predictions=None, original_targets=original_target, image_ids=["im1", "im2"],
            epoch=0, predictions_logits=logits_dict
        )
        report = metrics_calculator.compute()

        for task_name, task_report in report['tasks'].items():
            assert task_report['ungrouped']['task_summary']['pixel_accuracy'] == pytest.approx(1.0)
            assert task_report['ungrouped']['task_summary']['mIoU'] == pytest.approx(1.0)

    def test_completely_wrong_prediction(self, metrics_calculator, splitter_mtl, device):
        """
        Tests that completely wrong predictions result in minimal scores.
        """
        h, w, b = 16, 16, 2
        original_target = torch.ones(b, h, w, dtype=torch.long, device=device)

        logits_dict = {}
        for task, details in splitter_mtl.hierarchical_definitions.items():
            num_classes = len(details['ungrouped']['id2label'])
            wrong_class_idx = 2 if num_classes > 2 else 0
            task_preds = torch.full((b, h, w), wrong_class_idx, dtype=torch.long, device=device)
            
            task_logits = torch.full((b, num_classes, h, w), -1000.0, device=device)
            task_logits.scatter_(1, task_preds.unsqueeze(1), 1000.0)
            logits_dict[task] = task_logits

        metrics_calculator.update(
            predictions=None, original_targets=original_target, image_ids=["im1", "im2"],
            epoch=0, predictions_logits=logits_dict
        )
        report = metrics_calculator.compute()

        opt_metrics = report['optimization_metrics']
        assert opt_metrics['global.mIoU'] == pytest.approx(0.0)
        assert opt_metrics['global.BIoU'] == pytest.approx(0.0)
        assert opt_metrics['global.Boundary_F1'] == pytest.approx(0.0)

    def test_ignore_index_handling(self, metrics_calculator, splitter_mtl, device):
        """
        Tests graceful handling of a batch containing only the ignore_index.
        """
        h, w, b = 16, 16, 2
        target = torch.full((b, h, w), 255, dtype=torch.long, device=device)
        
        logits_dict = {}
        for task, details in splitter_mtl.hierarchical_definitions.items():
            num_classes = len(details['ungrouped']['id2label'])
            logits_dict[task] = torch.randn(b, num_classes, h, w, device=device)

        metrics_calculator.update(
            predictions=None, original_targets=target, image_ids=["im1", "im2"],
            epoch=0, predictions_logits=logits_dict
        )
        report = metrics_calculator.compute()

        assert report['optimization_metrics']['global.mIoU'] == 0.0
        assert report['optimization_metrics']['global.NLL'] == 0.0
        assert report['optimization_metrics']['global.ECE'] == 0.0
        assert metrics_calculator.total_pixels.item() == 0

    def test_hierarchical_report_structure(self, metrics_calculator, splitter_mtl, device):
        """
        CRITICAL: Verifies the full, nested structure of the computed report,
        ensuring 'grouped' and 'ungrouped' levels are present and correct.
        """
        if not any(d.get('is_grouped') for d in splitter_mtl.hierarchical_definitions.values()):
            pytest.skip("This test requires a task definition with grouping.")

        h, w, b = 16, 16, 1
        # Create a target with multiple classes that will be grouped.
        original_target = torch.zeros(b, h, w, dtype=torch.long, device=device)
        genus_defs = splitter_mtl.hierarchical_definitions['genus']
        grouped_mapping = genus_defs.get('grouped', {}).get('mapping', {})
        group_raw_choices = []
        for group_id, raw_values in grouped_mapping.items():
            if group_id == 0:
                continue
            if isinstance(raw_values, list):
                if raw_values:
                    group_raw_choices.append(raw_values[0])
            else:
                group_raw_choices.append(raw_values)
        ungrouped_ids = [rid for rid in sorted(genus_defs['ungrouped']['id2label'].keys()) if rid != 0]
        fallback_id = ungrouped_ids[0] if ungrouped_ids else 0
        raw_a = group_raw_choices[0] if group_raw_choices else fallback_id
        raw_b = group_raw_choices[1] if len(group_raw_choices) > 1 else raw_a
        if raw_a != 0:
            original_target[:, :, :w//2] = raw_a
        if raw_b != 0:
            original_target[:, :, w//2:] = raw_b

        logits_dict = {}
        for task, details in splitter_mtl.hierarchical_definitions.items():
            num_classes = len(details['ungrouped']['id2label'])
            mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(device)
            task_target = mapping[original_target]
            task_logits = torch.nn.functional.one_hot(task_target, num_classes).permute(0, 3, 1, 2).float()
            logits_dict[task] = task_logits

        metrics_calculator.update(
            predictions=None, original_targets=original_target, image_ids=["im1"],
            epoch=0, predictions_logits=logits_dict
        )
        report = metrics_calculator.compute()

        # --- Assertions ---
        assert 'tasks' in report
        genus_report = report['tasks'].get('genus')
        assert genus_report is not None, "Genus task missing from report"
        
        assert 'ungrouped' in genus_report
        assert 'grouped' in genus_report
        
        ungrouped_summary = genus_report['ungrouped']['task_summary']
        grouped_summary = genus_report['grouped']['task_summary']
        
        # At the ungrouped level, we predicted two different classes perfectly.
        assert ungrouped_summary['mIoU'] == pytest.approx(1.0)
        
        # At the grouped level, both classes mapped to the same super-class, also perfect.
        assert grouped_summary['mIoU'] == pytest.approx(1.0)
        assert 'per_class' in genus_report['grouped']
        grouped_labels = set(genus_report['grouped']['per_class'].keys())
        expected_grouped_labels = set(
            splitter_mtl.hierarchical_definitions['genus']['grouped']['id2label'].values()
        )
        assert expected_grouped_labels.issubset(grouped_labels)