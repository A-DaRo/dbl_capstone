# tests/coral_mtl/model/decoders/test_hierarchical_context_aware_decoder.py

import pytest
import torch
from typing import List, Dict

from coral_mtl.model.decoders import HierarchicalContextAwareDecoder, MLP


def create_dummy_encoder_features(
    batch_size: int = 2,
    h: int = 64,
    w: int = 64,
    channels: List[int] = None,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Helper to generate a list of multi-scale feature maps."""
    if channels is None:
        channels = [32, 64, 160, 256]  # Corresponds to mit-b0
    downsample_factors = [4, 8, 16, 32]
    features = [
        torch.randn(batch_size, c, h // f, w // f, device=device)
        for c, f in zip(channels, downsample_factors)
    ]
    return features

# --- Dynamic Test Configurations ---
BASE_NUM_CLASSES = {
    "task_a": 15, "task_b": 4, "task_c": 2, "task_d": 5,
}

# A list of different configurations to test
# Each tuple is (test_id, primary_tasks_list, aux_tasks_list)
TASK_CONFIGURATIONS = [
    ("standard", ["task_a", "task_b"], ["task_c", "task_d"]),
    ("inverted", ["task_c", "task_d"], ["task_a", "task_b"]),
    ("all_primary", ["task_a", "task_b", "task_c", "task_d"], []),
    ("all_auxiliary", [], ["task_a", "task_b", "task_c", "task_d"]),
    ("single_primary", ["task_a"], ["task_b", "task_c", "task_d"]),
    ("single_auxiliary", ["task_a", "task_b", "task_c"], ["task_d"]),
    ("no_tasks", [], []),
]


@pytest.mark.gpu
class TestHierarchicalContextAwareDecoder:
    """Test suite for the HierarchicalContextAwareDecoder."""

    @pytest.mark.parametrize("test_id, primary_tasks, aux_tasks", TASK_CONFIGURATIONS)
    def test_initialization(self, test_id, primary_tasks, aux_tasks):
        """
        Tests that the decoder initializes correctly for various task configurations,
        creating the correct types of decoders and attention modules for each role.
        """
        # Setup
        encoder_channels = [32, 64, 160, 256]
        all_tasks = primary_tasks + aux_tasks

        # Action
        decoder = HierarchicalContextAwareDecoder(
            encoder_channels=encoder_channels, decoder_channel=128,
            num_classes=BASE_NUM_CLASSES, primary_tasks=primary_tasks, aux_tasks=aux_tasks,
        )

        # Assertion
        assert set(decoder.decoders.keys()) == set(all_tasks)
        assert set(decoder.predictors.keys()) == set(all_tasks)
        assert set(decoder.to_qkv.keys()) == set(all_tasks)

        # Assert attention modules only exist for primary tasks
        assert set(decoder.attn_proj.keys()) == set(primary_tasks)
        assert set(decoder.gating_layers.keys()) == set(primary_tasks)
        
        # Assert correct decoder types were created
        for task in primary_tasks:
            assert isinstance(decoder.decoders[task], MLP)
        for task in aux_tasks:
            assert isinstance(decoder.decoders[task], torch.nn.Conv2d)

    @pytest.mark.parametrize("test_id, primary_tasks, aux_tasks", TASK_CONFIGURATIONS)
    def test_forward_pass_shape(self, device, test_id, primary_tasks, aux_tasks):
        """
        Verifies the forward pass for various configurations, checking that the output
        is a dictionary with correctly shaped tensors for each defined task.
        """
        # Setup
        batch_size, h, w = 2, 128, 128
        encoder_channels = [32, 64, 160, 256]
        features = create_dummy_encoder_features(batch_size, h, w, encoder_channels, device)
        decoder = HierarchicalContextAwareDecoder(
            encoder_channels=encoder_channels, decoder_channel=128,
            num_classes=BASE_NUM_CLASSES, primary_tasks=primary_tasks, aux_tasks=aux_tasks
        ).to(device)
        decoder.eval()

        # Action
        with torch.no_grad():
            output = decoder(features)

        # Assertion
        all_tasks = primary_tasks + aux_tasks
        assert isinstance(output, dict)
        assert set(output.keys()) == set(all_tasks)

        for task in all_tasks:
            num_cls = BASE_NUM_CLASSES[task]
            expected_shape = (batch_size, num_cls, h // 4, w // 4)
            assert output[task].shape == expected_shape, f"Shape mismatch for task '{task}'."
            assert output[task].device.type == device.type

    @pytest.mark.parametrize("test_id, primary_tasks, aux_tasks", TASK_CONFIGURATIONS)
    def test_gradient_flow(self, device, test_id, primary_tasks, aux_tasks):
        """
        Ensures gradients flow through the entire architecture for any valid task configuration.
        """
        if not primary_tasks and not aux_tasks:
            pytest.skip("Skipping gradient test for no-task configuration.")

        # Setup
        encoder_channels = [16, 32, 64, 128]
        features = create_dummy_encoder_features(channels=encoder_channels, device=device)
        decoder = HierarchicalContextAwareDecoder(
            encoder_channels=encoder_channels, decoder_channel=64, attention_dim=32,
            num_classes=BASE_NUM_CLASSES, primary_tasks=primary_tasks, aux_tasks=aux_tasks
        ).to(device)
        decoder.train()

        # Action
        output = decoder(features)
        loss = sum(logit_tensor.mean() for logit_tensor in output.values())
        loss.backward()

        # Assertion
        def module_has_grad(module: torch.nn.Module) -> bool:
            return any(
                p.requires_grad and p.grad is not None and torch.any(p.grad != 0)
                for p in module.parameters()
            )

        # Decoder heads should always contribute to the loss
        for task, head in decoder.decoders.items():
            assert module_has_grad(head), f"Decoder head for task '{task}' received no gradient."

        # Prediction heads must also receive gradients
        for task, predictor in decoder.predictors.items():
            assert module_has_grad(predictor), f"Predictor for task '{task}' received no gradient."

        if decoder.primary_tasks:
            # Query projections and gating for primary tasks must have gradients
            for task in decoder.primary_tasks:
                query_proj = decoder.to_qkv[task][0]
                assert module_has_grad(query_proj), f"Query projection for primary task '{task}' has no gradient."
                assert module_has_grad(decoder.attn_proj[task]), f"Attention projector for '{task}' has no gradient."
                assert module_has_grad(decoder.gating_layers[task]), f"Gating layer for '{task}' has no gradient."

            # At least one key/value projection should participate when multiple tasks provide context
            if len(decoder.tasks) > 1:
                has_context_grad = any(
                    module_has_grad(decoder.to_qkv[task][idx])
                    for task in decoder.tasks
                    for idx in (1, 2)
                    if task not in decoder.primary_tasks or len(decoder.tasks) > 1
                )
                assert has_context_grad, "No gradients observed for any context key/value projections."

    def test_init_fails_on_task_overlap(self):
        """Tests that initialization fails if a task is in both primary and aux lists."""
        with pytest.raises(AssertionError, match="A task cannot be both primary and auxiliary"):
            HierarchicalContextAwareDecoder(
                encoder_channels=[16, 32, 64, 128], decoder_channel=64,
                num_classes={"task_a": 10},
                primary_tasks=["task_a"],
                aux_tasks=["task_a"],
            )

    def test_init_fails_on_missing_num_classes(self):
        """Tests that initialization fails if a task is missing from num_classes."""
        with pytest.raises(AssertionError, match="is defined in primary/aux_tasks but not in num_classes"):
            HierarchicalContextAwareDecoder(
                encoder_channels=[16, 32, 64, 128], decoder_channel=64,
                num_classes={"task_a": 10},
                primary_tasks=["task_a", "task_b"], # task_b is missing from num_classes
                aux_tasks=[],
            )