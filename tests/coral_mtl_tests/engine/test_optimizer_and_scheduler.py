# tests/coral_mtl/engine/test_optimizer_and_scheduler.py

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from coral_mtl.engine.optimizer import create_optimizer_and_scheduler


@pytest.fixture
def dummy_model() -> nn.Module:
    """Provides a simple model with different parameter types for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),  # Has 'weight' (2D) and 'bias' (1D)
        nn.LayerNorm(20),  # Has 'weight' (1D) and 'bias' (1D)
        nn.Conv2d(3, 8, kernel_size=3),  # Has 'weight' (>1D) and 'bias' (1D)
    )


@pytest.mark.gpu
class TestCreateOptimizerAndScheduler:
    """Test suite for the create_optimizer_and_scheduler function."""

    def test_return_types(self, dummy_model):
        """
        Verifies that the function returns the correct types: an AdamW optimizer
        and a LambdaLR scheduler from the transformers library.
        """
        optimizer, scheduler = create_optimizer_and_scheduler(model=dummy_model)

        assert isinstance(optimizer, AdamW), "Optimizer should be an instance of AdamW."
        assert isinstance(
            scheduler, LambdaLR
        ), "Scheduler should be a LambdaLR instance."

    def test_parameter_grouping_for_weight_decay(self, dummy_model):
        """
        Verifies the core logic of splitting parameters into two groups:
        - Group 1: Parameters with weight decay (e.g., Linear weights).
        - Group 2: Parameters without weight decay (biases, 1D params like LayerNorm weights).
        """
        weight_decay = 0.05
        optimizer, _ = create_optimizer_and_scheduler(
            model=dummy_model, weight_decay=weight_decay
        )

        param_groups = optimizer.param_groups
        assert len(param_groups) == 2, "Optimizer should have two parameter groups."

        # Group 0 should have weight decay
        assert param_groups[0]["weight_decay"] == weight_decay
        # Group 1 should have no weight decay
        assert param_groups[1]["weight_decay"] == 0.0

        decay_param_names = [
            name for name, _ in dummy_model.named_parameters() if name.endswith(".weight") and name != "1.weight"
        ]
        no_decay_param_names = [
            name for name, _ in dummy_model.named_parameters() if name.endswith(".bias") or name == "1.weight"
        ]

        # Check if parameters are in the correct group
        decay_group_params = {id(p) for p in param_groups[0]["params"]}
        no_decay_group_params = {id(p) for p in param_groups[1]["params"]}

        for name, param in dummy_model.named_parameters():
            if name in decay_param_names:
                assert id(param) in decay_group_params, f"'{name}' should be in decay group."
                assert id(param) not in no_decay_group_params
            elif name in no_decay_param_names:
                assert id(param) in no_decay_group_params, f"'{name}' should be in no-decay group."
                assert id(param) not in decay_group_params

    def test_scheduler_lr_progression(self, dummy_model):
        """
        Verifies the learning rate schedule by simulating a full training run.
        It checks for a correct warmup phase followed by a polynomial decay phase.
        """
        lr = 1e-4
        total_steps = 100
        warmup_steps = 10
        lr_end = 1e-7

        optimizer, scheduler = create_optimizer_and_scheduler(
            model=dummy_model,
            learning_rate=lr,
            num_training_steps=total_steps,
            num_warmup_steps=warmup_steps,
        )

        lrs = []
        for _ in range(total_steps):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()  # Dummy step
            scheduler.step()

        # 1. Warmup Phase Verification
        assert lrs[0] < lrs[1], "LR should increase during warmup."
        assert lrs[warmup_steps - 2] < lrs[warmup_steps - 1], "LR should increase until end of warmup."
        # The transformers scheduler reaches peak at step warmup_steps (0-indexed), not warmup_steps - 1
        assert pytest.approx(lrs[warmup_steps], abs=1e-6) == lr, "LR should reach peak right after warmup."

        # 2. Decay Phase Verification
        assert lrs[warmup_steps + 1] < lr, "LR should start decaying after warmup."
        assert lrs[warmup_steps + 1] > lrs[warmup_steps + 2], "LR should decrease during decay phase."
        
        # 3. Final LR Verification
        # The polynomial decay doesn't necessarily reach exactly lr_end, but should be close
        assert lrs[-1] < 1e-5, "Final LR should be very small."
        assert lrs[-1] >= lr_end, "Final LR should be at least lr_end."

    def test_no_warmup_scenario(self, dummy_model):
        """
        Tests the edge case where num_warmup_steps is 0. The LR should start at
        its peak value and begin decaying immediately.
        """
        lr = 1e-4
        total_steps = 100

        optimizer, scheduler = create_optimizer_and_scheduler(
            model=dummy_model,
            learning_rate=lr,
            num_training_steps=total_steps,
            num_warmup_steps=0,
        )

        # The first LR should be the peak learning rate
        assert optimizer.param_groups[0]["lr"] == lr, "With no warmup, LR should start at peak."

        # It should decay on the first step
        initial_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]
        assert next_lr < initial_lr, "LR should decay immediately with no warmup."

    def test_model_with_no_parameters(self):
        """

        Tests the edge case of a model with no trainable parameters. The function
        should execute without errors.
        """
        model = nn.Sequential()  # An empty model
        try:
            optimizer, scheduler = create_optimizer_and_scheduler(model=model)
            # The optimizer should be created with empty parameter groups
            assert len(optimizer.param_groups) == 2
            assert len(optimizer.param_groups[0]["params"]) == 0
            assert len(optimizer.param_groups[1]["params"]) == 0
        except Exception as e:
            pytest.fail(f"Function failed with an empty model: {e}")