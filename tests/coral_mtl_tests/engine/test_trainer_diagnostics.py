import torch
import pytest
from types import SimpleNamespace

from coral_mtl.engine.trainer import Trainer


class RecordingStorer:
    def __init__(self):
        self.records = []

    def open_for_run(self, is_testing: bool = False) -> None:  # pragma: no cover - noop
        pass

    def close(self) -> None:  # pragma: no cover - noop
        pass

    def store_loss_diagnostics(self, *, step: int, epoch: int, diagnostics):
        self.records.append({
            "step": step,
            "epoch": epoch,
            "diagnostics": diagnostics.copy(),
        })


class DummyMetricsCalculator:
    def reset(self) -> None:  # pragma: no cover - noop
        pass

    def update(self, **kwargs) -> None:  # pragma: no cover - noop
        pass

    def compute(self):  # pragma: no cover - minimal stub
        return {}


class DummyWeightingStrategy:
    def __init__(self):
        self.tasks = ["genus", "health"]
        self._latest_losses = {}
        self._latest_diagnostics = {}

    def requires_manual_backward_update(self) -> bool:
        return False

    def get_last_diagnostics(self):
        return dict(self._latest_diagnostics)


@pytest.fixture
def trainer_stub(tmp_path):
    model = torch.nn.Linear(4, 4)
    train_loader = []
    val_loader = []
    strategy = DummyWeightingStrategy()
    loss_fn = SimpleNamespace(weighting_strategy=strategy)
    metrics_calculator = DummyMetricsCalculator()
    storer = RecordingStorer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    config = SimpleNamespace(
        device="cpu",
        output_dir=str(tmp_path),
        use_mixed_precision=False,
        gradient_accumulation_steps=1,
        log_frequency=2,
        pcgrad=None,
        patch_size=(8, 8),
        inference_stride=(4, 4),
        inference_batch_size=1,
        epochs=1,
        model_selection_metric="optimization_metrics.H-Mean",
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics_calculator=metrics_calculator,
        metrics_storer=storer,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    return trainer, storer, strategy


def test_compute_gradient_diagnostics(trainer_stub):
    trainer, _, _ = trainer_stub
    grads = {
        "genus": torch.tensor([1.0, 0.0]),
        "health": torch.tensor([0.0, 2.0]),
    }
    norms, cosine = trainer._compute_gradient_diagnostics(grads)
    assert norms["genus"] == pytest.approx(1.0)
    assert norms["health"] == pytest.approx(2.0)
    assert cosine["genus_vs_health"] == pytest.approx(0.0)


def test_extract_weighting_diagnostics(trainer_stub):
    trainer, _, strategy = trainer_stub
    strategy._latest_diagnostics = {
        "task_weights": {"genus": 0.5, "health": 0.25},
        "log_variances": {"genus": 0.1},
    }
    diagnostics = trainer._extract_weighting_diagnostics()
    assert diagnostics["task_weights"]["genus"] == pytest.approx(0.5)
    assert diagnostics["task_weights"]["health"] == pytest.approx(0.25)


def test_maybe_record_loss_diagnostics_respects_frequency(trainer_stub):
    trainer, storer, _ = trainer_stub
    trainer.global_step = 1
    trainer._maybe_record_loss_diagnostics(epoch_index=1, diagnostics={"foo": "bar"})
    assert storer.records == []

    trainer.global_step = 2
    trainer._maybe_record_loss_diagnostics(epoch_index=1, diagnostics={"foo": "bar"})
    assert len(storer.records) == 1
    record = storer.records[0]
    assert record["step"] == 2
    assert record["epoch"] == 1
    assert record["diagnostics"]["strategy_type"] == "DummyWeightingStrategy"