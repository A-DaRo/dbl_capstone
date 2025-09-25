# Coral-MTL Project

This repository hosts the Coral-MTL multi-task learning experiments for coral reef segmentation. Models are configured via YAML files under `configs/` and orchestrated through `ExperimentFactory` in `src/coral_mtl`.

## Environment

Install the project in editable mode so package-style imports work:

```cmd
pip install -e .
```

The default `pytest.ini` now enables coverage reporting with `pytest-cov`. Ensure the tooling dependencies are installed:

```cmd
pip install -r requirements.txt
```

The test harness now clears `CUDA_VISIBLE_DEVICES` automatically so unit tests execute on CPU by default. To opt back into GPU execution set the variable explicitly before running commands:

```cmd
set CUDA_VISIBLE_DEVICES=0
```

## Running Tests with Coverage

### Full Suite

From the project root, run:

```cmd
pytest
```

The command automatically emits a terminal coverage report because of the `addopts` configuration.

### Targeted Suites

Use path selection to limit the scope, for example to the loss engine tests:

```cmd
pytest tests/coral_mtl_tests/engine/losses
```

### Custom Coverage Output

Additional coverage reports can be added via CLI flags. For example, to create an HTML report:

```cmd
pytest --cov-report=html
```

Reports will be written to the default `.coverage` data file plus any explicit report directories (e.g., `htmlcov/`).

## Next Steps

- See `project_specification/` for detailed design notes and metrics specifications.
- Use the configuration files under `configs/` to reproduce baseline runs.
