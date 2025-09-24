---
applyTo: '**/*.py'
description: Core Python code style & structure rules
title: Python Code Style
---
- Use Python 3.11+ features cautiously; prefer explicit over implicit.
- Four-space indentation; double quotes for strings unless a single quote avoids escaping.
- Keep functions < 60 lines; extract helpers inside module when exceeding.
- Public nn.Module classes go under `src/coral_mtl/<domain>/` following existing patterns.
- Never hard-code class counts; always derive via the appropriate TaskSplitter.
- Prefer pure functions for transformations; stateful objects only for datasets, models, processors.
- Log warnings (not errors) for optional dependency absence.
- Avoid network calls in unit tests; mark integration scenarios with @pytest.mark.integration.
- Use torch.device from factory config; do not re-detect GPU in lower layers.
- Maintain deterministic seeds in tests: torch, numpy, random.
