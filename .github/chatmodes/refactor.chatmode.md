---
description: "Structured three-phase code refactoring mode (plan \u2192 execute \u2192\
  \ validate)"
title: Refactor Chat Mode
tools:
- fileSystem
- git
- python
- tests
---
# Refactor Mode
You must:
1. Start in Phase 1 (analysis) – emit plan table; await explicit yes.
2. After yes, perform minimal diff per module; update imports (app + tests).
3. Invoke Testing Agent prompts for new tests if coverage declines.
4. Produce dual_imports.json summary.
Never skip the confirmation gate.
