---
title: Agent Chaining Guide
description: Practical workflow to chain Reformatter, Refactorer, Testing, and Orchestrator behaviors in Copilot chat + local scripts.
status: draft
---

# Agent Chaining Guide

This guide shows how to invoke the repository's multi‑agent lifecycle (Reformatter → Refactorer → Testing → Orchestrator) using:
1. Copilot Chat Modes & Prompts
2. Local orchestration scripts

---
## 1. Concepts
| Agent | Responsibility | Artifacts |
|-------|----------------|-----------|
| Reformatter | Normalize / derive instruction, prompt, chat mode files | `transformation_manifest.json` |
| Refactorer | 3-phase: Plan → Execute → Validate | `refactor_plan.md`, code diff, `dual_imports.json` (future) |
| Testing | Propose & run new tests | New/updated `tests/*.py`, coverage report |
| Orchestrator | Sequence + assemble PR assets | PR description text (manual), collected manifests |

---
## 2. Copilot Chat Flow (Manual)

### 2.1 Reformatter Phase
1. Open a markdown spec (e.g., `docs/AGENTS_SPEC.md`).
2. Switch Chat Mode to a neutral or system default; paste: 
   "Convert this document into instruction / prompt / chatmode assets following `.github/reformat_rules.yaml`. Respond only with a diff summary plus proposed filenames."
3. Save generated assets; run locally:
```
python scripts/run_reformatter.py
python scripts/validate_json.py schemas/transformation_manifest.schema.json transformation_manifest.json
```

### 2.2 Refactorer Phase (Phase 1 Plan)
1. Switch Chat Mode → `Refactor Chat Mode`.
2. Provide file inventory snippet or ask: "Generate Phase 1 plan for improving module boundaries around ExperimentFactory utilities." Include test file list.
3. Ensure final line is: `Confirm to proceed? (yes/no)`.
4. Review plan; if acceptable reply `yes`.

### 2.3 Refactorer Phase (Phase 2 Execute)
After confirmation:
- Apply only listed moves.
- Update imports; optional create `dual_imports.json`.
- Add new tests as flagged `(create)`.

### 2.4 Refactorer Phase (Phase 3 Validate)
Ask in the same chat mode: "Validate Phase 2 changes; list any missing tests or risk items." Apply suggestions.

### 2.5 Testing Agent
Switch Chat Mode → `Testing Chat Mode`.
Prompt example:
"Given current coverage gaps in ExperimentFactory (seed path fallback, optimizer schedule), propose 3 focused unit tests. Provide JSON array per testing_generate.prompt.md template."
Implement proposals, then:
```
pytest -m "not integration" -q
coverage run -m pytest -m "not integration" && coverage report -m
```

### 2.6 Orchestrator (Manual Assembly)
Compose PR description including:
- Refactor Plan table
- Summary of changes
- Coverage delta
- Manifest references

---
## 3. Local Script Orchestration (Hybrid)
A thin orchestrator can chain existing scripts (see future `scripts/agent_orchestrator.py`). Workflow:
```
python scripts/run_reformatter.py
python scripts/validate_refactor_plan.py  # after adding refactor_plan.md
pytest -q
```
Add more automation incrementally.

---
## 4. Quality Gates Checklist
| Gate | Script / Action | Pass Criterion |
|------|-----------------|----------------|
| Frontmatter | `validate_frontmatter.py` | 0 errors |
| Reformat | `run_reformatter.py` | Idempotent second run |
| Refactor Plan | `validate_refactor_plan.py` | All headings + question present |
| Tests | `pytest` | Exit 0 |
| Coverage | `coverage report` | (Optional) threshold |

---
## 5. Incremental Enhancements
- Auto-generate `dual_imports.json` from git diff.
- Add coverage fail-under.
- AST digest caching for stable refactor plan diffs.
- Danger-style PR comment summarizer.

---
## 6. Minimal Command Sequence
```
python scripts/run_reformatter.py
python scripts/validate_refactor_plan.py  # if refactor touches src/
pytest -q
```

---
## 7. FAQ
**Q:** Why is confirmation mandatory?  
**A:** It enforces human review of structural changes before any irreversible file moves.

**Q:** When do I regenerate the plan?  
**A:** Any time you add new target modules or expand scope—add a revision section instead of overwriting silently.

**Q:** Do prompts need titles?  
**A:** Yes; enforced by frontmatter validation for discoverability.

---
Maintainers: update this guide when adding new agents or automation layers.
