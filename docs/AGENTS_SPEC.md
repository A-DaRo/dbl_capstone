---
description: Canonical multi-agent orchestration, roles, interfaces, and lifecycle
  for Copilot customization.
status: stable
tags:
- agents
- governance
- refactor
title: Agents Specification
---

# Coral-MTL Copilot Agents Specification

(This document is derived from the formal spec provided in chat and normalized for repository inclusion. It defines the contracts for the MD Reformatter, Code Refactorer, Testing Agent, and Orchestrator.)

## 1. Purpose & Scope
Transform human-authored `.md` specs into consistent Copilot customization artifacts and safe, validated code changes (Python 3.11+). Pipeline spans transform → plan → apply → test → report.

## 2. Components Overview
- Global instructions file: `.github/copilot-instructions.md` (already present) – concise architectural + workflow guidance.
- Agents:
  1. MD Reformatter Agent
  2. Code Refactorer Agent
  3. Testing Agent
  4. Orchestrator Agent

## 3. File Types & Canonical Formats
| Type | Location | Required Frontmatter | Key Notes |
|------|----------|----------------------|-----------|
| Repo Instructions | `.github/copilot-instructions.md` | None | Short, directive guidance referenced by other agents |
| Scoped Instructions | `.github/instructions/*.instructions.md` | `description`, `applyTo` | Bulleted imperatives; `applyTo` glob (default `**/*.py`) |
| Prompt Templates | `.github/prompts/*.prompt.md` | `mode`, `description` | May include `${input:var}` placeholders |
| Chat Modes | `.github/chatmodes/*.chatmode.md` | `description` | Defines persona + allowed tools |
| Agents Spec | `docs/AGENTS_SPEC.md` | (Optional) | Human + machine reference |
| Agent Manifests | `copilot-agents/agent-manifests/*.yaml` | N/A | Declares inputs, outputs, permissions |

## 4. Agent Contracts
### 4.1 MD Reformatter Agent
Inputs: Markdown file(s) + optional `reformat_rules.yaml`.
Outputs: one or more `.instructions.md`, `.prompt.md`, `.chatmode.md` plus JSON Transformation Manifest.
Invariants:
- Every prompt has `mode` + `description`.
- Each instructions file has `applyTo` (default `**/*.py`).
Failure: produce `*.candidate.md` when ambiguous.

### 4.2 Code Refactorer Agent
Phases (MUST confirm between 1→2):
1. Analysis & Plan → Table mapping Source Module → Test → Target.
2. Execution → Apply moves, dual import resolution, migrate / create tests.
3. Validate & Report → Lint + run Testing Agent.
Invariants: No lost tests; every moved symbol has updated imports in app & tests.

### 4.3 Testing Agent
Responsibilities: generate/update unit & integration tests, run pytest, compute coverage, identify flaky tests, run adversarial prompt safety checks.
Outputs: `pytest` report, `coverage.xml`, flakiness log, safety log.
Policies: Unit tests deterministic; integration tests tagged `@pytest.mark.integration`.

### 4.4 Orchestrator Agent
Sequences pipeline: Reformatter → Refactorer (Phase 1 pause) → Testing → PR packaging.
Outputs: PR metadata (Refactor Plan, test summary, coverage deltas, manifests).
Never auto-merges.

## 5. Pipeline DAG
```
Markdown / PR Trigger
  → MD Reformatter
     → Code Refactorer (Phase 1 plan → confirm → Phase 2 execute)
        → Testing Agent
           → Orchestrator (assemble PR artifacts)
```

## 6. Verification & Acceptance
Required artifacts when code changes occur:
1. Refactoring Plan (explicit user confirmation).
2. Patchset diff.
3. Test report: all unit tests pass; coverage drop ≤ 5%.
4. Dual import manifest proving all import paths updated.

## 7. Governance
- Code owners review for `.instructions.md`, `.prompt.md`, test changes.
- Smaller, atomic commits encouraged; large refactors splitted by module.
- `.github/copilot-instructions.md` treated as code – PR required.

## 8. Role Document Bindings
| Source Doc | Binding |
|------------|---------|
| `Code_Refactorer.md` | Enforces 3-phase, confirmation gate, dual import resolution |
| `Project_Coders.md` | Deterministic unit tests, spec-anchored implementation |
| `Project_developer.md` | Spec-first justification and task-based increments |

## 9. Implementation Notes
- Add provenance comments referencing spec sections for generated code.
- Provide `dual_imports.json` in PR summarizing updated import graph.
- Red-team prompts against secret exfiltration & unsafe ops.

## 10. Future Enhancements
- Add CI workflow to auto-run Reformatter on changed `.md` in `docs/`.
- Introduce risk scoring for refactors (import fanout, cyclomatic delta).
- Cache AST digests to accelerate iterative planning.

---
Maintainers: please update this spec alongside any agent behavior changes.
