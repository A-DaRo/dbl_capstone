#!/usr/bin/env python3
"""Lightweight Agent Orchestrator Stub

Sequences existing local steps for the multi-agent lifecycle.

Phases:
 1. (Optional) Reformatter
 2. Refactor Plan Validation (only if Python changes detected)
 3. Tests (+ optional coverage)

Options:
  --no-reformat              Skip markdown reformatter
  --coverage                 Run coverage after tests
  --fail-under PERCENT       Enforce coverage threshold (implies --coverage)
  --pattern GLOB             Additional glob(s) for markdown reformatter (repeatable)

Outputs:
  - transformation_manifest.json (if reformat produced changes)
  - stdout summaries
Exit codes:
  0 success
  1 failure in any stage
"""
from __future__ import annotations
import argparse
import pathlib
import subprocess
import sys
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent.parent


def run(cmd: List[str], check: bool = True) -> int:
    print(f"[orchestrator] RUN: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT)
    if check and proc.returncode != 0:
        print(f"[orchestrator] Command failed with {proc.returncode}: {' '.join(cmd)}")
        return proc.returncode
    return proc.returncode


def changed_python_files() -> bool:
    try:
        base = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT).decode().strip()
    except Exception:
        return True  # fallback: assume changed to be safe
    try:
        diff = subprocess.check_output(["git", "diff", "--name-only", "origin/main"], cwd=ROOT).decode()
        return any(l.endswith('.py') and l.startswith('src/') for l in diff.splitlines())
    except Exception:
        return True


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Agent orchestrator stub")
    parser.add_argument("--no-reformat", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--fail-under", type=float, default=None)
    parser.add_argument("--pattern", action="append", default=[])
    args = parser.parse_args(argv)

    if args.fail_under is not None:
        args.coverage = True

    # Phase 1: Reformatter
    if not args.no_reformat:
        patterns = [p for p in args.pattern] if args.pattern else []
        cmd = [sys.executable, "scripts/run_reformatter.py"] + (["--patterns"] + patterns if patterns else [])
        if run(cmd) != 0:
            return 1

    # Phase 2: Refactor Plan Validation (only if python changed)
    if changed_python_files():
        if run([sys.executable, "scripts/validate_refactor_plan.py"], check=False) != 0:
            print("[orchestrator] Refactor plan invalid or missing.")
            return 1
    else:
        print("[orchestrator] No Python changes; skipping plan validation.")

    # Phase 3: Tests / Coverage
    if args.coverage:
        if run([sys.executable, "-m", "coverage", "run", "-m", "pytest", "-m", "not integration"]) != 0:
            return 1
        if run([sys.executable, "-m", "coverage", "report", "-m"]) != 0:
            return 1
        if args.fail_under is not None:
            # parse coverage summary
            try:
                out = subprocess.check_output([sys.executable, "-m", "coverage", "report"], cwd=ROOT).decode()
                # last line like: TOTAL   1234  56   10  5  91%
                for line in out.splitlines():
                    if line.startswith("TOTAL"):
                        pct = float(line.strip().split()[-1].strip('%'))
                        if pct < args.fail_under:
                            print(f"Coverage {pct}% < threshold {args.fail_under}%")
                            return 1
                        break
            except Exception as e:
                print(f"[orchestrator] Failed to enforce coverage threshold: {e}")
                return 1
    else:
        if run([sys.executable, "-m", "pytest", "-m", "not integration"]) != 0:
            return 1

    print("[orchestrator] Pipeline complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
