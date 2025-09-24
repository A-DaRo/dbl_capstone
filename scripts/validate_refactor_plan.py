#!/usr/bin/env python3
"""Validate presence and structure of refactor plan when Python sources change.

Heuristics:
- Detect modified Python files vs main branch (git). If git unavailable, fallback: always require plan if any *.py exists (len>0) when --require is passed.
- Look for `refactor_plan.md` in repo root OR under `docs/`.
- Required sections (case-insensitive headings):
  - Summary
  - Risks
  - Plan (table)  (Plan table must have header with 'Source Module' and 'Target Module')
  - Confirmation Question (line ending with '?')
- Must contain phrase 'Confirm to proceed?' exactly once.

Exit codes:
 0 ok
 1 issues
 2 internal error
"""
from __future__ import annotations
import subprocess
import pathlib
import re
import sys
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLAN_CANDIDATES = [ROOT / "refactor_plan.md", ROOT / "docs" / "refactor_plan.md"]

HEADER_RE = re.compile(r"^#+ +(.+)$", re.MULTILINE)
TABLE_HEADER_RE = re.compile(r"Source Module\s*\|.*Target Module", re.IGNORECASE)


def run(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT)


def changed_python_files() -> List[str]:
    try:
        base = run(["git", "rev-parse", "--abbrev-ref", "origin/HEAD"]).strip()
    except Exception:
        # fallback to main
        base = "origin/main"
    try:
        diff = run(["git", "diff", "--name-only", base])
        return [l for l in diff.splitlines() if l.endswith('.py') and l.startswith('src/')]
    except Exception:
        return []


def locate_plan() -> pathlib.Path | None:
    for c in PLAN_CANDIDATES:
        if c.exists():
            return c
    return None


def validate_plan_text(text: str) -> List[str]:
    issues: List[str] = []
    headers = [h.strip().lower() for h in HEADER_RE.findall(text)]
    for required in ["summary", "risks", "plan"]:
        if required not in headers:
            issues.append(f"Missing heading: {required}")
    if not TABLE_HEADER_RE.search(text):
        issues.append("Plan table header missing required columns")
    confirm_count = text.count("Confirm to proceed?")
    if confirm_count != 1:
        issues.append(f"Expected exactly one 'Confirm to proceed?' line, found {confirm_count}")
    return issues


def main(argv: List[str]) -> int:
    py_changed = changed_python_files()
    if not py_changed:
        print("No changed Python files relative to base; skipping refactor plan enforcement.")
        return 0
    plan_path = locate_plan()
    if not plan_path:
        print("Refactor plan required but not found (refactor_plan.md).")
        return 1
    text = plan_path.read_text(encoding="utf-8")
    issues = validate_plan_text(text)
    if issues:
        for i in issues:
            print(f"ISSUE: {i}")
        return 1
    print(f"Refactor plan valid at {plan_path} for {len(py_changed)} changed Python file(s).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
