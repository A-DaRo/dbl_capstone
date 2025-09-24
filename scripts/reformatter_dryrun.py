#!/usr/bin/env python3
"""Stub MD Reformatter dry-run script.

Current behavior:
- Scans repository for markdown (*.md) under docs/ and .github/ (excluding workflows) and reports a summary.
- Placeholder for future deterministic reformat preview using `.github/reformat_rules.yaml`.
Exit codes:
  0 success
  1 unexpected error
"""
from __future__ import annotations
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET_DIRS = [ROOT / "docs", ROOT / ".github"]


def gather_markdown():
    for base in TARGET_DIRS:
        if not base.exists():
            continue
        for p in base.rglob("*.md"):
            # skip workflow yaml docs, only markdown
            yield p


def main() -> int:
    files = list(gather_markdown())
    print("[md-reformatter-dryrun] Detected markdown files:")
    for f in files:
        print(f" - {f.relative_to(ROOT)}")
    print(f"Total: {len(files)} file(s). (No transformations applied in stub)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(1)
