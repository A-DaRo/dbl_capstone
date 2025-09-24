#!/usr/bin/env python3
"""Validate markdown frontmatter blocks across docs/ and .github/.

Rules (initial minimal implementation):
- Frontmatter (YAML) must start at first line with '---' and end with '---'.
- Must include at least: title
- Optional recommended keys: description, tags (list), status.
- Disallow tabs in frontmatter region.
Exit codes:
 0 success, 1 validation errors, 2 internal error.
"""
from __future__ import annotations
import sys
import pathlib
import re
import yaml
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET_DIRS = [ROOT / "docs", ROOT / ".github"]

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
REQUIRED_KEYS = {"title"}


def extract_frontmatter(text: str):
    match = FRONTMATTER_RE.match(text)
    if not match:
        return None, text
    return match.group(1), text[match.end():]


def validate_file(path: pathlib.Path) -> List[str]:
    errors: List[str] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:  # pragma: no cover
        return [f"{path}: unreadable ({e})"]

    front, _ = extract_frontmatter(raw)
    if front is None:
        errors.append(f"{path}: missing frontmatter")
        return errors

    if "\t" in front:
        errors.append(f"{path}: tab character found in frontmatter")

    try:
        data = yaml.safe_load(front) or {}
    except yaml.YAMLError as e:
        errors.append(f"{path}: YAML parse error: {e}")
        return errors

    missing = REQUIRED_KEYS - set(data)
    if missing:
        errors.append(f"{path}: missing required keys: {', '.join(sorted(missing))}")

    if "tags" in data and not isinstance(data["tags"], list):
        errors.append(f"{path}: 'tags' must be a list if present")

    return errors


def gather_markdown_files():
    for base in TARGET_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*.md"):
            yield path


def main():
    all_errors: List[str] = []
    for md in gather_markdown_files():
        all_errors.extend(validate_file(md))

    if all_errors:
        for e in all_errors:
            print(e)
        print(f"FAILED: {len(all_errors)} frontmatter issues detected.")
        return 1

    print("Frontmatter validation passed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
