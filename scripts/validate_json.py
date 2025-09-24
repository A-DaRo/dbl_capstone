#!/usr/bin/env python3
"""Validate JSON files against provided schema.

Usage (CI): python scripts/validate_json.py <schema_path> <json_path>
Multiple files: python scripts/validate_json.py <schema_path> <glob>
"""
from __future__ import annotations
import sys
import json
import pathlib
import glob
from typing import List

try:
    import jsonschema
except ImportError:  # pragma: no cover
    print("jsonschema package required. Install with 'pip install jsonschema'.", file=sys.stderr)
    sys.exit(2)


def load(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate(schema_path: pathlib.Path, json_paths: List[pathlib.Path]) -> int:
    schema = load(schema_path)
    validator = jsonschema.Draft7Validator(schema)
    failures = 0
    for p in json_paths:
        try:
            data = load(p)
            errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
            if errors:
                print(f"FAIL {p}:")
                for e in errors:
                    loc = "/".join(str(x) for x in e.path)
                    print(f"  - {loc or '<root>'}: {e.message}")
                failures += 1
            else:
                print(f"OK   {p}")
        except Exception as e:  # pragma: no cover
            print(f"ERR  {p}: {e}")
            failures += 1
    return failures


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print("Usage: validate_json.py <schema> <json or glob> [more json paths]", file=sys.stderr)
        return 2
    schema = pathlib.Path(argv[1])
    if not schema.exists():
        print(f"Schema not found: {schema}", file=sys.stderr)
        return 2

    targets: List[pathlib.Path] = []
    for pattern in argv[2:]:
        matches = [pathlib.Path(p) for p in glob.glob(pattern)]
        if not matches:
            print(f"No match for pattern: {pattern}")
        targets.extend(matches)

    if not targets:
        print("No JSON files to validate")
        return 0

    failures = validate(schema, targets)
    if failures:
        print(f"Validation failed for {failures} file(s)")
        return 1
    print("All JSON files valid.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv))
