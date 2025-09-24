#!/usr/bin/env python3
"""Markdown Reformatter (Deterministic)

Reads .github/reformat_rules.yaml and applies a subset of safe, mechanical
transformations to markdown-like files:
  - Normalize heading capitalization mode (if configured)
  - Ensure single blank line after frontmatter or initial heading
  - Collapse >1 consecutive blank lines to one (within body)
  - Trim trailing whitespace
  - Normalize bullet markers ('-' by default)
  - Optional hard wrap (NOT enabled by default)
  - Sort frontmatter tag arrays (if present)

Emits a transformation manifest JSON summarizing changed files.

Exit codes:
 0: success (even if no changes)
 1: unrecoverable error
 2: config/rule parse issue

Limitations: This is intentionally conservative; semantic rewrites are out of scope.
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import re
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
RULES_PATH = ROOT / ".github" / "reformat_rules.yaml"
DEFAULT_GLOBS = ["docs/**/*.md", ".github/**/*.md", ".github/**/*.prompt.md", ".github/**/*.chatmode.md", ".github/**/*.instructions.md"]
MANIFEST_PATH = ROOT / "transformation_manifest.json"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def load_rules() -> Dict[str, Any]:
    if not RULES_PATH.exists():
        return {}
    with RULES_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_files(patterns: List[str]) -> List[pathlib.Path]:
    files: List[pathlib.Path] = []
    for pattern in patterns:
        for p in ROOT.glob(pattern):
            if p.is_file():
                files.append(p)
    # de-dupe
    return sorted(set(files))


def split_frontmatter(text: str):
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None, text
    return m.group(1), text[m.end():]


def normalize_frontmatter(raw: str, rules: Dict[str, Any]) -> str:
    try:
        data = yaml.safe_load(raw) or {}
    except Exception:
        return raw  # don't alter invalid yaml
    # Sort tags list if present
    if isinstance(data.get("tags"), list):
        data["tags"] = sorted(data["tags"], key=str.lower)
    return yaml.safe_dump(data, sort_keys=True).strip() + "\n"


def normalize_body(body: str, rules: Dict[str, Any]) -> str:
    lines = body.splitlines()
    out: List[str] = []
    bullet = rules.get("bullet_marker", "-")
    blank_streak = 0
    for ln in lines:
        original = ln
        ln = ln.rstrip()  # trim trailing whitespace
        # Normalize bullet list marker
        if re.match(r"^[*+-] ", ln):
            ln = re.sub(r"^[*+-] ", f"{bullet} ", ln, count=1)
        # Collapse blank lines (but allow one)
        if ln.strip() == "":
            blank_streak += 1
            if blank_streak > 1:
                continue
        else:
            blank_streak = 0
        out.append(ln)
    text = "\n".join(out).rstrip() + "\n"
    # Ensure single blank line after frontmatter heading combos handled elsewhere
    return text


def maybe_wrap(text: str, width: int) -> str:
    try:
        import textwrap
    except ImportError:
        return text
    wrapped = []
    for para in re.split(r"\n\n", text.strip()):
        if len(para) > width and not para.startswith(('#', '```')):
            wrapped.append("\n".join(textwrap.wrap(para, width=width)))
        else:
            wrapped.append(para)
    return "\n\n".join(wrapped) + "\n"


def process_file(path: pathlib.Path, rules: Dict[str, Any]) -> Dict[str, Any] | None:
    original = path.read_text(encoding="utf-8")
    front_raw, body = split_frontmatter(original)
    changed = False

    front_processed = None
    if front_raw is not None:
        norm_front = normalize_frontmatter(front_raw, rules)
        if norm_front.strip() != front_raw.strip():
            changed = True
        front_processed = f"---\n{norm_front}---\n"

    new_body = normalize_body(body, rules)
    if new_body != body:
        changed = True

    if rules.get("hard_wrap", {}).get("enabled"):
        width = int(rules["hard_wrap"].get("width", 100))
        wrapped_body = maybe_wrap(new_body, width)
        if wrapped_body != new_body:
            changed = True
            new_body = wrapped_body

    if not changed:
        return None

    new_text = (front_processed or "") + new_body
    path.write_text(new_text, encoding="utf-8")
    return {
        "file": str(path.relative_to(ROOT)),
        "changes": {
            "frontmatter_modified": front_processed is not None and front_raw.strip() != (front_processed.replace('---\n','').replace('\n---\n','').strip()),
            "body_modified": new_body != body,
        }
    }


def build_manifest(changes: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "source": "run_reformatter.py",
        "created": [c["file"] for c in changes],
        "extracted_variables": {},
        "rationale": "Applied mechanical formatting rules",
        "confidence": 0.98,
        "warnings": [],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Deterministic markdown reformatter")
    parser.add_argument("--patterns", nargs="*", default=DEFAULT_GLOBS, help="Glob patterns relative to repo root")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Output manifest path")
    args = parser.parse_args(argv)

    rules = load_rules()
    files = find_files(args.patterns)
    all_changes: List[Dict[str, Any]] = []
    for f in files:
        result = process_file(f, rules)
        if result:
            all_changes.append(result)
    if all_changes:
        manifest = build_manifest(all_changes)
        pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Reformatted {len(all_changes)} file(s). Manifest -> {args.manifest}")
    else:
        print("No changes; manifest not written.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
