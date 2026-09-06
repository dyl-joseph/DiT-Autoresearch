#!/usr/bin/env python3
"""Scaffold the DiT AutoResearch skill into an existing target repository.

This deliberately does not overwrite project benchmark/evaluator scripts. It copies
control docs, the full knowledge handbook, and example configs for the human to edit.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_file(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        print(f"skip existing: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied: {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="path to target git repository")
    ap.add_argument("--force", action="store_true", help="overwrite existing SKILL/program/example config files")
    args = ap.parse_args()

    src_root = Path(__file__).resolve().parent.parent
    target = Path(args.target).resolve()
    if not (target / ".git").exists():
        raise SystemExit(f"target does not look like a git repository: {target}")

    copy_file(src_root / "SKILL.md", target / "SKILL.md", args.force)
    copy_file(src_root / "program.md", target / "program.md", args.force)
    copy_file(src_root / "config" / "autoresearch.example.json", target / "config" / "autoresearch.json", args.force)
    copy_file(src_root / "config" / "quality_contract.example.json", target / "config" / "quality_contract.json", args.force)

    ksrc = src_root / "knowledge"
    kdst = target / "knowledge"
    if kdst.exists() and args.force:
        shutil.rmtree(kdst)
    if not kdst.exists():
        shutil.copytree(ksrc, kdst)
        print(f"copied: {kdst}")
    else:
        print(f"skip existing knowledge dir: {kdst}")

    gitignore = target / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if ".autoresearch/" not in existing.splitlines():
        with gitignore.open("a", encoding="utf-8") as f:
            if existing and not existing.endswith("\n"):
                f.write("\n")
            f.write(".autoresearch/\n")
        print(f"updated: {gitignore}")

    print("\nNext steps:")
    print("1. Edit config/autoresearch.json mutable_paths/fixed_paths/commands.")
    print("2. Implement fixed perf + quality scripts using templates from the skill package.")
    print("3. Install this package: python -m pip install -e <skill-folder>")
    print("4. Create autoresearch/<tag> branch, then run dit-ar doctor/init/baseline.")


if __name__ == "__main__":
    main()
