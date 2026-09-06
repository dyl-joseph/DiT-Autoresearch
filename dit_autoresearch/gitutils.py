from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Iterable


class GitError(RuntimeError):
    pass


def run_git(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        raise GitError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def is_git_repo(root: Path) -> bool:
    try:
        return run_git(root, "rev-parse", "--is-inside-work-tree") == "true"
    except Exception:
        return False


def head_commit(root: Path) -> str:
    return run_git(root, "rev-parse", "HEAD")


def short_commit(root: Path) -> str:
    return run_git(root, "rev-parse", "--short=7", "HEAD")


def parent_commit(root: Path) -> str:
    return run_git(root, "rev-parse", "HEAD^")


def current_branch(root: Path) -> str:
    return run_git(root, "branch", "--show-current")


def changed_paths_last_commit(root: Path) -> list[str]:
    out = run_git(root, "diff", "--name-only", "HEAD^", "HEAD")
    return [line.strip() for line in out.splitlines() if line.strip()]


def working_tree_dirty(root: Path) -> bool:
    return bool(run_git(root, "status", "--porcelain"))


def path_allowed(path: str, allowed_roots: Iterable[str]) -> bool:
    norm = path.replace("\\", "/").lstrip("./")
    for root in allowed_roots:
        r = str(root).replace("\\", "/").strip("/")
        if norm == r or norm.startswith(r + "/"):
            return True
    return False


def validate_last_commit_scope(root: Path, mutable_paths: list[str]) -> tuple[bool, list[str]]:
    changed = changed_paths_last_commit(root)
    bad = [p for p in changed if not path_allowed(p, mutable_paths)]
    return not bad, bad


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_paths(root: Path, paths: Iterable[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    for rel in paths:
        p = (root / rel).resolve()
        if not p.exists():
            entries[rel] = "<missing>"
            continue
        if p.is_file():
            entries[rel] = _hash_file(p)
            continue
        for child in sorted(x for x in p.rglob("*") if x.is_file()):
            key = os.path.relpath(child, root).replace(os.sep, "/")
            entries[key] = _hash_file(child)
    return entries
