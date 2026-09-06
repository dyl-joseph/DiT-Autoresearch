from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config, ConfigError, load_config, load_quality_contract
from .gates import evaluate_performance, evaluate_quality
from .gitutils import (
    GitError,
    current_branch,
    hash_paths,
    head_commit,
    is_git_repo,
    parent_commit,
    short_commit,
    validate_last_commit_scope,
    working_tree_dirty,
)
from .hardware import detect_a5500, write_environment
from .knowledge import build_index, search_index
from .results import append_result, ensure_results, read_json, read_results, write_json
from .runner import format_command, require_metrics, run_command
from .schemas import get_metric


DEFAULT_CONFIG = "config/autoresearch.json"


def _cfg(args: argparse.Namespace) -> Config:
    return load_config(args.config)


def _state(cfg: Config) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "runs").mkdir(parents=True, exist_ok=True)
    ensure_results(cfg.results_tsv)


def _load_baseline(cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        read_json(cfg.state_dir / "baseline" / "perf.json"),
        read_json(cfg.state_dir / "baseline" / "quality.json"),
    )


def _load_best(cfg: Config) -> dict[str, Any]:
    p = cfg.state_dir / "best.json"
    if not p.exists():
        raise RuntimeError("No best.json. Run baseline first.")
    return read_json(p)


def cmd_index(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    _state(cfg)
    db = cfg.state_dir / "knowledge.sqlite3"
    n = build_index(cfg.knowledge_dir, db)
    print(f"Indexed {n} Markdown files -> {db}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    _state(cfg)
    db = cfg.state_dir / "knowledge.sqlite3"
    if not db.exists():
        build_index(cfg.knowledge_dir, db)
    rows = search_index(db, args.query, args.limit)
    if not rows:
        print("No matching knowledge entries.")
        return 1
    for i, row in enumerate(rows, 1):
        print(f"[{i}] {row['title']} — {row['path']}")
        print(f"    {str(row['snippet']).replace(chr(10), ' ')}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    _state(cfg)
    problems: list[str] = []
    warnings: list[str] = []

    if not is_git_repo(cfg.root):
        problems.append(f"project root is not a git repository: {cfg.root}")
    if working_tree_dirty(cfg.root):
        warnings.append("working tree is dirty; experiments should start from a clean commit")

    for rel in cfg.mutable_paths:
        if not (cfg.root / rel).exists():
            warnings.append(f"mutable path does not currently exist: {rel}")
    for rel in cfg.fixed_paths:
        if not (cfg.root / rel).exists():
            problems.append(f"fixed path missing: {rel}")

    env = write_environment(cfg.state_dir / "environment.json")
    strict_gpu = bool(cfg.target.get("require_a5500", True))
    if strict_gpu and not detect_a5500(env):
        warnings.append("RTX A5500 not detected in current environment; run doctor on the target machine before research")

    if shutil.which("nsys") is None:
        warnings.append("nsys not found; Nsight Systems profiling commands will not be available")
    if shutil.which("ncu") is None:
        warnings.append("ncu not found; Nsight Compute profiling commands will not be available")

    db = cfg.state_dir / "knowledge.sqlite3"
    if not db.exists():
        n = build_index(cfg.knowledge_dir, db)
        print(f"Knowledge index created ({n} Markdown files).")

    print(f"Project: {cfg.project_name}")
    print(f"Root: {cfg.root}")
    print(f"Branch: {current_branch(cfg.root) if is_git_repo(cfg.root) else '<none>'}")
    print(f"Knowledge: {cfg.knowledge_dir}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"- {w}")
    if problems:
        print("\nErrors:")
        for p in problems:
            print(f"- {p}")
        return 2
    print("\nDoctor: PASS")
    return 0


def _run_suite(cfg: Config, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    logs: list[dict[str, Any]] = []
    run_dir.mkdir(parents=True, exist_ok=True)
    write_environment(run_dir / "environment.json")

    for name in ("smoke", "perf", "quality"):
        template = cfg.commands.get(name)
        if not template:
            if name == "smoke":
                continue
            raise RuntimeError(f"missing required command: {name}")
        cmd = format_command(template, root=cfg.root, run_dir=run_dir)
        timeout = int(cfg.timeouts.get(name, 1800))
        result = run_command(cmd, cwd=cfg.root, log_path=run_dir / f"{name}.log", timeout_s=timeout)
        logs.append({
            "name": name,
            "command": cmd,
            "returncode": result.returncode,
            "elapsed_s": result.elapsed_s,
            "timed_out": result.timed_out,
            "log": str(result.log_path),
        })
        write_json(run_dir / "commands.json", logs)
        if result.returncode != 0:
            raise RuntimeError(f"{name} command failed ({result.returncode}); inspect {result.log_path}")

    perf = require_metrics(run_dir / "perf.json", "perf")
    quality = require_metrics(run_dir / "quality.json", "quality")
    return perf, quality, logs


def cmd_init(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    _state(cfg)
    if not is_git_repo(cfg.root):
        print("ERROR: project root must be a git repository", file=sys.stderr)
        return 2
    if working_tree_dirty(cfg.root):
        print("ERROR: clean/commit your working tree before init", file=sys.stderr)
        return 2

    branch = current_branch(cfg.root)
    if not branch.startswith("autoresearch/"):
        print(f"WARNING: recommended branch name is autoresearch/<tag>; current branch is {branch!r}")

    fixed = hash_paths(cfg.root, cfg.integrity_paths)
    write_json(cfg.state_dir / "fixed_hashes.json", fixed)
    write_json(cfg.state_dir / "session.json", {
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "start_commit": head_commit(cfg.root),
        "project": cfg.project_name,
    })
    db = cfg.state_dir / "knowledge.sqlite3"
    n = build_index(cfg.knowledge_dir, db)
    print(f"Initialized state in {cfg.state_dir}")
    print(f"Locked {len(fixed)} fixed-file hashes")
    print(f"Indexed {n} knowledge documents")
    print("Next: dit-ar baseline")
    return 0


def _check_fixed(cfg: Config) -> tuple[bool, list[str]]:
    p = cfg.state_dir / "fixed_hashes.json"
    if not p.exists():
        return False, ["fixed hash manifest missing; run dit-ar init"]
    old = read_json(p)
    new = hash_paths(cfg.root, cfg.integrity_paths)
    changed = sorted(set(old) | set(new))
    bad = [k for k in changed if old.get(k) != new.get(k)]
    return not bad, bad


def cmd_baseline(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    _state(cfg)
    ok, changed = _check_fixed(cfg)
    if not ok:
        print("ERROR: fixed harness changed: " + ", ".join(changed), file=sys.stderr)
        return 2
    if working_tree_dirty(cfg.root):
        print("ERROR: baseline requires a clean working tree", file=sys.stderr)
        return 2

    run_dir = cfg.state_dir / "baseline"
    if run_dir.exists() and not args.force:
        print("ERROR: baseline already exists; use --force only if intentionally rebuilding the reference", file=sys.stderr)
        return 2
    if run_dir.exists():
        shutil.rmtree(run_dir)
    perf, quality, logs = _run_suite(cfg, run_dir)
    metric = str(cfg.objective["metric"])
    val = get_metric(perf, metric)
    best = {
        "commit": head_commit(cfg.root),
        "perf": perf,
        "quality": quality,
        "objective_metric": metric,
        "objective_value": val,
        "run_dir": str(run_dir),
    }
    write_json(cfg.state_dir / "best.json", best)
    print(f"Baseline complete: {metric}={val:g}")
    print(f"Reference quality metrics: {json.dumps(quality.get('metrics', quality), sort_keys=True)}")
    return 0


def _classification_is_quantization(classification: str) -> bool:
    return classification.lower() in {"quantization", "quantized", "low-bit", "low_bit"}


def cmd_experiment(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    _state(cfg)
    ok, changed_fixed = _check_fixed(cfg)
    if not ok:
        print("DISCARD: fixed evaluation/benchmark harness was modified: " + ", ".join(changed_fixed))
        return 3
    if working_tree_dirty(cfg.root):
        print("ERROR: commit the candidate before running an experiment", file=sys.stderr)
        return 2
    scope_ok, bad_paths = validate_last_commit_scope(cfg.root, cfg.mutable_paths)
    if not scope_ok:
        print("DISCARD: candidate changed out-of-scope paths: " + ", ".join(bad_paths))
        return 3

    baseline_perf, baseline_quality = _load_baseline(cfg)
    best = _load_best(cfg)
    best_perf = best["perf"]

    commit = short_commit(cfg.root)
    parent = parent_commit(cfg.root)[:7]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.state_dir / "runs" / f"{stamp}_{commit}"

    status = "crash"
    quality_summary = "not-run"
    cand_val: float | str = ""
    best_val: float | str = ""
    imp_pct: float | str = ""
    peak_vram: float | str = ""
    try:
        perf, quality, logs = _run_suite(cfg, run_dir)
        contract = load_quality_contract(cfg)
        is_quant = _classification_is_quantization(args.classification)
        if is_quant and not bool(cfg.quantization.get("enabled_for_research", True)):
            q_decision = evaluate_quality(baseline_quality, quality, contract, strict_zero_regression=True)
            q_decision.passed = False
            q_decision.reasons.insert(0, "quantization research is disabled by config")
        else:
            strict = bool(cfg.quantization.get("strict_no_regression", True)) if is_quant else bool(contract.get("strict_zero_regression", False))
            q_decision = evaluate_quality(baseline_quality, quality, contract, strict_zero_regression=strict)
        p_decision = evaluate_performance(best_perf, perf, cfg.objective, cfg.guardrails)
        write_json(run_dir / "quality_gate.json", {"passed": q_decision.passed, "reasons": q_decision.reasons, "details": q_decision.details})
        write_json(run_dir / "performance_gate.json", {"passed": p_decision.passed, "reasons": p_decision.reasons, "details": p_decision.details})

        metric = str(cfg.objective["metric"])
        cand_val = get_metric(perf, metric)
        best_val = get_metric(best_perf, metric)
        direction = str(cfg.objective["direction"])
        raw_imp = (best_val - cand_val) if direction == "lower" else (cand_val - best_val)
        imp_pct = 100.0 * raw_imp / max(abs(best_val), 1e-12)
        try:
            peak_vram = get_metric(perf, "peak_vram_mb")
        except Exception:
            peak_vram = ""

        if not q_decision.passed:
            status = "discard-quality"
            quality_summary = "; ".join(q_decision.reasons)
        elif not p_decision.passed:
            status = "discard-performance"
            quality_summary = "pass"
        else:
            status = "keep"
            quality_summary = "pass"
            write_json(cfg.state_dir / "best.json", {
                "commit": head_commit(cfg.root),
                "perf": perf,
                "quality": quality,
                "objective_metric": metric,
                "objective_value": cand_val,
                "run_dir": str(run_dir),
            })

        print(status.upper())
        print(f"quality: {'PASS' if q_decision.passed else 'FAIL'}")
        if q_decision.reasons:
            for r in q_decision.reasons:
                print(f"  - {r}")
        print(f"performance: {'PASS' if p_decision.passed else 'FAIL'}")
        if p_decision.reasons:
            for r in p_decision.reasons:
                print(f"  - {r}")
        print(f"objective: {metric}: best={best_val:g} candidate={cand_val:g} improvement={float(imp_pct):.3f}%")
    except Exception as exc:
        status = "crash"
        quality_summary = repr(exc)
        print(f"CRASH: {exc}")

    append_result(cfg.results_tsv, {
        "commit": commit,
        "parent": parent,
        "status": status,
        "class": args.classification,
        "objective_metric": cfg.objective.get("metric", ""),
        "best_before": best_val,
        "candidate": cand_val,
        "improvement_pct": imp_pct,
        "peak_vram_mb": peak_vram,
        "quality": quality_summary,
        "description": args.description,
        "run_dir": str(run_dir),
    })
    return 0 if status == "keep" else 3


def cmd_summary(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    rows = read_results(cfg.results_tsv)
    if not rows:
        print("No experiments logged.")
        return 0
    print("commit\tstatus\tclass\tcandidate\timprovement_pct\tquality\tdescription")
    for row in rows[-args.limit:]:
        print("\t".join(str(row.get(k, "")) for k in ["commit", "status", "class", "candidate", "improvement_pct", "quality", "description"]))
    return 0


def cmd_profile_plan(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    run_dir = cfg.state_dir / "profiles" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    template = cfg.commands.get("profile")
    if template:
        cmd = format_command(template, root=cfg.root, run_dir=run_dir)
        print(cmd)
    else:
        print("No commands.profile configured. Recommended starting point:")
        print(f"nsys profile --trace=cuda,nvtx,osrt --sample=none -o {run_dir}/nsys <your one-case benchmark command>")
        print("Then rank GPU time by total contribution before using ncu on a narrow kernel family.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dit-ar", description="Autonomous DiT inference research harness")
    p.add_argument("--config", default=DEFAULT_CONFIG, help=f"config JSON (default: {DEFAULT_CONFIG})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="validate project, hardware, harness, and knowledge index")
    s.set_defaults(func=cmd_doctor)
    s = sub.add_parser("init", help="initialize immutable hashes and knowledge index")
    s.set_defaults(func=cmd_init)
    s = sub.add_parser("index", help="rebuild local Markdown knowledge index")
    s.set_defaults(func=cmd_index)
    s = sub.add_parser("search", help="search the compiled DiT optimization handbook")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=8)
    s.set_defaults(func=cmd_search)
    s = sub.add_parser("baseline", help="run and lock the reference performance+quality baseline")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_baseline)
    s = sub.add_parser("experiment", help="run fixed perf+quality suite for current committed candidate")
    s.add_argument("--description", required=True)
    s.add_argument("--classification", default="exact", choices=["exact", "numerical", "approximate", "quantization"])
    s.set_defaults(func=cmd_experiment)
    s = sub.add_parser("summary", help="print recent experiment outcomes")
    s.add_argument("--limit", type=int, default=30)
    s.set_defaults(func=cmd_summary)
    s = sub.add_parser("profile-plan", help="print configured or recommended profiler command")
    s.set_defaults(func=cmd_profile_plan)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, GitError, RuntimeError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
