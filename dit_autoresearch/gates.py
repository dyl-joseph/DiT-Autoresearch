from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import get_metric


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _regression(
    baseline: float,
    candidate: float,
    direction: str,
) -> tuple[float, float]:
    """Return absolute and relative regression, both >=0 when worse."""
    if direction == "higher":
        abs_reg = max(0.0, baseline - candidate)
    elif direction == "lower":
        abs_reg = max(0.0, candidate - baseline)
    else:
        raise ValueError(f"unknown direction: {direction}")
    rel_reg = abs_reg / max(abs(baseline), 1e-12)
    return abs_reg, rel_reg


def evaluate_quality(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    contract: dict[str, Any],
    *,
    strict_zero_regression: bool = False,
) -> GateDecision:
    reasons: list[str] = []
    details: dict[str, Any] = {"metrics": {}, "cases": {}}
    metric_specs = contract.get("metrics", {})
    if not metric_specs:
        return GateDecision(False, ["quality contract contains no metrics"], details)

    base_metrics = baseline.get("metrics", baseline)
    cand_metrics = candidate.get("metrics", candidate)

    for name, spec in metric_specs.items():
        if not isinstance(spec, dict):
            spec = {}
        if name not in base_metrics:
            reasons.append(f"baseline quality metric missing: {name}")
            continue
        if name not in cand_metrics:
            reasons.append(f"candidate quality metric missing: {name}")
            continue
        b = float(base_metrics[name])
        c = float(cand_metrics[name])
        direction = spec.get("direction", "higher")
        abs_reg, rel_reg = _regression(b, c, direction)
        if strict_zero_regression:
            max_abs = 0.0
            max_rel = 0.0
        else:
            max_abs = float(spec.get("max_abs_regression", 0.0))
            max_rel = float(spec.get("max_rel_regression", 0.0))
        ok = abs_reg <= max_abs + 1e-15 and rel_reg <= max_rel + 1e-15
        details["metrics"][name] = {
            "baseline": b,
            "candidate": c,
            "direction": direction,
            "abs_regression": abs_reg,
            "rel_regression": rel_reg,
            "max_abs_regression": max_abs,
            "max_rel_regression": max_rel,
            "pass": ok,
        }
        if not ok:
            reasons.append(
                f"quality regression {name}: baseline={b:g}, candidate={c:g}, "
                f"abs_reg={abs_reg:g}, rel_reg={rel_reg:.6%}"
            )

    require_case_passes = bool(contract.get("require_case_passes", True))
    if require_case_passes:
        base_cases = baseline.get("cases", {}) or {}
        cand_cases = candidate.get("cases", {}) or {}
        for case_id, base_case in base_cases.items():
            base_pass = bool(base_case.get("passed", base_case) if isinstance(base_case, dict) else base_case)
            if not base_pass:
                continue
            if case_id not in cand_cases:
                reasons.append(f"candidate missing required quality case: {case_id}")
                details["cases"][case_id] = {"pass": False, "reason": "missing"}
                continue
            cand_case = cand_cases[case_id]
            cand_pass = bool(cand_case.get("passed", cand_case) if isinstance(cand_case, dict) else cand_case)
            details["cases"][case_id] = {"baseline_pass": True, "candidate_pass": cand_pass, "pass": cand_pass}
            if not cand_pass:
                reasons.append(f"new critical case failure: {case_id}")

    return GateDecision(not reasons, reasons, details)


def evaluate_performance(
    best_perf: dict[str, Any],
    candidate_perf: dict[str, Any],
    objective: dict[str, Any],
    guardrails: dict[str, Any],
) -> GateDecision:
    reasons: list[str] = []
    details: dict[str, Any] = {"objective": {}, "guardrails": {}}
    metric = str(objective["metric"])
    direction = str(objective["direction"])
    min_frac = float(objective.get("min_improvement_fraction", 0.0))
    min_abs = float(objective.get("min_improvement_abs", 0.0))

    try:
        b = get_metric(best_perf, metric)
        c = get_metric(candidate_perf, metric)
    except (KeyError, TypeError) as exc:
        return GateDecision(False, [f"primary performance metric invalid or missing: {exc}"], details)

    if direction == "lower":
        abs_imp = b - c
    else:
        abs_imp = c - b
    rel_imp = abs_imp / max(abs(b), 1e-12)
    obj_ok = abs_imp > 0 and abs_imp + 1e-15 >= min_abs and rel_imp + 1e-15 >= min_frac
    details["objective"] = {
        "metric": metric,
        "direction": direction,
        "best": b,
        "candidate": c,
        "abs_improvement": abs_imp,
        "rel_improvement": rel_imp,
        "min_improvement_abs": min_abs,
        "min_improvement_fraction": min_frac,
        "pass": obj_ok,
    }
    if not obj_ok:
        reasons.append(
            f"primary objective did not improve enough: {metric} best={b:g}, candidate={c:g}, "
            f"improvement={rel_imp:.4%}"
        )

    for path, spec in guardrails.items():
        if not isinstance(spec, dict):
            spec = {}
        try:
            gb = get_metric(best_perf, path)
            gc = get_metric(candidate_perf, path)
        except (KeyError, TypeError) as exc:
            if bool(spec.get("required", True)):
                reasons.append(f"guardrail metric invalid or missing {path}: {exc}")
            continue
        gdir = spec.get("direction", "lower")
        abs_reg, rel_reg = _regression(gb, gc, gdir)
        has_abs = "max_abs_regression" in spec
        has_rel = "max_rel_regression" in spec
        if has_abs or has_rel:
            max_abs = float(spec["max_abs_regression"]) if has_abs else float("inf")
            max_rel = float(spec["max_rel_regression"]) if has_rel else float("inf")
        else:
            max_abs = 0.0
            max_rel = 0.0
        ok = abs_reg <= max_abs + 1e-15 and rel_reg <= max_rel + 1e-15
        details["guardrails"][path] = {
            "best": gb,
            "candidate": gc,
            "direction": gdir,
            "abs_regression": abs_reg,
            "rel_regression": rel_reg,
            "max_abs_regression": max_abs,
            "max_rel_regression": max_rel,
            "pass": ok,
        }
        if not ok:
            reasons.append(
                f"performance guardrail failed {path}: best={gb:g}, candidate={gc:g}, rel_reg={rel_reg:.4%}"
            )

    return GateDecision(not reasons, reasons, details)
