import unittest

from dit_autoresearch.gates import evaluate_performance, evaluate_quality


class GateTests(unittest.TestCase):
    def test_quality_strict_rejects_any_drop(self):
        baseline = {"metrics": {"acc": 0.9}, "cases": {"hard": {"passed": True}}}
        candidate = {"metrics": {"acc": 0.8999}, "cases": {"hard": {"passed": True}}}
        contract = {"metrics": {"acc": {"direction": "higher"}}, "require_case_passes": True}
        d = evaluate_quality(baseline, candidate, contract, strict_zero_regression=True)
        self.assertFalse(d.passed)

    def test_quality_case_regression_rejected(self):
        baseline = {"metrics": {"acc": 0.9}, "cases": {"hard": {"passed": True}}}
        candidate = {"metrics": {"acc": 0.91}, "cases": {"hard": {"passed": False}}}
        contract = {"metrics": {"acc": {"direction": "higher"}}, "require_case_passes": True}
        d = evaluate_quality(baseline, candidate, contract, strict_zero_regression=True)
        self.assertFalse(d.passed)

    def test_performance_win_with_relative_guardrail(self):
        best = {"latency": {"median_s": 10.0, "p95_s": 11.0}, "peak_vram_mb": 20000}
        cand = {"latency": {"median_s": 9.0, "p95_s": 11.05}, "peak_vram_mb": 20400}
        objective = {"metric": "latency.median_s", "direction": "lower", "min_improvement_fraction": 0.01}
        guards = {
            "latency.p95_s": {"direction": "lower", "max_rel_regression": 0.01},
            "peak_vram_mb": {"direction": "lower", "max_rel_regression": 0.05},
        }
        d = evaluate_performance(best, cand, objective, guards)
        self.assertTrue(d.passed, d.reasons)


if __name__ == "__main__":
    unittest.main()
