import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist

from al_reweighting import (
    _VoronoiL2ConvergenceTracker,
    _VoronoiL2TraceRecorder,
    _relative_primal_dual_gap,
    _voronoi_l2_primal_dual_metrics,
    _voronoi_l2_weights_numpy,
    _voronoi_l2_weights_torch,
)


def tracker_metrics(objective, relative_gap=1.0, grad_inf=1.0, valid=True):
    return {
        "dual_objective": float(objective),
        "relative_primal_dual_gap": float(relative_gap),
        "grad_inf": float(grad_inf),
        "gap_certificate_valid": bool(valid),
    }


class VoronoiL2ConvergenceTrackerTests(unittest.TestCase):
    def make_tracker(self, **overrides):
        values = {
            "relative_gap_tol": 1e-2,
            "gradient_tol": 1e-4,
            "stability_window": 1,
            "dual_relative_tol": 1e-4,
            "weight_l1_tol": 5e-3,
            "stability_patience": 2,
        }
        values.update(overrides)
        return _VoronoiL2ConvergenceTracker(**values)

    def test_relative_gap_has_priority_over_gradient(self):
        tracker = self.make_tracker()
        result = tracker.observe(
            tracker_metrics(1.0, relative_gap=5e-3, grad_inf=5e-5),
            np.array([0.5, 0.5]),
        )
        self.assertEqual(result["stop_reason"], "relative_gap_tolerance")

    def test_invalid_gap_certificate_cannot_stop(self):
        tracker = self.make_tracker(gradient_tol=0.0)
        result = tracker.observe(
            tracker_metrics(1.0, relative_gap=0.0, grad_inf=1.0, valid=False),
            np.array([0.5, 0.5]),
        )
        self.assertIsNone(result["stop_reason"])

    def test_gradient_tolerance_is_an_independent_stop(self):
        tracker = self.make_tracker(relative_gap_tol=0.0)
        result = tracker.observe(
            tracker_metrics(1.0, relative_gap=1.0, grad_inf=9e-5),
            np.array([0.5, 0.5]),
        )
        self.assertEqual(result["stop_reason"], "gradient_tolerance")

    def test_stability_requires_dual_and_weight_conditions_with_patience(self):
        tracker = self.make_tracker()
        weights = [
            np.array([0.5000, 0.5000]),
            np.array([0.5010, 0.4990]),
            np.array([0.5020, 0.4980]),
        ]
        results = [
            tracker.observe(tracker_metrics(obj), weight)
            for obj, weight in zip([1.0, 0.99995, 0.99990], weights)
        ]
        self.assertEqual([r["stability_streak"] for r in results], [0, 1, 2])
        self.assertIsNone(results[1]["stop_reason"])
        self.assertEqual(results[2]["stop_reason"], "stable_not_certified")

    def test_objective_worsening_and_weight_change_reset_stability(self):
        tracker = self.make_tracker(stability_patience=3)
        base = np.array([0.5, 0.5])
        tracker.observe(tracker_metrics(1.0), base)
        self.assertEqual(
            tracker.observe(tracker_metrics(0.99995), base)["stability_streak"], 1
        )
        worsened = tracker.observe(tracker_metrics(1.0), base)
        self.assertLess(worsened["dual_relative_improvement_window"], 0.0)
        self.assertEqual(worsened["stability_streak"], 0)
        tracker.observe(tracker_metrics(0.99995), base)
        changed = tracker.observe(
            tracker_metrics(0.99990), np.array([0.6, 0.4])
        )
        self.assertGreater(changed["normalized_weight_l1_change_window"], 5e-3)
        self.assertEqual(changed["stability_streak"], 0)

    def test_current_worsening_resets_even_when_window_net_improves(self):
        tracker = self.make_tracker(
            stability_window=2,
            dual_relative_tol=2e-4,
            stability_patience=3,
        )
        weights = np.array([0.5, 0.5])
        tracker.observe(tracker_metrics(1.0), weights)
        tracker.observe(tracker_metrics(0.9998), weights)
        result = tracker.observe(tracker_metrics(0.9999), weights)

        self.assertGreater(result["dual_relative_improvement_window"], 0.0)
        self.assertLess(result["objective_improvement"], 0.0)
        self.assertEqual(result["stability_streak"], 0)


class VoronoiL2CertificateTests(unittest.TestCase):
    def test_relative_gap_scaling_and_negative_roundoff(self):
        rel, valid = _relative_primal_dual_gap(2.0, 1.98)
        self.assertTrue(valid)
        self.assertAlmostEqual(rel, 0.01)

        rel, valid = _relative_primal_dual_gap(1.0, 1.0 + 5e-8)
        self.assertTrue(valid)
        self.assertEqual(rel, 0.0)

        _, valid = _relative_primal_dual_gap(1.0, 1.0 + 5e-6)
        self.assertFalse(valid)

    def test_primal_upper_bound_dominates_dual_lower_bound(self):
        X_pool = np.array([[0.0], [0.1], [0.8], [1.0]], dtype=np.float64)
        X_labeled = np.array([[0.0], [1.0]], dtype=np.float64)
        z = np.array([0.3, 0.1], dtype=np.float64)
        distances_plus_z = cdist(X_pool, X_labeled) + z[None, :]
        assignments = np.argmin(distances_plus_z, axis=1)
        counts = np.bincount(assignments, minlength=len(X_labeled))
        total_min_sum = np.min(distances_plus_z, axis=1).sum()

        metrics = _voronoi_l2_primal_dual_metrics(
            z, counts, total_min_sum, len(X_pool), reweight_lambda=0.5
        )

        self.assertGreaterEqual(metrics["primal_dual_gap"], -1e-12)
        self.assertTrue(metrics["gap_certificate_valid"])
        self.assertLessEqual(
            metrics["dual_lower_bound"], metrics["primal_upper_bound"] + 1e-12
        )
        expected_grad = np.maximum(z, 0.0) - counts / len(X_pool)
        self.assertAlmostEqual(metrics["grad_inf"], np.max(np.abs(expected_grad)))
        self.assertAlmostEqual(metrics["normalized_weights"].sum(), 1.0)

    def test_stability_stop_has_explicit_noncertified_class(self):
        recorder = _VoronoiL2TraceRecorder(
            backend="test",
            max_iter=512,
            relative_gap_tol=1e-2,
            gradient_tol=1e-4,
            stability_window=1,
            dual_relative_tol=1e-4,
            weight_l1_tol=5e-3,
            stability_patience=2,
            emit=lambda _line: None,
        )
        for update, objective in enumerate([1.0, 0.99995, 0.99990]):
            metrics = {
                "dual_objective": objective,
                "dual_lower_bound": -objective,
                "primal_upper_bound": 1.0,
                "primal_dual_gap": 2.0,
                "relative_primal_dual_gap": 2.0,
                "gap_certificate_valid": True,
                "grad_inf": 1.0,
                "raw_weight_sum": 1.0,
                "normalized_weights": np.array([0.5, 0.5]),
            }
            reason = recorder.observe(metrics, update + 1, update)
        self.assertEqual(reason, "stable_not_certified")
        trace = recorder.finish(reason, function_evaluations=3)
        self.assertTrue(trace["converged"])
        self.assertFalse(trace["certified"])
        self.assertTrue(trace["stable_not_certified"])
        self.assertEqual(trace["termination_class"], "stable_not_certified")


class VoronoiL2NumpyBackendTests(unittest.TestCase):
    def setUp(self):
        self.X_pool = np.array(
            [[0.0], [0.05], [0.10], [0.15], [1.0]], dtype=np.float64
        )
        self.X_labeled = np.array([[0.0], [1.0]], dtype=np.float64)

    def solver_kwargs(self, **overrides):
        values = {
            "relative_gap_tol": 1e-2,
            "gradient_tol": 1e-4,
            "stability_window": 10,
            "dual_relative_tol": 1e-4,
            "weight_l1_tol": 5e-3,
            "stability_patience": 2,
        }
        values.update(overrides)
        return values

    def test_trace_contains_certificates_and_termination_class(self):
        state = {}
        lines = []
        weights = _voronoi_l2_weights_numpy(
            self.X_pool,
            self.X_labeled,
            reweight_lambda=0.2,
            state=state,
            max_iter=512,
            trace_context={"trial": 1, "seed": 42, "n_queries": 0, "solve": 1},
            trace_logger=lines.append,
            **self.solver_kwargs(),
        )
        trace = state["last_optimizer_trace"]

        self.assertIn(
            trace["stop_reason"], {"relative_gap_tolerance", "gradient_tolerance"}
        )
        self.assertTrue(trace["converged"])
        self.assertTrue(trace["certified"])
        self.assertFalse(trace["stable_not_certified"])
        self.assertEqual(trace["termination_class"], "certified")
        self.assertAlmostEqual(weights.sum(), len(self.X_labeled))
        self.assertGreaterEqual(trace["final_primal_dual_gap"], -1e-10)
        self.assertEqual(
            len(trace["records"]), trace["accepted_updates_completed"] + 1
        )
        self.assertIn("relative_primal_dual_gap", trace["records"][-1])
        self.assertIn("accepted_update", trace["records"][-1])
        self.assertTrue(any("[Voronoi-L2][summary]" in line for line in lines))

    def test_one_update_budget_reports_max_iter_not_converged(self):
        state = {}
        _voronoi_l2_weights_numpy(
            self.X_pool,
            self.X_labeled,
            reweight_lambda=0.2,
            state=state,
            max_iter=1,
            trace_logger=lambda _line: None,
            **self.solver_kwargs(
                relative_gap_tol=0.0,
                gradient_tol=0.0,
                stability_window=10,
            ),
        )
        trace = state["last_optimizer_trace"]
        self.assertEqual(trace["stop_reason"], "max_iter")
        self.assertEqual(trace["termination_class"], "max_iter_not_converged")
        self.assertFalse(trace["converged"])
        self.assertEqual(trace["accepted_updates_completed"], 1)

    def test_trace_metrics_do_not_add_distance_evaluations(self):
        state = {}
        with patch("al_reweighting.cdist", wraps=cdist) as cdist_mock:
            _voronoi_l2_weights_numpy(
                self.X_pool,
                self.X_labeled,
                reweight_lambda=0.2,
                state=state,
                max_iter=1,
                trace_logger=lambda _line: None,
                **self.solver_kwargs(
                    relative_gap_tol=0.0,
                    gradient_tol=0.0,
                    stability_window=10,
                ),
            )

        trace = state["last_optimizer_trace"]
        self.assertEqual(cdist_mock.call_count, trace["function_evaluations"])


class VoronoiL2TorchBackendTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("torch").cuda.is_available(), "CUDA is not available"
    )
    def test_repeated_single_updates_preserve_history_and_update_semantics(self):
        rng = np.random.RandomState(7)
        X_pool = rng.normal(size=(100, 3)).astype(np.float32)
        X_labeled = rng.normal(size=(12, 3)).astype(np.float32)
        state = {}
        _voronoi_l2_weights_torch(
            X_pool,
            X_labeled,
            reweight_lambda=0.2,
            state=state,
            max_iter=5,
            relative_gap_tol=0.0,
            gradient_tol=0.0,
            stability_window=10,
            dual_relative_tol=0.0,
            weight_l1_tol=0.0,
            stability_patience=10,
            trace_logger=lambda _line: None,
        )
        trace = state["last_optimizer_trace"]
        self.assertIn(trace["stop_reason"], {"max_iter", "relative_gap_tolerance"})
        self.assertEqual(trace["accepted_updates_completed"], 5)
        self.assertGreater(trace["function_evaluations"], 6)
        self.assertEqual(
            [record["accepted_update"] for record in trace["records"]],
            list(range(6)),
        )
        objectives = [record["dual_objective"] for record in trace["records"]]
        self.assertTrue(
            all(next_value <= value for value, next_value in zip(
                objectives, objectives[1:]
            ))
        )


if __name__ == "__main__":
    unittest.main()
