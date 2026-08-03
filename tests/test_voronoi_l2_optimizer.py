import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist

from al_reweighting import (
    _VoronoiL2ConvergenceTracker,
    _voronoi_l2_primal_dual_metrics,
    _voronoi_l2_weights_numpy,
)


class VoronoiL2ConvergenceTrackerTests(unittest.TestCase):
    def test_two_consecutive_small_improvements_are_required(self):
        tracker = _VoronoiL2ConvergenceTracker(
            objective_tol=1e-4, objective_patience=2, gradient_tol=1e-5
        )
        observations = [
            tracker.observe(1.0, 1.0),
            tracker.observe(0.99995, 1.0),
            tracker.observe(0.99970, 1.0),
            tracker.observe(0.99965, 1.0),
            tracker.observe(0.99960, 1.0),
        ]
        self.assertEqual([item[1] for item in observations], [0, 1, 0, 1, 2])
        self.assertIsNone(observations[-2][2])
        self.assertEqual(observations[-1][2], "objective_tolerance")

    def test_objective_worsening_resets_patience(self):
        tracker = _VoronoiL2ConvergenceTracker(
            objective_tol=1e-4, objective_patience=2, gradient_tol=1e-5
        )
        tracker.observe(1.0, 1.0)
        self.assertEqual(tracker.observe(0.99995, 1.0)[1], 1)
        improvement, streak, reason = tracker.observe(1.00000, 1.0)
        self.assertLess(improvement, 0.0)
        self.assertEqual(streak, 0)
        self.assertIsNone(reason)

    def test_gradient_tolerance_is_an_independent_stop(self):
        tracker = _VoronoiL2ConvergenceTracker(
            objective_tol=1e-4, objective_patience=2, gradient_tol=1e-5
        )
        _, _, reason = tracker.observe(1.0, 9e-6)
        self.assertEqual(reason, "gradient_tolerance")


class VoronoiL2CertificateTests(unittest.TestCase):
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
        self.assertLessEqual(
            metrics["dual_lower_bound"], metrics["primal_upper_bound"] + 1e-12
        )
        expected_grad = np.maximum(z, 0.0) / 1.0 - counts / len(X_pool)
        self.assertAlmostEqual(metrics["grad_inf"], np.max(np.abs(expected_grad)))


class VoronoiL2NumpyBackendTests(unittest.TestCase):
    def setUp(self):
        self.X_pool = np.array(
            [[0.0], [0.05], [0.10], [0.15], [1.0]], dtype=np.float64
        )
        self.X_labeled = np.array([[0.0], [1.0]], dtype=np.float64)

    def test_trace_contains_certificates_and_gradient_stop(self):
        state = {}
        lines = []
        weights = _voronoi_l2_weights_numpy(
            self.X_pool,
            self.X_labeled,
            reweight_lambda=0.2,
            state=state,
            max_iter=128,
            objective_tol=1e-4,
            objective_patience=2,
            gradient_tol=1e-5,
            trace_context={"trial": 1, "seed": 42, "n_queries": 0, "solve": 1},
            trace_logger=lines.append,
        )
        trace = state["last_optimizer_trace"]

        self.assertEqual(trace["stop_reason"], "gradient_tolerance")
        self.assertTrue(trace["converged"])
        self.assertAlmostEqual(weights.sum(), len(self.X_labeled))
        self.assertGreaterEqual(trace["final_primal_dual_gap"], -1e-10)
        self.assertLessEqual(trace["final_grad_inf"], 1e-5)
        self.assertEqual(len(trace["records"]), trace["iterations_completed"] + 1)
        self.assertIn("primal_upper_bound", trace["records"][-1])
        self.assertIn("dual_lower_bound", trace["records"][-1])
        self.assertTrue(any("[Voronoi-L2][summary]" in line for line in lines))

    def test_one_update_budget_reports_max_iter(self):
        state = {}
        _voronoi_l2_weights_numpy(
            self.X_pool,
            self.X_labeled,
            reweight_lambda=0.2,
            state=state,
            max_iter=1,
            objective_tol=1e-20,
            objective_patience=2,
            gradient_tol=0.0,
            trace_logger=lambda _line: None,
        )
        trace = state["last_optimizer_trace"]
        self.assertEqual(trace["stop_reason"], "max_iter")
        self.assertFalse(trace["converged"])
        self.assertEqual(trace["iterations_completed"], 1)

    def test_trace_metrics_do_not_add_distance_evaluations(self):
        state = {}
        with patch("al_reweighting.cdist", wraps=cdist) as cdist_mock:
            _voronoi_l2_weights_numpy(
                self.X_pool,
                self.X_labeled,
                reweight_lambda=0.2,
                state=state,
                max_iter=1,
                objective_tol=1e-20,
                objective_patience=2,
                gradient_tol=0.0,
                trace_logger=lambda _line: None,
            )

        trace = state["last_optimizer_trace"]
        self.assertEqual(cdist_mock.call_count, trace["function_evaluations"])


if __name__ == "__main__":
    unittest.main()
