import unittest
from unittest import mock

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from al_queries import (
    _correct_power_block_numpy,
    _correct_power_block_torch,
    _restricted_coordinate_numpy,
    _restricted_coordinate_torch,
    _wasserstein_initial_wwds_numpy,
    _wasserstein_l2_base_cells_numpy,
    _wasserstein_l2_coupling_numpy,
    _wasserstein_l2_coupling_torch,
    _wasserstein_l2_full_penalties_numpy,
    _wasserstein_l2_full_penalties_torch,
    _wasserstein_l2_initial_capture_counts_numpy,
    _wasserstein_l2_v2_coupling_numpy,
    _wasserstein_l2_v2_coupling_torch,
    query_kmedianpp,
    query_wasserstein_l2,
)
from diagnose_wasserstein_l2_objectives import (
    certify_exact_winner,
    compare_objectives,
    solve_regularized_ot_objective,
)


def brute_candidate_scores(target, support, reweight_lambda):
    scores = []
    for candidate in target:
        augmented = np.vstack([support, candidate])
        distances = cdist(target, augmented)
        assignments = distances.argmin(axis=1)
        transport = distances[np.arange(len(target)), assignments].mean()
        masses = np.bincount(assignments, minlength=len(augmented)) / len(target)
        scores.append(transport + reweight_lambda * np.dot(masses, masses))
    return np.asarray(scores)


def brute_greedy_plan(target, support, n_pick, reweight_lambda):
    support = np.asarray(support).copy()
    available = np.ones(len(target), dtype=bool)
    selected = []
    for _ in range(n_pick):
        scores = brute_candidate_scores(target, support, reweight_lambda)
        scores[~available] = np.inf
        best = int(np.argmin(scores))
        selected.append(best)
        available[best] = False
        support = np.vstack([support, target[best]])
    return selected


class KMedianCandidateSubpoolTests(unittest.TestCase):
    def test_candidate_pool_size_is_honored_and_indices_are_mapped(self):
        X_pool = np.arange(30, dtype=np.float32).reshape(10, 3)
        expected_rng = np.random.RandomState(7)
        expected_subpool = expected_rng.choice(10, 4, replace=False)
        rng = np.random.RandomState(7)
        state = {"min_dists": np.ones(10, dtype=np.float32)}

        with mock.patch(
            "al_queries._query_kmedianpp_numpy",
            return_value=np.array([0, 3], dtype=np.intp),
        ) as mocked:
            selected = query_kmedianpp(
                X_pool,
                None,
                2,
                rng,
                X_labeled=X_pool[:1],
                state=state,
                pool_size=4,
            )

        np.testing.assert_array_equal(
            selected, expected_subpool[np.array([0, 3], dtype=np.intp)]
        )
        self.assertEqual(len(mocked.call_args.args[0]), 4)
        self.assertEqual(state, {})


class FullVoronoiL2ScoreTests(unittest.TestCase):
    def test_penalties_match_brute_force_and_chunk_size(self):
        rng = np.random.RandomState(7)
        target = rng.normal(size=(13, 4)).astype(np.float32)
        support = rng.normal(size=(5, 4)).astype(np.float32)
        intra = cdist(target, target).astype(np.float32)
        base, cell_ids, cell_counts = _wasserstein_l2_base_cells_numpy(
            target, support
        )

        expected = []
        for candidate in target:
            augmented = np.vstack([support, candidate])
            assignments = cdist(target, augmented).argmin(axis=1)
            masses = (
                np.bincount(assignments, minlength=len(augmented)) / len(target)
            )
            expected.append(np.dot(masses, masses))

        for row_chunk in (1, 3, len(target)):
            actual = _wasserstein_l2_full_penalties_numpy(
                base, intra, cell_ids, cell_counts, row_chunk=row_chunk
            )
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_ranking_reversal_from_legacy_captured_mass(self):
        target = np.arange(5, dtype=np.float32)[:, None]
        support = np.array([[-0.5], [0.5]], dtype=np.float32)
        lam = 0.5
        intra = cdist(target, target).astype(np.float32)
        base, cell_ids, cell_counts = _wasserstein_l2_base_cells_numpy(
            target, support
        )
        transport = _wasserstein_initial_wwds_numpy(base, intra).astype(float)
        captured = _wasserstein_l2_initial_capture_counts_numpy(base, intra)
        legacy = transport + lam * np.square(captured / len(target))
        full = transport + lam * _wasserstein_l2_full_penalties_numpy(
            base, intra, cell_ids, cell_counts, row_chunk=2
        )

        self.assertEqual(int(np.argmin(legacy)), 4)
        self.assertEqual(int(np.argmin(full)), 3)
        self.assertLess(full[3], full[4])

    def test_multistep_incremental_plan_matches_full_recomputation(self):
        rng = np.random.RandomState(11)
        target = rng.normal(size=(16, 3)).astype(np.float32)
        support = rng.normal(size=(6, 3)).astype(np.float32)
        expected = brute_greedy_plan(target, support, 5, 3.0)
        actual = _wasserstein_l2_v2_coupling_numpy(target, support, 5, 3.0)
        self.assertEqual(actual, expected)

    def test_empty_support_ties_and_zero_mass_cells(self):
        target = np.array([[0.0], [0.0], [1.0], [2.0]], dtype=np.float32)
        intra = cdist(target, target).astype(np.float32)
        base, cell_ids, cell_counts = _wasserstein_l2_base_cells_numpy(target, None)
        penalties = _wasserstein_l2_full_penalties_numpy(
            base, intra, cell_ids, cell_counts, row_chunk=1
        )
        np.testing.assert_allclose(penalties, np.ones(len(target)))

        support = np.array([[0.0], [10.0]], dtype=np.float32)
        base, cell_ids, cell_counts = _wasserstein_l2_base_cells_numpy(
            target, support
        )
        # The far support atom is a zero-mass cell and does not appear in the
        # compact state; strict ties keep duplicate targets in the old cell.
        self.assertEqual(len(cell_counts), 1)
        penalties = _wasserstein_l2_full_penalties_numpy(
            base, intra, cell_ids, cell_counts, row_chunk=2
        )
        expected = []
        for candidate in target:
            assignments = cdist(target, np.vstack([support, candidate])).argmin(axis=1)
            masses = np.bincount(assignments, minlength=3) / len(target)
            expected.append(np.dot(masses, masses))
        np.testing.assert_allclose(penalties, expected)

    def test_penalty_reduction_does_not_compute_distances(self):
        target = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        support = np.array([[0.5]], dtype=np.float32)
        intra = cdist(target, target).astype(np.float32)
        base, cell_ids, cell_counts = _wasserstein_l2_base_cells_numpy(
            target, support
        )
        with mock.patch("al_queries.cdist", side_effect=AssertionError("extra cdist")):
            _wasserstein_l2_full_penalties_numpy(
                base, intra, cell_ids, cell_counts, row_chunk=1
            )


class FullVoronoiL2CudaTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("torch").cuda.is_available(), "CUDA is unavailable"
    )
    def test_cuda_penalties_and_plan_match_numpy(self):
        import torch

        rng = np.random.RandomState(19)
        target = rng.normal(size=(12, 3)).astype(np.float32)
        support = rng.normal(size=(4, 3)).astype(np.float32)
        intra = cdist(target, target).astype(np.float32)
        base, cell_ids, cell_counts = _wasserstein_l2_base_cells_numpy(
            target, support
        )
        expected = _wasserstein_l2_full_penalties_numpy(
            base, intra, cell_ids, cell_counts, row_chunk=2
        )

        actual = _wasserstein_l2_full_penalties_torch(
            torch.as_tensor(base, device="cuda"),
            torch.as_tensor(intra, device="cuda"),
            torch.as_tensor(cell_ids, dtype=torch.long, device="cuda"),
            torch.as_tensor(cell_counts, dtype=torch.float32, device="cuda"),
            row_chunk=3,
        ).cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        chunk_one = _wasserstein_l2_full_penalties_torch(
            torch.as_tensor(base, device="cuda"),
            torch.as_tensor(intra, device="cuda"),
            torch.as_tensor(cell_ids, dtype=torch.long, device="cuda"),
            torch.as_tensor(cell_counts, dtype=torch.float32, device="cuda"),
            row_chunk=1,
        ).cpu().numpy()
        np.testing.assert_allclose(chunk_one, actual, rtol=0, atol=0)
        self.assertEqual(
            _wasserstein_l2_v2_coupling_torch(target, support, 4, 2.0),
            _wasserstein_l2_v2_coupling_numpy(target, support, 4, 2.0),
        )


class PowerCellV3Tests(unittest.TestCase):
    def test_scalar_coordinate_matches_independent_convex_minimization(self):
        rng = np.random.RandomState(23)
        residuals = np.vstack([
            rng.normal(size=31),
            np.zeros(31),
            np.array([1.0] * 10 + [0.0] * 21),
            rng.normal(loc=0.2, scale=2.0, size=31),
        ]).astype(np.float32)
        for lam in (0.25, 3.0, 10000.0):
            z, values = _restricted_coordinate_numpy(
                residuals, lam, steps=48
            )
            for row, (actual_z, actual_value) in enumerate(zip(z, values)):
                upper = min(2.0 * lam, max(float(residuals[row].max()), 0.0))

                def objective(value):
                    return (
                        value * value / (4.0 * lam)
                        + np.maximum(residuals[row] - value, 0.0).mean()
                    )

                breakpoints = np.clip(residuals[row], 0.0, upper)
                stationary = 2.0 * lam * np.arange(
                    len(residuals[row]) + 1, dtype=np.float64
                ) / len(residuals[row])
                exact_candidates = np.unique(np.concatenate([
                    np.array([0.0, upper]),
                    breakpoints,
                    stationary[(stationary >= 0.0) & (stationary <= upper)],
                ]))
                expected_value = min(objective(value) for value in exact_candidates)
                self.assertGreaterEqual(actual_z, 0.0)
                self.assertLessEqual(actual_z, 2.0 * lam)
                self.assertAlmostEqual(actual_value, expected_value, places=7)

    def test_block_correction_matches_restricted_convex_solve(self):
        rng = np.random.RandomState(29)
        base = rng.uniform(0.7, 2.0, size=17)
        selected_dists = rng.uniform(0.0, 2.5, size=(3, 17))
        lam = 2.5
        z, current_base, trace = _correct_power_block_numpy(
            base,
            selected_dists,
            np.array([0.4, 0.8, 0.2]),
            lam,
            coordinate_steps=48,
            max_sweeps=256,
            dual_relative_tol=1e-11,
            z_relative_tol=1e-10,
            patience=2,
        )

        def objective(values):
            adjusted = selected_dists + values[:, None]
            return (
                np.dot(values, values) / (4.0 * lam)
                + np.maximum(base - adjusted.min(axis=0), 0.0).mean()
            )

        reference = minimize(
            objective,
            np.full(3, 0.5),
            method="Powell",
            bounds=[(0.0, 2.0 * lam)] * 3,
            options={"xtol": 1e-12, "ftol": 1e-12, "maxiter": 10000},
        )
        self.assertTrue(trace["converged"])
        self.assertAlmostEqual(objective(z), reference.fun, places=7)
        self.assertTrue(np.all(np.diff(trace["objective_history"]) <= 1e-9))
        np.testing.assert_allclose(
            current_base,
            np.minimum(base, (selected_dists + z[:, None]).min(axis=0)),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_block_correction_reports_cap_and_rejects_nonfinite(self):
        base = np.array([1.0, 1.2, 0.8])
        dists = np.array([[0.2, 0.9, 0.4], [0.8, 0.3, 0.7]])
        _, _, capped = _correct_power_block_numpy(
            base,
            dists,
            np.array([0.1, 0.1]),
            1.0,
            max_sweeps=1,
            patience=2,
        )
        self.assertEqual(capped["stop_reason"], "max_sweeps_not_converged")
        self.assertFalse(capped["converged"])
        with self.assertRaises(FloatingPointError):
            _correct_power_block_numpy(
                np.array([1.0, np.nan]),
                np.array([[0.2, 0.3]]),
                np.array([0.1]),
                1.0,
            )

    def test_multistep_plan_is_chunk_invariant_with_separate_target(self):
        rng = np.random.RandomState(31)
        candidates = rng.normal(size=(8, 3)).astype(np.float32)
        target = rng.normal(size=(13, 3)).astype(np.float32)
        base = rng.uniform(0.8, 2.0, size=len(target)).astype(np.float32)
        kwargs = dict(
            coordinate_steps=48,
            corrective_max_sweeps=128,
            corrective_dual_relative_tol=1e-10,
            corrective_z_relative_tol=1e-9,
            corrective_patience=2,
        )
        plan_one, trace_one = _wasserstein_l2_coupling_numpy(
            candidates, target, base, 4, 3.0,
            candidate_chunk_size=1, **kwargs,
        )
        plan_many, trace_many = _wasserstein_l2_coupling_numpy(
            candidates, target, base, 4, 3.0,
            candidate_chunk_size=5, **kwargs,
        )
        self.assertEqual(plan_one, plan_many)
        self.assertEqual(len(set(plan_one)), len(plan_one))
        np.testing.assert_allclose(
            [step["restricted_dual_drop"] for step in trace_one["steps"]],
            [step["restricted_dual_drop"] for step in trace_many["steps"]],
            rtol=0,
            atol=1e-12,
        )
        self.assertTrue(all(
            "corrective_grad_inf" in step for step in trace_one["steps"]
        ))

    def test_query_requires_reweight_state_and_records_v3_trace(self):
        pool = np.arange(8, dtype=np.float32)[:, None]
        target = np.linspace(0.0, 7.0, 11, dtype=np.float32)[:, None]
        support = np.array([[0.0], [7.0]], dtype=np.float32)
        optimizer_state = {
            "z": np.array([0.5, 0.5], dtype=np.float64),
            "last_optimizer_trace": {
                "trial": 1,
                "seed": 42,
                "n_queries": 0,
                "solve": 1,
                "termination_class": "stable_not_certified",
                "stop_reason": "stable_not_certified",
            },
        }
        strategy_state = {}
        selected = query_wasserstein_l2(
            pool,
            None,
            3,
            np.random.RandomState(5),
            X_labeled=support,
            state=strategy_state,
            pool_size=6,
            plan_size=6,
            available_mask=np.ones(len(pool), dtype=bool),
            reweight_lambda=2.0,
            reweight_target=target,
            voronoi_l2_state=optimizer_state,
            coordinate_steps=40,
            corrective_max_sweeps=64,
            candidate_chunk_size=2,
        )
        self.assertEqual(len(selected), 3)
        trace = strategy_state["last_plan_trace"]
        self.assertEqual(trace["query_implementation_version"], 3)
        self.assertEqual(trace["target_rows"], len(target))
        self.assertEqual(trace["candidate_rows"], 6)
        self.assertEqual(len(trace["candidate_subpool_sha256"]), 64)
        self.assertEqual(
            trace["source_reweight_solve"]["termination_class"],
            "stable_not_certified",
        )
        self.assertTrue(all("pool_index" in step for step in trace["steps"]))

        with self.assertRaisesRegex(ValueError, "requires the actual"):
            query_wasserstein_l2(
                pool,
                None,
                1,
                np.random.RandomState(1),
                X_labeled=support,
                reweight_lambda=2.0,
                voronoi_l2_state=optimizer_state,
            )


class PowerCellV3CudaTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("torch").cuda.is_available(), "CUDA is unavailable"
    )
    def test_cuda_scalar_block_and_plan_match_numpy(self):
        import torch

        rng = np.random.RandomState(37)
        residuals = rng.normal(size=(5, 19)).astype(np.float32)
        z_np, value_np = _restricted_coordinate_numpy(
            residuals, 4.0, steps=48
        )
        z_t, value_t = _restricted_coordinate_torch(
            torch.as_tensor(residuals, device="cuda"), 4.0, steps=48
        )
        np.testing.assert_allclose(z_t.cpu(), z_np, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(value_t.cpu(), value_np, rtol=1e-6, atol=1e-7)

        candidates = rng.normal(size=(7, 2)).astype(np.float32)
        target = rng.normal(size=(11, 2)).astype(np.float32)
        base = rng.uniform(0.7, 1.8, size=len(target)).astype(np.float32)
        kwargs = dict(
            coordinate_steps=40,
            corrective_max_sweeps=128,
            corrective_dual_relative_tol=1e-9,
            corrective_z_relative_tol=1e-8,
            corrective_patience=2,
            candidate_chunk_size=3,
        )
        numpy_plan, numpy_trace = _wasserstein_l2_coupling_numpy(
            candidates, target, base, 3, 2.0, **kwargs
        )
        torch_plan, torch_trace = _wasserstein_l2_coupling_torch(
            candidates, target, base, 3, 2.0, **kwargs
        )
        self.assertEqual(torch_plan, numpy_plan)
        np.testing.assert_allclose(
            [step["restricted_dual_drop"] for step in torch_trace["steps"]],
            [step["restricted_dual_drop"] for step in numpy_trace["steps"]],
            rtol=1e-5,
            atol=1e-7,
        )


class ExactOracleTests(unittest.TestCase):
    def test_objective_bounds_are_valid_and_tight_on_small_problem(self):
        target = np.array([[0.0], [1.0], [3.0], [5.0]])
        support = np.array([[0.0], [5.0]])
        result = solve_regularized_ot_objective(
            target, support, 1.0, max_iter=1000, tolerance=1e-11
        )
        self.assertTrue(result["certificate_valid"])
        self.assertLessEqual(
            result["dual_lower_bound"], result["primal_upper_bound"] + 1e-10
        )
        self.assertGreaterEqual(result["primal_dual_gap"], -1e-10)
        self.assertLess(result["primal_dual_gap"], 1e-7)

    def test_winner_requires_separated_intervals(self):
        certified = certify_exact_winner([
            {
                "candidate_index": 0,
                "primal_upper_bound": 1.0,
                "dual_lower_bound": 0.9,
                "certificate_valid": True,
            },
            {
                "candidate_index": 1,
                "primal_upper_bound": 2.0,
                "dual_lower_bound": 1.5,
                "certificate_valid": True,
            },
        ])
        self.assertEqual(certified["status"], "certified")
        self.assertEqual(certified["candidate_index"], 0)

        unresolved = certify_exact_winner([
            {
                "candidate_index": 0,
                "primal_upper_bound": 1.0,
                "dual_lower_bound": 0.8,
                "certificate_valid": True,
            },
            {
                "candidate_index": 1,
                "primal_upper_bound": 1.1,
                "dual_lower_bound": 0.95,
                "certificate_valid": True,
            },
        ])
        self.assertEqual(unresolved["status"], "unresolved")
        self.assertIsNone(unresolved["candidate_index"])

        single = certify_exact_winner([
            {
                "candidate_index": 7,
                "primal_upper_bound": 1.0,
                "dual_lower_bound": 0.9,
                "certificate_valid": True,
            }
        ])
        self.assertEqual(single["status"], "certified")
        self.assertIsNone(single["best_competitor_lower_bound"])

    def test_four_way_comparison_records_overlap_and_regret(self):
        target = np.array([[0.0], [1.0], [3.0], [5.0]])
        support = np.array([[0.0], [5.0]])
        result = compare_objectives(
            target, support, 1.0, n_pick=1, max_iter=1000, tolerance=1e-10
        )
        self.assertEqual(len(result["candidate_records"]), len(target))
        step = result["steps"][0]
        self.assertEqual(step["exact_decision"]["status"], "certified")
        self.assertIn("legacy_regret_interval", step)
        self.assertIn("full_voronoi_regret_interval", step)
        self.assertIn("power_v3_regret_interval", step)
        self.assertIn("power_v3_choice", step)
        self.assertEqual(len(result["production_v3_batch_plan"]), 1)


if __name__ == "__main__":
    unittest.main()
