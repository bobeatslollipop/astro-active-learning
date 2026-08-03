import unittest
from unittest import mock

import numpy as np
from scipy.spatial.distance import cdist

from al_queries import (
    _wasserstein_initial_wwds_numpy,
    _wasserstein_l2_base_cells_numpy,
    _wasserstein_l2_coupling_numpy,
    _wasserstein_l2_coupling_torch,
    _wasserstein_l2_full_penalties_numpy,
    _wasserstein_l2_full_penalties_torch,
    _wasserstein_l2_initial_capture_counts_numpy,
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
        actual = _wasserstein_l2_coupling_numpy(target, support, 5, 3.0)
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
            _wasserstein_l2_coupling_torch(target, support, 4, 2.0),
            _wasserstein_l2_coupling_numpy(target, support, 4, 2.0),
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

    def test_three_way_comparison_records_overlap_and_regret(self):
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


if __name__ == "__main__":
    unittest.main()
