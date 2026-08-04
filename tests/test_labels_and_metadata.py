import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from al_data import (
    MP_LABEL,
    MR_LABEL,
    labels_from_feh,
    mp_probability,
    mp_target,
)
from al_metadata import (
    ACTIVE_DERIVED_KEYS,
    ACTIVE_INPUT_SECTIONS,
    build_active_params,
    git_metadata,
    update_params_status,
    write_params,
)
from al_queries import (
    WASSERSTEIN_L2_IMPLEMENTATION_VERSION,
    WASSERSTEIN_L2_QUERY_OBJECTIVE,
)
from run_xgboost_full_eval import model_metrics


class DummyClassifier:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self._probabilities = np.asarray(probabilities, dtype=np.float64)

    def predict_proba(self, X):
        return self._probabilities[:len(X)]


def make_h5(path):
    with h5py.File(path, "w") as handle:
        handle["bp_1"] = np.arange(12, dtype=np.float32)
        handle["rp_1"] = np.arange(12, dtype=np.float32) + 1
        handle["ebv"] = np.linspace(0, 1, 12, dtype=np.float32)
        handle["feh"] = np.linspace(-3, 0, 12, dtype=np.float32)
        handle["source_id"] = np.arange(100, 112, dtype=np.int64)


def active_args(warm_path, full_path, out_dir):
    values = {
        "warm_start_file": str(warm_path),
        "full_data_file": str(full_path),
        "feh_threshold": -2.0,
        "warm_start_max": 8,
        "pool_max": 12,
        "eval_size": 4,
        "eval_source": "full_heldout",
        "strategy": "kmedianpp",
        "total_queries": 6,
        "eval_every": 2,
        "wass_pool_size": 10,
        "wass_plan_size": 2,
        "eot_temperature": 1.0,
        "moment_ridge": 1.0,
        "reweighting": "voronoi_l2",
        "reweight_lambda": 100.0,
        "voronoi_l2_max_iter": 512,
        "voronoi_l2_relative_gap_tol": 1e-2,
        "voronoi_l2_gradient_tol": 1e-4,
        "voronoi_l2_stability_window": 10,
        "voronoi_l2_dual_relative_tol": 1e-4,
        "voronoi_l2_weight_l1_tol": 5e-3,
        "voronoi_l2_stability_patience": 2,
        "temperature": 1.0,
        "soft_topk": 0,
        "reweight_pool_size": 10,
        "reweight_source": "full_non_eval",
        "moment_weight_iters": 200,
        "model": "xgboost",
        "lambda_MP": 1.0,
        "class_balance_mode": "none",
        "train_weight_sum_mode": "fixed",
        "train_weight_sum": 10000.0,
        "C": 1.0,
        "ridge_alpha": 1.0,
        "xgb_n_estimators": 10,
        "xgb_max_depth": 3,
        "xgb_learning_rate": 0.1,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.8,
        "xgb_min_child_weight": 1.0,
        "xgb_gamma": 0.0,
        "xgb_reg_lambda": 1.0,
        "xgb_tree_method": "hist",
        "xgb_device": "cpu",
        "xgb_n_jobs": 1,
        "n_trials": 2,
        "n_snapshots": 3,
        "seed": 42,
        "include_zero_snapshot": True,
        "out_dir": str(out_dir),
        "initial_labeled_count": 8,
        "original_warm_start_count": 8,
        "eval_actual_size": 4,
        "eval_original_warm_count": 1,
        "eval_original_warm_fraction": 0.25,
        "eval_final_warm_overlap": 0,
        "eval_query_pool_overlap": 0,
        "query_rng_mode": "dedicated_for_kmedianpp",
        "initial_train_weight_target_sum": 10000.0,
        "reweight_source_total_count": 16,
        "reweight_source_final_warm_count": 8,
        "reweight_source_final_warm_fraction": 0.5,
    }
    return SimpleNamespace(**values)


class LabelConventionTests(unittest.TestCase):
    def test_feh_labels_and_metric_target(self):
        labels = labels_from_feh(np.array([-3.0, -2.0, -1.0]), -2.0)
        np.testing.assert_array_equal(labels, [MP_LABEL, MR_LABEL, MR_LABEL])
        np.testing.assert_array_equal(mp_target(labels), [1, 0, 0])

    def test_mp_probability_uses_classes_not_column_number(self):
        X = np.zeros((2, 1))
        canonical = DummyClassifier([MP_LABEL, MR_LABEL], [[0.8, 0.2], [0.3, 0.7]])
        reversed_order = DummyClassifier([MR_LABEL, MP_LABEL], [[0.2, 0.8], [0.7, 0.3]])
        np.testing.assert_allclose(mp_probability(canonical, X), [0.8, 0.3])
        np.testing.assert_allclose(mp_probability(reversed_order, X), [0.8, 0.3])

    def test_full_benchmark_metrics_match_legacy_mp_positive_semantics(self):
        legacy_labels = np.array([1, 0, 1, 0, 0], dtype=np.int32)
        canonical_labels = np.where(legacy_labels == 1, MP_LABEL, MR_LABEL)
        p_mp = np.array([0.9, 0.2, 0.7, 0.1, 0.4])
        metrics = model_metrics(canonical_labels, p_mp)
        self.assertAlmostEqual(metrics["pr_auc"], average_precision_score(legacy_labels, p_mp))
        self.assertAlmostEqual(metrics["roc_auc"], roc_auc_score(legacy_labels, p_mp))


class MetadataTests(unittest.TestCase):
    def test_git_metadata_ignores_only_untracked_result_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def git(*args):
                subprocess.run(
                    ["git", *args], cwd=root, check=True,
                    text=True, capture_output=True,
                )

            git("init", "-q")
            git("config", "user.name", "Metadata Test")
            git("config", "user.email", "metadata@example.invalid")
            source = root / "tracked.py"
            source.write_text("VALUE = 1\n")
            git("add", "tracked.py")
            git("commit", "-qm", "initial")

            result_path = root / "results" / "active_learning" / "run" / "params.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text("{}\n")
            clean = git_metadata(root)
            self.assertFalse(clean["dirty"])
            self.assertEqual(clean["ignored_untracked_result_artifacts"], 1)

            untracked_source = root / "new_code.py"
            untracked_source.write_text("VALUE = 2\n")
            self.assertTrue(git_metadata(root)["dirty"])
            untracked_source.unlink()

            source.write_text("VALUE = 3\n")
            self.assertTrue(git_metadata(root)["dirty"])

    def test_params_hashes_and_protocol_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            warm = tmp / "warm.h5"
            full = tmp / "full.h5"
            make_h5(warm)
            make_h5(full)
            args = active_args(warm, full, tmp / "results" / "active_learning" / "family" / "run")
            y_warm = np.array([MP_LABEL, MR_LABEL, MR_LABEL])
            y_pool = np.array([MP_LABEL, MR_LABEL, MR_LABEL, MR_LABEL])
            y_eval = np.array([MP_LABEL, MR_LABEL])

            first = build_active_params(
                args, y_warm=y_warm, y_pool=y_pool, y_eval=y_eval,
                data_load_seconds=0.1,
            )
            second = build_active_params(
                args, y_warm=y_warm, y_pool=y_pool, y_eval=y_eval,
                data_load_seconds=9.9,
            )
            self.assertEqual(first["run"]["config_hash"], second["run"]["config_hash"])
            self.assertEqual(first["run"]["protocol_id"], second["run"]["protocol_id"])

            wasserstein_args = SimpleNamespace(**vars(args))
            wasserstein_args.strategy = "wasserstein_l2"
            wasserstein_args.query_rng_mode = "shared"
            wasserstein_args.query_objective = WASSERSTEIN_L2_QUERY_OBJECTIVE
            wasserstein_args.query_implementation_version = (
                WASSERSTEIN_L2_IMPLEMENTATION_VERSION
            )
            wasserstein = build_active_params(
                wasserstein_args, y_warm=y_warm, y_pool=y_pool, y_eval=y_eval,
                data_load_seconds=0.1,
            )
            self.assertEqual(
                wasserstein["query"]["query_objective"],
                WASSERSTEIN_L2_QUERY_OBJECTIVE,
            )
            self.assertEqual(
                wasserstein["query"]["query_implementation_version"], 2
            )
            wasserstein_args.query_implementation_version = 3
            changed_implementation = build_active_params(
                wasserstein_args, y_warm=y_warm, y_pool=y_pool, y_eval=y_eval,
                data_load_seconds=0.1,
            )
            self.assertNotEqual(
                wasserstein["run"]["config_hash"],
                changed_implementation["run"]["config_hash"],
            )
            self.assertEqual(
                wasserstein["run"]["protocol_id"],
                changed_implementation["run"]["protocol_id"],
            )

            args.strategy = "random"
            args.query_rng_mode = "shared"
            args.reweighting = "none"
            args.reweight_lambda = 1.0
            alternative = build_active_params(
                args, y_warm=y_warm, y_pool=y_pool, y_eval=y_eval,
                data_load_seconds=0.1,
            )
            self.assertNotEqual(first["run"]["config_hash"], alternative["run"]["config_hash"])
            self.assertEqual(first["run"]["protocol_id"], alternative["run"]["protocol_id"])

            args.eval_size = 5
            changed_protocol = build_active_params(
                args, y_warm=y_warm, y_pool=y_pool, y_eval=y_eval,
                data_load_seconds=0.1,
            )
            self.assertNotEqual(
                alternative["run"]["protocol_id"], changed_protocol["run"]["protocol_id"]
            )

    def test_all_arguments_are_recorded_and_status_updates_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            warm = tmp / "warm.h5"
            full = tmp / "full.h5"
            make_h5(warm)
            make_h5(full)
            out = tmp / "results" / "active_learning" / "family" / "run"
            args = active_args(warm, full, out)
            payload = build_active_params(
                args,
                y_warm=np.array([MP_LABEL, MR_LABEL]),
                y_pool=np.array([MP_LABEL, MR_LABEL]),
                y_eval=np.array([MP_LABEL, MR_LABEL]),
                data_load_seconds=0.1,
            )
            recorded = set(payload["data"]["inputs"])
            for section in ("split", "query", "reweighting", "training", "trials"):
                recorded.update(
                    key for key in payload[section]
                    if key not in {"actual", "derived"}
                )
            recorded.update(payload["other_inputs"])
            expected = set(vars(args)) - set(ACTIVE_DERIVED_KEYS) - {"out_dir"}
            self.assertEqual(recorded, expected)

            write_params(out, payload)
            self.assertTrue(update_params_status(out, "completed", total_seconds=1.25))
            saved = json.loads((out / "params.json").read_text())
            self.assertEqual(saved["run"]["status"], "completed")
            self.assertEqual(saved["timing"]["total_seconds"], 1.25)


if __name__ == "__main__":
    unittest.main()
