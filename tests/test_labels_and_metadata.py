import json
from pathlib import Path
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
    update_params_status,
    write_params,
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
        "voronoi_l2_max_iter": 8,
        "voronoi_l2_initial_max_iter": 16,
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
