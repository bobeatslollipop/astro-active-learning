# Office-Home round-1 baseline and lambda campaign

## Question

Compare source-only, random, finite-pool Wasserstein, hard-Voronoi, and regularized-Wasserstein weighting across all 12 directed Office-Home tasks, and select one global lambda using seed-0 heldout cross-entropy.

## Configuration

- Git commit used for execution: `89dee303ccd9661a59fb968ffd5328286f407070`
- Frozen row-wise L2-normalized ResNet-50 V1 features; 65-class weighted softmax.
- Class-stratified 80/20 heldout split plus full-target transductive evaluation.
- Query budget 150; seeds [0, 1, 2, 3, 4]; no class balancing.
- Lambda grid [10.0, 100.0, 1000.0, 10000.0]; lambda_0 and scale-setting OT were not computed.
- Selection and reweighting geometry used the complete label-free target public pool.

## Results

- Completed runs: 336/336
- Failed runs: 0
- Campaign wall time: 720.68 seconds (12.01 minutes).
- Classifier fit time across all runs: mean 0.891 seconds, median 0.859 seconds, maximum 1.327 seconds.
- All query plans contain exactly 150 unique IDs from the public target pool. Random and Wasserstein query IDs are reused exactly across their paired weighting methods.
- Global lambda status: `selected`
- Selected lambda: `10.0`
- Full numerical results: `results/.../aggregates/per_run_results.csv` and `lambda_calibration_seed0.csv`.

Seed-0 lambda calibration used the same 12 directed pairs for every lambda:

| Lambda | Mean heldout CE | Mean heldout top-1 | Mean heldout macro accuracy |
| ---: | ---: | ---: | ---: |
| 10 | 1.3152 | 0.6698 | 0.6421 |
| 100 | 1.3240 | 0.6652 | 0.6384 |
| 1000 | 1.3506 | 0.6590 | 0.6326 |
| 10000 | 1.3687 | 0.6545 | 0.6293 |

Seeds 1--4, pooled equally over 48 pair/seed runs per method:

| Method | Heldout CE | Heldout top-1 | Heldout macro | Full-target CE | Full-target top-1 | Full-target macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source-only + uniform | 1.6068 | 0.5967 | 0.5806 | 1.6108 | 0.5962 | 0.5820 |
| random queries + uniform | 1.3921 | 0.6448 | 0.6225 | 1.3562 | 0.6564 | 0.6369 |
| random queries + regularized Wasserstein, lambda 10 | 1.3381 | 0.6581 | 0.6297 | 1.2948 | 0.6720 | 0.6478 |
| Wasserstein-greedy queries + uniform | 1.3387 | 0.6568 | 0.6335 | 1.2905 | 0.6739 | 0.6517 |
| Wasserstein-greedy queries + hard Voronoi | **1.3099** | **0.6740** | **0.6412** | **1.2555** | **0.6907** | **0.6604** |

## Failures

No failed run records. All 96 regularized-Wasserstein solves succeeded on their first attempt, so no doubled-ceiling retry was needed. Sixty met the strict convergence certificate; 36 were recorded as `stable_not_certified`. Those 36 retain their diagnostics and are not described as strictly certified.

## Verdict

Lambda 10 is the unique heldout-CE winner on the complete 12-pair common-valid set. For seeds 1--4, regularized reweighting improves the random-query baseline, while Wasserstein-greedy plus hard Voronoi is the strongest pooled method on both heldout and full-target transductive metrics.

## Next step

Inspect per-pair heterogeneity and the 36 stable-but-not-certified regularized traces before expanding the lambda grid or query budgets. Future campaign reports should continue to include both heldout and full-target transductive metrics.
