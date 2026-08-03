# Hard Voronoi No-Class-Balance Add-On

Setup:
- Strategy: `kmedianpp`
- Model: XGBoost deep config used by the folder's existing runs
- Reweighting: hard Voronoi, representing the unregularized lambda=0 endpoint
- Class balancing: none
- Weight normalization: fixed total train weight 10000
- Eval source: full heldout
- Total queries: 150
- Eval interval: 10 queries
- Seeds: 5

Final 150-query average precision:

| Run | Mean AP | Std AP |
| --- | ---: | ---: |
| lambda=0 hard | 0.2063 | 0.0086 |
| lambda=1e2 | 0.3062 | 0.0042 |
| lambda=1e3 | 0.3020 | 0.0056 |
| lambda=1e4 | 0.2983 | 0.0033 |
| lambda=1e5 | 0.2772 | 0.0023 |
| lambda=1e6 | 0.2546 | 0.0038 |
| lambda=inf uniform | 0.2518 | 0.0061 |

Final effective sample size:

| Run | Mean ESS |
| --- | ---: |
| lambda=0 hard | 126.477 |
| lambda=1e2 | 4843.68 |
| lambda=1e3 | 116429 |
| lambda=1e4 | 342994 |
| lambda=1e5 | 383465 |
| lambda=1e6 | 384533 |
| lambda=inf uniform | 384546 |

Conclusion:

Hard Voronoi reweighting is worse than the regularized Voronoi-L2 sweep in this
no-class-balance setting, and it is also worse than the uniform no-reweighting
baseline.  The likely proximate issue is weight concentration: hard Voronoi has
ESS near 126 at 150 queries, far below even lambda=1e2.  In this experiment,
some L2 regularization is necessary for useful XGBoost training weights.
