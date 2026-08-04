# Experiment results

All experiment outputs live below this directory.  Code should write new runs
to one of these roots rather than creating result directories at repository
top level:

- `active_learning/<family>/<run>/`
- `full_data/<run>/`
- `diagnostics/<diagnostic-name>/`
- `archive/<historical-bundle>/`
- `logs/`

## Git policy

The result tree is ignored by default.  `.gitignore` opts in compact artifacts
that are useful for reproduction and later analysis:

- `params.json`, `results.json`, and multi-trial JSON files, including
  `optimizer_trace_trials.json`
- small numeric CSV summaries and final model weights/importances
- diagnostic JSON, CSV, and text summaries
- archived README files and launch scripts
- curated family-level plots under `figures/final/`

Per-run plots under `figures/generated/`, logs, large raw arrays, model
checkpoints, and full top-candidate tables stay on the local machine.  They can
be regenerated from the tracked numeric artifacts or rerun from the recorded
parameters.

Voronoi-L2 runs also write two convergence artifacts. `optimizer.log` is a
local, line-buffered text trace. `optimizer_trace_trials.json` is the compact
tracked representation, grouped by trial and solve. The optimizer minimizes
the stored `dual_objective`; the corresponding lower bound is
`dual_lower_bound = -dual_objective`, and `primal_dual_gap` is the difference
between the recorded feasible primal upper bound and that dual lower bound.
Schema-v2 traces also record the scale-free `relative_primal_dual_gap`, accepted
update and function-evaluation counts, rolling dual improvement, normalized
weight L1 change, and a termination class. `certified` means a gap or gradient
tolerance was met; `stable_not_certified` means only the explicit dual/weight
stability rule was met. Historical schema-v1 traces retain their original
objective-plateau semantics.

The small-scale Wasserstein-L2 objective audit writes `diagnostics.json`,
`diagnostics.csv`, and `summary.txt` below `diagnostics/`. Candidate-wise
regularized-OT results are called exact only when their primal/dual intervals
certify a unique winner; an overlapping comparison is recorded as unresolved.

Before adding a new artifact type, check whether it is compact and genuinely
needed for analysis.  Add a narrow allow rule instead of unignoring an entire
run directory.

## Figure layout

Experiment runners write automatically generated per-run figures to
`figures/generated/`.  Family comparison scripts write selected plots to
`figures/final/`.  Only the latter are eligible for Git tracking.

Use `git check-ignore -v --no-index <path>` to see which policy applies to a
file. A compact artifact should resolve to one of the negated (`!`) allow
rules, while generated figures and logs should resolve to an explicit
local-only rule.
