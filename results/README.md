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

- `params.json`, `results.json`, and multi-trial JSON files
- small numeric CSV summaries and final model weights/importances
- diagnostic JSON, CSV, and text summaries
- archived README files and launch scripts
- curated family-level plots under `figures/final/`

Per-run plots under `figures/generated/`, logs, large raw arrays, model
checkpoints, and full top-candidate tables stay on the local machine.  They can
be regenerated from the tracked numeric artifacts or rerun from the recorded
parameters.

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
