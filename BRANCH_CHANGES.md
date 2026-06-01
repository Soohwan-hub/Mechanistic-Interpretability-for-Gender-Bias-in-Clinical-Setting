# Changes from Most Recent `git pull` (2026-05-30)

Scope: everything brought into local `main` by today's `git pull` (fast-forward from `15822d9` → `c479587`). Branch `om` is created at `c479587` (same commit as `main`) — fully up to date, no commits added.

---

## Commits pulled

Two commits, both authored against `activation_patching_simple` and merged into `main` via PR #6:

| SHA | Message |
|---|---|
| `8f1e4cb` | Add simple patching without-BHC results and derived analyses. Commit `female5_patch_male` heatmaps/condition-token outputs and `olmo31_rewrite_only` analysis. |
| `c479587` | Merge pull request #6 from Soohwan-hub/activation_patching_simple |

---

## Summary

- **36 files added**, **9,948 insertions**, **0 deletions**
- **0 code changes** — every file is a generated artifact (heatmap PNG, summary CSV, or metadata JSON)
- All artifacts live under [activation_patching/simple_patching/](activation_patching/simple_patching/) in two new result bundles:
  - `female5_patch_male/` — Qwen run, female→male patching
  - `olmo31_rewrite_only/` — OLMo-3.1 run, rewrite-score-only analysis

Both bundles are produced by the same downstream analysis pipeline (identical `analysis_metadata.json` schema, identical `condition_token_strategy: "last_subtoken"`, `top_k: 15`, `score_keys: ["rewrite_scores"]`). Source run dirs in the metadata are Windows paths under `C:\Users\soohw\rtar_parent\…`, confirming both bundles were generated on Soohwan's machine and committed as results-only.

---

## Bundle 1 — `female5_patch_male/` (Qwen)

**Source metadata**: 155 artifacts, 4,340 summary rows. `score_keys = ["rewrite_scores"]`.

### `condition_token_analysis/` (4 files)
- `aggregate_by_cohort_layer_condition_token_summary.csv` (+161 lines) — per-cohort × layer × condition-token aggregation
- `aggregate_layer_condition_token_summary.csv` (+33 lines) — across-cohort aggregation
- `analysis_metadata.json` (+15 lines)
- `per_unit_layer_condition_token_summary.csv` (+4,961 lines) — per-unit rows (the raw table)

### `heatmap_plots_rewrite_aggregate_mpl/` (4 files)
Cross-cohort aggregate heatmaps, matplotlib backend, `top_k_runs: 5`, `agg_stat: "mean"`:
- `aggregate_all_rewrite_focus_condition_span_mean.png` + `.meta.json`
- `aggregate_all_rewrite_full_mean.png` + `.meta.json`

### `heatmap_plots_rewrite_per_cohort/` (10 files)
Per-cohort full-span heatmaps for: asthma, depression, multiple_sclerosis, rheumatoid_arthritis, sarcoidosis — each with `.png` + `.meta.json`.

### `heatmap_plots_rewrite_per_cohort_span/` (10 files)
Same 5 cohorts, but `x_axis: "focus_condition_span"` instead of `"full"` — heatmaps zoom into the condition-token span only.

---

## Bundle 2 — `olmo31_rewrite_only/` (OLMo-3.1)

**Source metadata**: 155 artifacts, 4,960 summary rows (slightly larger than the Qwen bundle).

### `condition_token_analysis/` (4 files)
Same schema as the Qwen bundle.

### `heatmap_plots_rewrite_aggregate_mpl/` (4 files)
Cross-cohort aggregate heatmaps — same two views (focus_condition_span_mean, full_mean) as in the Qwen bundle.

> **Note**: this pull only includes the **aggregate** heatmaps for OLMo, not the per-cohort or per-cohort-span versions. The Qwen bundle is more complete.

---

## What this pull does NOT contain

- No `.py` source changes
- No changes to `simple_patching.py`, `sae_localization.py`, `patch_targets.py`, prompt configs, or any other code
- No new branch merges aside from PR #6
- No data files, prompts, or model configs

This is a pure results-commit pull: it adds Soohwan's pre-computed analysis output for two model runs so it is browsable from `main`.

---

## File-by-file index (36 files)

```
activation_patching/simple_patching/female5_patch_male/
├── condition_token_analysis/
│   ├── aggregate_by_cohort_layer_condition_token_summary.csv     (+161)
│   ├── aggregate_layer_condition_token_summary.csv               (+33)
│   ├── analysis_metadata.json                                    (+15)
│   └── per_unit_layer_condition_token_summary.csv                (+4961)
├── heatmap_plots_rewrite_aggregate_mpl/
│   ├── aggregate_all_rewrite_focus_condition_span_mean.png       (binary)
│   ├── aggregate_all_rewrite_focus_condition_span_mean.meta.json (+13)
│   ├── aggregate_all_rewrite_full_mean.png                       (binary)
│   └── aggregate_all_rewrite_full_mean.meta.json                 (+13)
├── heatmap_plots_rewrite_per_cohort/
│   ├── aggregate_asthma_rewrite_full_mean.png|meta.json
│   ├── aggregate_depression_rewrite_full_mean.png|meta.json
│   ├── aggregate_multiple_sclerosis_rewrite_full_mean.png|meta.json
│   ├── aggregate_rheumatoid_arthritis_rewrite_full_mean.png|meta.json
│   └── aggregate_sarcoidosis_rewrite_full_mean.png|meta.json
└── heatmap_plots_rewrite_per_cohort_span/
    ├── aggregate_asthma_rewrite_focus_condition_span_mean.png|meta.json
    ├── aggregate_depression_rewrite_focus_condition_span_mean.png|meta.json
    ├── aggregate_multiple_sclerosis_rewrite_focus_condition_span_mean.png|meta.json
    ├── aggregate_rheumatoid_arthritis_rewrite_focus_condition_span_mean.png|meta.json
    └── aggregate_sarcoidosis_rewrite_focus_condition_span_mean.png|meta.json

activation_patching/simple_patching/olmo31_rewrite_only/
├── condition_token_analysis/
│   ├── aggregate_by_cohort_layer_condition_token_summary.csv
│   ├── aggregate_layer_condition_token_summary.csv
│   ├── analysis_metadata.json
│   └── per_unit_layer_condition_token_summary.csv
└── heatmap_plots_rewrite_aggregate_mpl/
    ├── aggregate_all_rewrite_focus_condition_span_mean.png|meta.json
    └── aggregate_all_rewrite_full_mean.png|meta.json
```

---

## Branch state

- `om` branch created locally from `main` at commit `c479587`
- `om` and `main` point to the same commit (no divergence)
- This file (`BRANCH_CHANGES.md`) is untracked — no commits made on `om`
