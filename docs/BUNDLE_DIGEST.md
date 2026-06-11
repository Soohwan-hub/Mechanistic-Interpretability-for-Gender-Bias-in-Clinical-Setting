# Bundle Digest — `female5_patch_male/` and `olmo31_rewrite_only/`

Both bundles live under [activation_patching/simple_patching/](activation_patching/simple_patching/) and were committed via PR #6 in today's pull. This digest is built directly from the CSVs and every JSON / `.meta.json` in each bundle. Branch: `om`.

---

## A. Hook site / what was patched

**Neither bundle's JSON metadata records the hook site, model name, or scaling factor explicitly.** The metadata files only describe the *analysis pass* over pre-existing per-prompt `.pkl` artifacts; they do not include the run-time configuration of the patching itself.

What we *can* infer from the source code and surrounding context:

- The artifacts (one `.pkl` per `(cohort, prompt_id)`) were produced by [activation_patching/simple_patching/simple_patching.py](activation_patching/simple_patching/simple_patching.py). That script patches **MLP `down_proj.output`** — the output of each transformer block's MLP sub-layer, before it is added back to the residual stream. There is no `--patch-target` flag in `simple_patching.py`, so this is the only hook site available to it.
- Model is hardcoded as `Qwen/Qwen2.5-7B-Instruct` in `simple_patching.py` (line 36). The `female5_patch_male/` bundle is from this Qwen run.
- The `olmo31_rewrite_only/` bundle's folder name implies OLMo-3.1 was the underlying model, but the metadata does not name it directly. OLMo-3.1 (32-layer architecture) is consistent with the 32 layers visible in the OLMo aggregate CSV (rows 0–31) vs. 28 layers (rows 0–27) for the Qwen bundle.
- No "scaling factor" appears anywhere in the metadata. The analysis pipeline aggregates raw rewrite scores; there is no `scale=N` or `multiplier` field. (The `task3-layer18-vignettes` branch has a separate "scaled metrics" script, but its outputs are not in either of these two bundles.)

| Bundle | Implied model | Layers present | Hook site |
|---|---|---|---|
| `female5_patch_male/` | Qwen/Qwen2.5-7B-Instruct | 0–27 (28 layers) | `model.layers[L].mlp.down_proj.output` |
| `olmo31_rewrite_only/` | OLMo-3.1 (~7B, 32-layer) | 0–31 (32 layers) | `mlp.down_proj.output` (assumed same pipeline) |

---

## B. CSV structure — both bundles

### `aggregate_layer_condition_token_summary.csv`
Across-cohort per-layer summary. Columns:
```
score_key, layer, condition_token_mean, condition_token_max,
full_token_max, full_token_topk_mean, full_token_mean, num_units
```
Row count: female5 = 28 layers + header = 29; OLMo = 32 layers + header = 33.

### `aggregate_by_cohort_layer_condition_token_summary.csv`
Per-(cohort, layer) summary. Columns:
```
cohort, score_key, layer, condition_token_mean, condition_token_max,
full_token_max, full_token_topk_mean, full_token_mean, num_units
```
Row count: female5 = 5 cohorts × 28 layers + header = 141; OLMo = 5 cohorts × 32 layers + header = 161.

### `per_unit_layer_condition_token_summary.csv`
One row per `(artifact, layer)`. Columns:
```
artifact, cohort, prompt_id, condition_name, score_key, layer,
condition_token_indices, condition_token_labels,
condition_token_mean, condition_token_max,
full_token_max, full_token_topk_mean, full_token_mean
```
Artifact id is e.g. `asthma_prompt1.pkl`. Row count: female5 = 4,340 data rows (155 artifacts × 28 layers); OLMo = 4,960 (155 artifacts × 32 layers).

Both bundles cover 155 artifacts = **5 cohorts × 31 prompt variants** (asthma, depression, multiple_sclerosis, rheumatoid_arthritis, sarcoidosis).

---

## C. Bundle 1 — `female5_patch_male/` (Qwen)

### C.1 `analysis_metadata.json` (condition_token_analysis)

| Field | Value |
|---|---|
| `source_run_dir` | `c:\Users\soohw\rtar_parent\Simple_Patching\…\female5_patch_male` |
| `num_artifacts` | 155 |
| `num_rows` | 4340 |
| `condition_token_strategy` | `last_subtoken` |
| `top_k` | 15 |
| `score_keys` | `["rewrite_scores"]` |

### C.2 Heatmap `.meta.json` sidecars (8 total)

All share: `agg_stat: "mean"`, `aggregate_align: "pad"`, `top_k_runs: 5`, `render_backend: "matplotlib"`.

**Aggregate (cross-cohort) heatmaps** — `heatmap_plots_rewrite_aggregate_mpl/`:
- `aggregate_all_rewrite_full_mean`: `x_axis="full"`, all 155 artifacts used.
- `aggregate_all_rewrite_focus_condition_span_mean`: `x_axis="focus_condition_span"`, all 155 artifacts.

These two aggregate sidecars do **not** carry `exclude_layers`.

**Per-cohort heatmaps** — `heatmap_plots_rewrite_per_cohort/` and `heatmap_plots_rewrite_per_cohort_span/` (5 each, one per cohort):
- Each uses `num_artifacts_in_batch: 31`, `aggregate_from: "token_grid"`, `aggregate_split_cohorts: true`.
- `x_axis="full"` for the per_cohort group, `x_axis="focus_condition_span"` for the per_cohort_span group.
- All per-cohort sidecars carry `"exclude_layers": [0]` — layer 0 is dropped from the plots (it dominates as the trivial source-of-truth baseline; see CSV values below).

### C.3 Other JSON in the bundle

- `progress.json`: 155 entries listing every `(cohort, prompt_id)` already processed. Confirms the 5×31 sweep completed without skips. No config fields.
- `aggregate_per_layer.json`: per-layer `mean / median / trimmed_mean / topk_mean` of `rewrite_scores` for each of 28 layers. Schema mirrors what `simple_patching.py` writes natively. No model/hook info.

### C.4 Per-layer `condition_token_mean` rewrite scores (Qwen, all cohorts)

| Layer | condition_token_mean | full_token_topk_mean |
|---|---|---|
| 0 | 0.6638 | 0.9917 |
| 1 | 0.0086 | 0.0102 |
| **2** | **0.3832** | 0.4546 |
| 3 | 0.0096 | 0.0126 |
| 4 | 0.0306 | 0.0173 |
| 5 | 0.0076 | 0.0084 |
| 6 | 0.0626 | 0.0254 |
| **7** | **0.1892** | 0.0505 |
| 8 | 0.0286 | 0.0141 |
| 9 | -0.0001 | 0.0094 |
| 10 | 0.0204 | 0.0155 |
| 11 | 0.0108 | 0.0092 |
| 12 | 0.0590 | 0.0203 |
| 13 | 0.0360 | 0.0124 |
| 14 | 0.0247 | 0.0186 |
| 15 | 0.0174 | 0.0120 |
| 16 | 0.0031 | 0.0057 |
| 17 | 0.0351 | 0.0209 |
| **18** | **0.3975** | 0.0990 |
| **19** | **0.1958** | 0.0573 |
| 20 | -0.0194 | 0.0109 |
| 21 | 0.0459 | 0.0281 |
| 22 | -0.0016 | 0.0021 |
| 23 | 0.0007 | 0.0023 |
| 24 | 0.0010 | 0.0027 |
| 25 | 0.0013 | 0.0015 |
| 26 | 0.0004 | 0.0046 |
| 27 | 0.0000 | 0.0026 |

**Strongest layers (Qwen, cross-cohort by `condition_token_mean`)**: layer 0 (0.664), then **layer 18 (0.398)**, then **layer 2 (0.383)**, then **layer 19 (0.196)**, then **layer 7 (0.189)**. Layer 0 reflects the trivial copy/source effect; among non-trivial layers, **layer 18 is the strongest, with layer 2 close behind and a second smaller peak at layer 19**.

### C.5 Per-cohort strongest layers (Qwen, excluding layer 0)

| Cohort | Top layer | Score | 2nd layer | Score | 3rd layer | Score |
|---|---|---|---|---|---|---|
| asthma | **2** | 0.8988 | 18 | 0.8331 | 19 | 0.6671 |
| depression | **2** | 0.7039 | 18 | 0.5594 | 19 | 0.1198 |
| multiple_sclerosis | **7** | 0.2670 | 18 | 0.2394 | 3 | 0.0429 |
| rheumatoid_arthritis | **18** | 0.0115 | 2 | 0.0103 | 7 | 0.0011 |
| sarcoidosis | **18** | 0.1902 | 2 | 0.0441 | 19 | 0.0310 |

Pattern: **layer 18 is the universal mid-late peak across all 5 cohorts**, with layer 2 dominant for asthma/depression (early bias signal) and layer 7 dominant only for multiple_sclerosis. Rheumatoid arthritis is overall extremely weak — peak is just 0.0115.

---

## D. Bundle 2 — `olmo31_rewrite_only/` (OLMo-3.1)

### D.1 `analysis_metadata.json`

| Field | Value |
|---|---|
| `source_run_dir` | `c:\Users\soohw\rtar_parent\…\olmo31_rewrite_only` |
| `num_artifacts` | 155 |
| `num_rows` | 4960 |
| `condition_token_strategy` | `last_subtoken` |
| `top_k` | 15 |
| `score_keys` | `["rewrite_scores"]` |

Identical schema to the Qwen bundle; only `num_rows` differs (32 layers × 155 = 4,960 vs. 28 × 155 = 4,340).

### D.2 Heatmap `.meta.json` sidecars (2 total)

Only the cross-cohort aggregate heatmaps are committed for OLMo (no per-cohort heatmaps):
- `aggregate_all_rewrite_full_mean`: `x_axis="full"`, 155 artifacts.
- `aggregate_all_rewrite_focus_condition_span_mean`: `x_axis="focus_condition_span"`, 155 artifacts.

Shared fields: `agg_stat: "mean"`, `aggregate_align: "pad"`, `top_k_runs: 5`, `render_backend: "matplotlib"`. No `exclude_layers` set.

### D.3 Per-layer `condition_token_mean` rewrite scores (OLMo, all cohorts)

| Layer | condition_token_mean | full_token_topk_mean |
|---|---|---|
| 0 | 0.8671 | 0.9483 |
| 1 | -0.0004 | 0.0009 |
| 2 | 0.0007 | 0.0009 |
| 3 | 0.0002 | 0.0009 |
| 4 | -0.0008 | 0.0013 |
| 5 | -0.0006 | 0.0015 |
| 6 | -0.0010 | 0.0016 |
| 7 | -0.0004 | 0.0009 |
| 8 | 0.0023 | 0.0012 |
| 9 | -0.0005 | 0.0005 |
| 10 | -0.0007 | 0.0008 |
| 11 | 0.0022 | 0.0012 |
| 12 | -0.0003 | 0.0008 |
| 13 | -0.0000 | 0.0003 |
| 14 | -0.0007 | 0.0006 |
| 15 | -0.0007 | 0.0006 |
| 16 | -0.0005 | 0.0009 |
| 17 | 0.0010 | 0.0012 |
| **18** | **0.1297** | 0.0236 |
| 19 | -0.0004 | 0.0006 |
| 20 | -0.0011 | 0.0010 |
| 21 | 0.0046 | 0.0032 |
| 22 | -0.0019 | 0.0003 |
| 23 | -0.0017 | 0.0002 |
| 24 | 0.0004 | 0.0007 |
| 25 | 0.0021 | 0.0004 |
| 26 | -0.0009 | 0.0014 |
| 27 | 0.0003 | 0.0002 |
| 28 | -0.0000 | 0.0002 |
| 29 | 0.0000 | 0.0017 |
| 30 | 0.0000 | 0.0010 |
| 31 | 0.0000 | 0.0000 |

**Strongest layers (OLMo, cross-cohort)**: layer 0 (0.867 — trivial source baseline), then a single sharp peak at **layer 18 (0.130)**. Every other layer is < 0.005 in absolute value. This is a much cleaner localization than Qwen — OLMo concentrates the bias signal almost entirely at layer 18.

### D.4 Per-cohort strongest layers (OLMo, excluding layer 0)

| Cohort | Top layer | Score | 2nd layer | Score |
|---|---|---|---|---|
| asthma | **18** | 0.4200 | 21 | 0.0136 |
| depression | **18** | 0.0743 | 11 | 0.0004 |
| multiple_sclerosis | **18** | 0.0194 | 21 | 0.0031 |
| rheumatoid_arthritis | **18** | 0.0052 | 21 | 0.0015 |
| sarcoidosis | N/A (NaN) | — | — | — |

> **OLMo sarcoidosis has `NaN` for every layer's `condition_token_mean`.** All 32 sarcoidosis rows show `nan,nan` in the per-cohort aggregate. `full_token_max` and `full_token_topk_mean` are non-null, so the run completed — the issue is specific to the condition-token strategy (`last_subtoken`) for sarcoidosis under OLMo's tokenization. Likely cause: the OLMo tokenizer splits "sarcoidosis" such that the recorded `condition_token_indices` are inconsistent or empty, propagating NaN through the per-cohort mean. Worth flagging — Qwen sarcoidosis numbers are clean (layer 18 = 0.190).

OLMo confirms the **layer 18 result much more strongly than Qwen**: it is the only non-trivial signal in the entire model, and it appears in 4 of 5 cohorts.

---

## E. Cross-bundle summary

| | Qwen (`female5_patch_male/`) | OLMo (`olmo31_rewrite_only/`) |
|---|---|---|
| Artifacts | 155 (5 × 31) | 155 (5 × 31) |
| Per-unit rows | 4,340 | 4,960 |
| Layers | 28 (0–27) | 32 (0–31) |
| Condition-token strategy | `last_subtoken` | `last_subtoken` |
| `top_k` (analysis) | 15 | 15 |
| `top_k_runs` (heatmaps) | 5 | 5 |
| Score keys | `["rewrite_scores"]` | `["rewrite_scores"]` |
| Strongest non-trivial layer (overall) | **18** (0.398) | **18** (0.130) |
| Secondary peaks | 2 (0.383), 19 (0.196), 7 (0.189) | 21 (0.0046), 11 (0.0022) |
| Per-cohort heatmaps committed | yes (×2 views) | no (only aggregate) |
| Scaling factor recorded | none | none |
| Hook site recorded | none (inferred MLP `down_proj.output` from source) | same |
| Sarcoidosis status | clean (layer 18 = 0.190) | NaN across all layers |

**Bottom line for cross-model comparison:** layer 18 is the consistent peak in both models on the same task, but Qwen also shows substantial early (layer 2) and secondary mid (layer 19, layer 7) activity that OLMo does not. OLMo's bias representation is more localized; Qwen's is more distributed. This is the cleanest data point in the repo for cross-model bias-strength comparison — it is also the closest match to the Task 4 framing Soohwan described in Slack.

---

## F. Branch state

- Branch `om` is current, at commit `c479587` (same as `main`).
- This file is untracked; no commits made.
