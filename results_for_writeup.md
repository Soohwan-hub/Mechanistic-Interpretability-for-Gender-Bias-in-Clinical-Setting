# Results for writeup — sections 2.5, 2.6, 2.7

Extracted on 2026-06-10. Numbers only, no interpretation.

Sources:
- `Log_delta/` — Qwen logprob_delta MLP results (token × layer CSVs; layer 0 already excluded; 31 layers L1–L31)
- `cot_patching_results_20260609_235406/mlp_cot_patching_results/` — CoT MLP results (model identified as **OLMo-7B-0424-Instruct** via tokenizer fingerprint; 32 layers L0–L31; rewrite_score per `safe_var_matrix_*.pkl` of shape `(n_patch_targets × 32_layers)`)
- Committed bundles `female5_patch_male/` (Qwen rewrite) and `olmo31_rewrite_only/` (OLMo rewrite) for cross-check.

Schema notes:
- Qwen logprob_delta CSV columns: `layer, 0:<tok0>, 1:<tok1>, …` — one column per token position, plus the layer label `L1..L31`. Values are logprob_delta per (layer, token). Per-layer "mean" averages across token columns; "topk15_mean" averages the top 15 token columns per layer.
- CoT `safe_var_matrix` keys: `matrix` (n_targets × 32), `patch_target_labels`, `frozen_tokens`, `patch_targets`, `baseline_prob`, `duration`. Values are rewrite scores per (patch_target × layer). Per-layer "mean" averages across files (43 files = (prompt_type, condition, variant) combinations); "topk5_mean" averages the 5 file-means with the largest value at each layer.

---

## §2.6 — Qwen logprob_delta (Log_delta/)

Source: `Log_delta/fig_token_layer_heatmap_all155_logdelta.csv` (overall, 155 units aggregated by the producer). 31 layers (L1–L31), 57 token positions.

### Per-layer logprob_delta (layer 0 excluded; layer 0 not present in source CSV)

| layer | mean | topk15_mean |
|---:|---:|---:|
| 1 | 0.1350 | 0.4725 |
| 2 | -0.0120 | 0.1445 |
| 3 | 0.0139 | 0.1271 |
| 4 | 0.1573 | 0.4663 |
| 5 | -0.0204 | 0.2340 |
| 6 | 0.0590 | 0.2396 |
| 7 | 0.0163 | 0.2850 |
| 8 | 0.1699 | 0.4223 |
| 9 | 0.0500 | 0.2472 |
| 10 | -0.0586 | 0.1018 |
| 11 | 0.0915 | 0.3584 |
| 12 | 0.1469 | 0.5718 |
| 13 | 0.1123 | 0.5555 |
| 14 | 0.0032 | 0.2204 |
| 15 | 0.0978 | 0.5397 |
| 16 | 0.0885 | 0.4707 |
| 17 | -0.0075 | 0.2014 |
| **18** | **0.7786** | **1.4683** |
| 19 | 0.2585 | 1.0111 |
| 20 | 0.2683 | 1.5474 |
| 21 | 0.7157 | 2.0513 |
| 22 | 0.2162 | 1.2346 |
| 23 | -0.1054 | 0.1717 |
| 24 | 0.4261 | 1.4116 |
| 25 | 0.3432 | 1.1205 |
| 26 | 0.6899 | 2.7432 |
| 27 | 0.2629 | 0.9843 |
| 28 | 0.5569 | 2.1523 |
| 29 | 0.6580 | 2.5039 |
| **30** | **0.7914** | **2.9958** |
| 31 | 0.3304 | 1.2578 |

### Peaks

| metric | peak layer | value |
|---|---:|---:|
| logprob_delta_scores_mean | **L30** | 0.7914 |
| logprob_delta_scores_topk15_mean | **L30** | 2.9958 |

OLMo reference (from committed `olmo31_derived_logprob_metrics/aggregate_per_layer.json`): peak at L18, mean 0.7314, topk_mean 1.1267.

**Qwen does NOT peak at layer 18.** Qwen peaks at L30 by both mean (0.7914) and topk (2.9958). Layer 18 is a local secondary peak (mean 0.7786, topk 1.4683) — within ~2 % of the L30 mean but well below it on topk.

Other strong Qwen layers (mean > 0.5): L18 (0.7786), L21 (0.7157), L26 (0.6899), L29 (0.6580), L28 (0.5569), L30 (0.7914).

### Per-cohort layer-18 logprob_delta

| cohort | layer-18 mean | layer-18 topk15_mean | layer-18 max token |
|---|---:|---:|---:|
| asthma | 1.1106 | 2.1018 | 3.4297 |
| depression | 1.0176 | 1.9202 | 4.1626 |
| multiple_sclerosis | 0.4648 | 1.0185 | 2.0460 |
| rheumatoid_arthritis | 0.6516 | 1.3804 | 2.2703 |
| sarcoidosis | 0.7803 | 1.7124 | 2.5561 |

### Top 5 (cohort, prompt) units by per-cell peak layer-mean

All 155 per-cell CSVs aggregated; each peak_value is the max over layers of the within-layer token mean.

| cohort_prompt | peak_layer | peak_value |
|---|---:|---:|
| sarcoidosis_prompt28 | 18 | 1.6836 |
| asthma_prompt18 | 18 | 1.6297 |
| asthma_prompt9 | 18 | 1.5313 |
| depression_prompt28 | 18 | 1.4951 |
| depression_prompt5 | 18 | 1.4730 |

All top-5 per-cell peaks land at layer 18 — but the cross-cohort aggregate peak by mean is L30. The pattern: per-individual-unit, L18 dominates frequently; averaged across all 155 units, L21/L26/L28/L29/L30 also contribute strongly and L30 wins the global mean.

---

## §2.5 — CoT MLP (OLMo-7B-0424-Instruct)

Source: 43 `safe_var_matrix_*.pkl` files across `prompt_A/` and `prompt_C/` × 6 conditions × 1–5 variants. 32 layers (L0–L31). Each matrix is `(n_patch_targets × 32_layers)` of rewrite scores at each layer for each patched condition-token position.

### Coverage

| | conditions present (≥1 variant) | conditions ABSENT |
|---|---|---|
| prompt_A | asthma, depression, essential hypertension, multiple sclerosis, rheumatoid arthritis, sarcoidosis | **bronchitis (0 variants)** |
| prompt_C | asthma, depression, essential hypertension, multiple sclerosis, rheumatoid arthritis, sarcoidosis | **bronchitis (0 variants)** |

Variants per (prompt_type, condition):

| prompt_type | condition | variant ids |
|---|---|---|
| A | asthma | 2, 3 |
| A | depression | 2, 3, 4, 5 |
| A | essential hypertension | 2, 3, 4 |
| A | multiple sclerosis | 1, 2, 3, 4, 5 |
| A | rheumatoid arthritis | 2, 3, 4 |
| A | sarcoidosis | 2, 4, 5 |
| C | asthma | 1, 2, 4 |
| C | depression | 1, 2, 3, 4, 5 |
| C | essential hypertension | 1, 4, 5 |
| C | multiple sclerosis | 1, 2, 3, 5 |
| C | rheumatoid arthritis | 1, 2, 3, 4, 5 |
| C | sarcoidosis | 1, 3, 4 |

**Total safe_var_matrix files: 43.** Variant 1 is missing for several (prompt_A, condition) combinations.

### Per-layer rewrite_score, averaged across all 43 files (mean over files of the per-file mean over patch_targets)

Layer 0 shown separately as reference (trivial source baseline).

| layer | per-file mean across 43 files | topk5 over file-means |
|---:|---:|---:|
| **0** (REF) | **0.3806** | — |
| 1 | -0.0001 | 0.0014 |
| 2 | -0.0006 | 0.0003 |
| 3 | -0.0003 | 0.0010 |
| 4 | 0.0020 | 0.0118 |
| 5 | 0.0001 | 0.0031 |
| 6 | 0.0005 | 0.0059 |
| 7 | -0.0005 | 0.0026 |
| 8 | 0.0009 | 0.0088 |
| 9 | -0.0004 | 0.0005 |
| 10 | 0.0005 | 0.0041 |
| 11 | 0.0058 | 0.0347 |
| 12 | -0.0007 | 0.0002 |
| 13 | -0.0005 | 0.0009 |
| 14 | -0.0009 | 0.0001 |
| 15 | 0.0011 | 0.0079 |
| 16 | -0.0002 | 0.0014 |
| 17 | -0.0001 | 0.0014 |
| **18** | **0.0669** | **0.3123** |
| 19 | -0.0015 | 0.0000 |
| 20 | -0.0013 | 0.0000 |
| 21 | 0.0055 | 0.0286 |
| 22 | -0.0021 | 0.0000 |
| 23 | -0.0023 | 0.0000 |
| 24 | 0.0012 | 0.0070 |
| 25 | 0.0069 | 0.0357 |
| 26 | -0.0007 | 0.0000 |
| 27 | 0.0006 | 0.0031 |
| 28 | -0.0001 | 0.0001 |
| 29 | -0.0001 | 0.0001 |
| 30 | -0.0001 | 0.0001 |
| 31 | 0.0000 | 0.0000 |

**Layer 0 reference: 0.3806.** Layer 0 is the only layer that produces a non-trivial average rewrite score. All other layers' grid-averaged values are between -0.0023 and 0.0669; the L18 value (0.0669) is the only non-L0 layer whose averaged mean exceeds 0.01.

Negative mean values exist at: L2, L3, L5, L7, L9, L12, L13, L14, L16, L17, L19, L20, L22, L23, L26, L28, L29, L30. None are large in magnitude (worst is L23 at -0.0023).

### Cells with rewrite_score > 0.1, EXCLUDING layer 0, sorted descending

31 non-L0 cells exceed 0.1.

| layer | condition | prompt_type | variant | patch_target | rewrite_score |
|---:|---|---|---|---|---:|
| 18 | essential hypertension | C | 1 | hypertension_14 | 0.9882 |
| 18 | essential hypertension | C | 4 | hypertension_15 | 0.9591 |
| 18 | essential hypertension | A | 4 | hypertension_15 | 0.9445 |
| 18 | essential hypertension | C | 1 | hypertension_50 | 0.9252 |
| 18 | essential hypertension | A | 2 | hypertension_13 | 0.8720 |
| 18 | essential hypertension | C | 5 | hypertension_13 | 0.8436 |
| 18 | asthma | C | 1 | asthma_48 | 0.7688 |
| 18 | sarcoidosis | C | 3 | oid_14 | 0.7163 |
| 18 | sarcoidosis | C | 3 | osis_15 | 0.7081 |
| 18 | sarcoidosis | A | 4 | oid_15 | 0.6983 |
| 18 | asthma | C | 1 | asthma_13 | 0.5845 |
| 18 | sarcoidosis | C | 3 | oid_42 | 0.5642 |
| 18 | depression | C | 3 | depression_13 | 0.5292 |
| 18 | essential hypertension | C | 4 | hypertension_43 | 0.5251 |
| 18 | asthma | A | 2 | asthma_12 | 0.4154 |
| 18 | asthma | C | 4 | asthma_14 | 0.3633 |
| 18 | essential hypertension | A | 3 | hypertension_14 | 0.2583 |
| 18 | rheumatoid arthritis | C | 3 | rheumatoid_40 | 0.2120 |
| 18 | sarcoidosis | C | 3 | osis_43 | 0.2076 |
| 11 | essential hypertension | C | 4 | hypertension_15 | 0.1781 |
| 25 | essential hypertension | C | 5 | hypertension_13 | 0.1739 |
| 11 | essential hypertension | C | 5 | hypertension_13 | 0.1718 |
| 18 | sarcoidosis | C | 4 | oid_15 | 0.1620 |
| 18 | essential hypertension | C | 5 | hypertension_40 | 0.1564 |
| 25 | essential hypertension | C | 4 | hypertension_15 | 0.1556 |
| 21 | essential hypertension | C | 5 | hypertension_13 | 0.1440 |
| 25 | depression | C | 3 | depression_13 | 0.1416 |
| 25 | essential hypertension | C | 1 | hypertension_14 | 0.1339 |
| 11 | essential hypertension | A | 2 | hypertension_13 | 0.1217 |
| 18 | multiple sclerosis | C | 3 | sclerosis_14 | 0.1182 |
| 21 | essential hypertension | C | 4 | hypertension_15 | 0.1177 |

Of 31 non-L0 cells > 0.1: **24 are at layer 18, 4 at layer 25, 3 at layer 11, 2 at layer 21**. No other layers exceed 0.1 anywhere in the grid.

**Bronchitis is not represented at all** (no `safe_var_matrix` files for bronchitis in either prompt_A or prompt_C). No bronchitis cells appear in the > 0.1 list because there are no bronchitis cells.

**Essential hypertension Prompt-C dominates the L18 list above 0.5**: 6 of the top 6 cells are essential hypertension. Variants 1, 3, 4, 5 of essential hypertension Prompt-A also surface in the > 0.1 list at L18, L11, L25.

---

## §2.7 — Simple vs CoT MLP comparison (numbers side by side)

Reference figures from committed bundles (cross-check; condition_token_mean from `aggregate_layer_condition_token_summary.csv`):

| model | simple-prompt layer-18 rewrite_score (condition_token_mean) | simple-prompt layer-0 |
|---|---:|---:|
| Qwen 2.5-7B (female5_patch_male) | 0.3975 | 0.6638 |
| OLMo 3.x simple (olmo31_rewrite_only) | 0.1297 | 0.8671 |

These match the user-supplied reference (Qwen 0.398, OLMo 0.130) — no contradiction.

### CoT vs simple side by side, OLMo

| | layer 0 | layer 18 |
|---|---:|---:|
| Simple-prompt rewrite_score (olmo31_rewrite_only condition_token_mean) | 0.8671 | 0.1297 |
| CoT MLP rewrite_score (this extraction, mean across 43 files) | 0.3806 | 0.0669 |

The CoT L18 average (0.0669) is roughly half the simple-prompt L18 average (0.1297), and CoT L0 (0.3806) is roughly half the simple-prompt L0 (0.8671) for the same model class. CoT L18 cells with rewrite_score > 0.5 exist (15 of them), concentrated in essential hypertension Prompt-C, sarcoidosis Prompt-C, asthma Prompt-C, depression Prompt-C — but the grid-averaged effect is small because most (layer, target) cells across all 43 files are near zero.

### Qwen logprob_delta peak vs OLMo logprob_delta peak

| | peak layer (mean) | peak mean | peak layer (topk_mean) | peak topk_mean |
|---|---:|---:|---:|---:|
| Qwen (Log_delta/, 31-layer source, L0 excluded) | L30 | 0.7914 | L30 | 2.9958 |
| OLMo (committed olmo31_derived_logprob_metrics) | L18 | 0.7314 | L18 | 1.1267 |

Qwen's logprob_delta surface is multi-modal: L18 (0.78), L21 (0.72), L26 (0.69), L29 (0.66), L30 (0.79) all dominate. OLMo's logprob_delta surface is sharply L18-peaked with no comparable secondary band.

---

## §2.5/§2.7 — CoT residual-stream

Search: `find Log_delta cot_patching_results_20260609_235406 -iname "*resid*"` → **no matches**.

No CoT residual-stream aggregates are present in either uploaded folder. Only MLP CoT results are present (43 `safe_var_matrix_*.pkl` files plus 3,510 per-cell `*_tok*_layers_*_*.pkl` files, plus 55 PNGs, plus the 12 `pred_gender_tracker_*.json` files which contain prediction counts (Female/Other/Total) per (prompt_type, condition), not patching scores).
