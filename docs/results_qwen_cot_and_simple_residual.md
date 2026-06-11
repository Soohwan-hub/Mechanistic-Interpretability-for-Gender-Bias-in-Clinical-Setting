# Results — Qwen CoT MLP + Simple-prompt residual-stream (Sam)

Extracted on 2026-06-11. Numbers only; no interpretation.

Sources (both uploaded at repo root):
- `raw_uploads/cot_patching_qwen/mlp_cot_patching_results/` — Qwen CoT MLP activation patching
- `raw_uploads/simple_prompt_residual/` — Sam's simple-prompt residual-stream patching

Cross-check references (from already-committed bundles):
- Qwen simple-prompt MLP layer-18 condition_token_mean = **0.3975** (`female5_patch_male/condition_token_analysis/aggregate_layer_condition_token_summary.csv`)
- OLMo simple-prompt MLP layer-18 condition_token_mean = **0.1297** (`olmo31_rewrite_only/condition_token_analysis/aggregate_layer_condition_token_summary.csv`)
- OLMo CoT MLP layer-0 grid mean = 0.3806; layer-18 grid mean = 0.0669 (from `results_for_writeup.md`, prior extraction)

---

## §2.5/§2.7 — Qwen CoT MLP (`raw_uploads/cot_patching_qwen/`)

### Inventory and schema

- 4,224 `.pkl` files, 12 `pred_gender_tracker_*.json` files, 13 `.json` total
- 32 `safe_var_matrix_*.pkl` files (one per (prompt_type, condition, variant) combination with data)
- Schema of each `safe_var_matrix`:
  - `matrix`: ndarray shape `(n_patch_targets, 28_layers)` dtype float32
  - `patch_target_labels`: list of token-position labels like `'depression_23'`
  - `frozen_tokens`: list of token IDs (max id 151645 — fits Qwen 152k vocab)
  - `patch_targets`: list of `(pos, label, tier)` tuples
  - `baseline_prob`: float
  - `duration`: float
- Model fingerprint: `frozen_tokens` max id = 151645 → **Qwen 2.5-7B-Instruct** (vocab 151,643 + special tokens; OLMo-1 caps at 50k, OLMo-2 at 100k)
- Architecture: **28 layers** (matrix second dim) — matches Qwen 2.5-7B

### Coverage by (prompt_type, condition, variant)

| prompt_type | condition | variants present |
|---|---|---|
| A | asthma | 3, 5 |
| A | **bronchitis** | **none (0 variants)** |
| A | depression | 1, 3 |
| A | essential hypertension | 1, 5 |
| A | multiple sclerosis | 3, 4, 5 |
| A | rheumatoid arthritis | 1, 3, 5 |
| A | sarcoidosis | 1, 2, 3, 4, 5 |
| C | asthma | 1, 4 |
| C | **bronchitis** | **none (0 variants)** |
| C | depression | 1, 4, 5 |
| C | essential hypertension | 3 |
| C | multiple sclerosis | 1, 2 |
| C | rheumatoid arthritis | 1, 2, 3, 4, 5 |
| C | sarcoidosis | 2, 4 |

Total: 32 `safe_var_matrix` files. **Bronchitis missing from both prompt_A and prompt_C**, same as OLMo CoT. Essential hypertension Prompt_C has **only 1 variant** here (variant 3) vs. 3 variants in OLMo CoT.

### Per-layer rewrite_score averaged across all 32 files

Each row averages the per-file `matrix.mean(axis=0)` (i.e. mean across patch_targets within file) then averages across the 32 files. `topk5_of_files` is the mean of the 5 largest file-means at that layer.

| layer | file_mean | topk5_of_files |
|---:|---:|---:|
| **0** (REF) | **0.0875** | 0.4983 |
| 1 | 0.0043 | 0.0341 |
| 2 | 0.0578 | 0.3700 |
| 3 | 0.0035 | 0.0363 |
| 4 | 0.0009 | 0.0192 |
| 5 | -0.0030 | 0.0115 |
| 6 | -0.0001 | 0.0150 |
| 7 | 0.0062 | 0.0395 |
| 8 | 0.0026 | 0.0214 |
| 9 | 0.0026 | 0.0309 |
| 10 | 0.0029 | 0.0460 |
| 11 | 0.0036 | 0.0270 |
| 12 | 0.0062 | 0.0394 |
| 13 | 0.0029 | 0.0183 |
| 14 | 0.0074 | 0.0472 |
| 15 | -0.0004 | 0.0081 |
| 16 | 0.0019 | 0.0225 |
| 17 | -0.0011 | 0.0034 |
| **18** | **0.0148** | 0.0950 |
| 19 | 0.0070 | 0.0445 |
| 20 | -0.0089 | 0.0000 |
| 21 | -0.0019 | 0.0000 |
| 22 | -0.0015 | 0.0006 |
| 23 | -0.0005 | 0.0000 |
| 24 | 0.0000 | 0.0000 |
| 25 | 0.0000 | 0.0000 |
| 26 | 0.0000 | 0.0000 |
| 27 | 0.0000 | 0.0000 |

### OLMo CoT vs Qwen CoT (grid averages, same metric)

| layer | OLMo CoT mean | Qwen CoT mean |
|---:|---:|---:|
| 0 (REF) | 0.3806 | 0.0875 |
| 2 | -0.0006 | 0.0578 |
| 11 | 0.0058 | 0.0036 |
| 18 | **0.0669** | **0.0148** |
| 21 | 0.0055 | -0.0019 |
| 25 | 0.0069 | 0.0000 |

Qwen CoT L18 grid mean (0.0148) is **roughly 1/4.5 of OLMo CoT L18** (0.0669). Qwen CoT L0 reference (0.0875) is also much lower than OLMo CoT L0 (0.3806). Qwen CoT shows a higher early-layer L2 mean (0.0578) than OLMo CoT did (-0.0006).

Grid is dominated by near-zero or negative-tiny values; only L0, L2, and L18 produce non-trivial averages above 0.01.

### Qwen CoT cells with rewrite_score > 0.1, EXCLUDING layer 0 (70 cells)

| layer | condition | prompt_type | variant | patch_target | rewrite_score |
|---:|---|---|---|---|---:|
| 2 | asthma | C | 4 | asthma_96 | 1.0000 |
| 2 | asthma | C | 4 | asthma_24 | 1.0000 |
| 2 | depression | C | 4 | depression_96 | 1.0000 |
| 2 | depression | C | 5 | depression_92 | 1.0000 |
| 2 | depression | C | 5 | depression_23 | 0.9931 |
| 2 | asthma | C | 4 | asthma_339 | 0.9911 |
| 2 | asthma | A | 5 | asthma_23 | 0.9609 |
| 2 | depression | C | 4 | depression_361 | 0.9425 |
| 2 | asthma | C | 4 | asthma_217 | 0.8891 |
| 18 | depression | C | 5 | depression_23 | 0.6621 |
| 2 | depression | C | 4 | depression_24 | 0.6207 |
| 2 | depression | C | 5 | depression_217 | 0.5931 |
| 7 | depression | C | 5 | depression_23 | 0.5103 |
| 11 | depression | C | 4 | depression_96 | 0.4828 |
| 18 | depression | C | 5 | depression_217 | 0.4690 |
| 10 | depression | C | 4 | depression_96 | 0.4425 |
| 7 | depression | C | 4 | depression_96 | 0.3966 |
| 18 | depression | C | 5 | depression_47 | 0.3724 |
| 3 | depression | C | 4 | depression_24 | 0.3506 |
| 12 | depression | C | 5 | depression_23 | 0.3241 |
| 19 | depression | C | 5 | depression_23 | 0.3241 |
| 12 | depression | C | 4 | depression_96 | 0.3046 |
| 3 | depression | C | 4 | depression_96 | 0.3046 |
| 19 | depression | C | 5 | depression_47 | 0.2690 |
| 19 | depression | C | 5 | depression_217 | 0.2690 |
| 9 | depression | C | 4 | depression_96 | 0.2615 |
| 3 | depression | C | 4 | depression_49 | 0.2615 |
| 10 | depression | C | 4 | depression_24 | 0.2615 |
| 1 | depression | C | 4 | depression_24 | 0.2615 |
| 8 | asthma | C | 4 | asthma_24 | 0.2284 |
| 7 | asthma | C | 4 | asthma_217 | 0.2284 |
| 9 | asthma | C | 4 | asthma_217 | 0.2284 |
| 14 | depression | C | 5 | depression_217 | 0.2207 |
| 16 | depression | C | 4 | depression_96 | 0.2155 |
| 2 | depression | C | 4 | depression_49 | 0.2155 |
| 8 | depression | C | 4 | depression_96 | 0.2155 |
| 4 | depression | C | 4 | depression_214 | 0.2155 |
| 12 | depression | C | 5 | depression_217 | 0.2138 |
| 18 | depression | C | 5 | depression_92 | 0.2138 |
| 5 | asthma | C | 4 | asthma_217 | 0.1973 |
| 1 | asthma | C | 4 | asthma_24 | 0.1707 |
| 3 | asthma | C | 4 | asthma_217 | 0.1707 |
| 14 | asthma | C | 4 | asthma_96 | 0.1707 |
| 18 | depression | C | 4 | depression_24 | 0.1695 |
| 18 | depression | C | 4 | depression_361 | 0.1695 |
| 14 | depression | C | 4 | depression_24 | 0.1695 |
| 1 | depression | C | 4 | depression_361 | 0.1695 |
| 18 | depression | C | 4 | depression_96 | 0.1695 |
| 8 | asthma | C | 4 | asthma_217 | 0.1685 |
| 14 | asthma | C | 4 | asthma_217 | 0.1685 |
| 13 | depression | C | 5 | depression_23 | 0.1655 |
| 7 | depression | C | 5 | depression_92 | 0.1655 |
| 8 | depression | C | 5 | depression_23 | 0.1586 |
| 10 | asthma | C | 4 | asthma_49 | 0.1430 |
| 7 | asthma | C | 4 | asthma_24 | 0.1430 |
| 6 | asthma | C | 4 | asthma_217 | 0.1430 |
| 1 | asthma | C | 4 | asthma_217 | 0.1430 |
| 12 | depression | C | 4 | depression_361 | 0.1264 |
| 18 | depression | C | 4 | depression_49 | 0.1264 |
| 11 | depression | C | 4 | depression_24 | 0.1264 |
| 6 | depression | C | 4 | depression_96 | 0.1264 |
| 14 | depression | C | 4 | depression_96 | 0.1264 |
| 10 | depression | C | 4 | depression_361 | 0.1264 |
| 16 | depression | C | 4 | depression_361 | 0.1264 |
| 13 | depression | C | 4 | depression_361 | 0.1264 |
| 9 | asthma | C | 4 | asthma_24 | 0.1175 |
| 12 | asthma | C | 4 | asthma_339 | 0.1175 |
| 1 | depression | C | 4 | depression_23 | 0.1103 |
| 14 | depression | C | 5 | depression_23 | 0.1069 |
| 4 | depression | C | 5 | depression_23 | 0.1069 |

### Layer / condition / prompt-type dominance of non-L0 cells > 0.1

By layer:

| layer | n_cells > 0.1 |
|---:|---:|
| 2 | 12 |
| 18 | 8 |
| 14 | 6 |
| 12 | 5 |
| 7 | 5 |
| 1 | 5 |
| 10 | 4 |
| 3 | 4 |
| 8 | 4 |
| 9 | 3 |
| 19 | 3 |
| 11 | 2 |
| 6 | 2 |
| 13 | 2 |
| 16 | 2 |
| 4 | 2 |
| 5 | 1 |

By condition:

| condition | n_cells > 0.1 |
|---|---:|
| depression | 50 |
| asthma | 20 |
| essential hypertension | 0 |
| sarcoidosis | 0 |
| rheumatoid arthritis | 0 |
| multiple sclerosis | 0 |

By prompt_type:

| prompt_type | n_cells > 0.1 |
|---:|---:|
| C | 69 |
| A | 1 |

**Essential hypertension does NOT dominate** the Qwen CoT > 0.1 list — zero cells. Instead **depression and asthma Prompt-C** account for all 70 non-L0 cells > 0.1, with Prompt-C variants 4 and 5 driving every entry. This is the opposite of OLMo CoT (where essential hypertension Prompt-C dominated the > 0.1 list at L18).

---

## §2.6/§2.7 — Simple-prompt residual-stream (`raw_uploads/simple_prompt_residual/`)

### Inventory and schema

- 155 `.pkl` artifacts (5 cohorts × 31 prompts; one per (cohort, prompt_id))
- `aggregate_per_layer.json` and `aggregate_per_layer.csv` (28 layers × 4 stats × num_units)
- `progress.json`
- Schema of `aggregate_per_layer.json`:
  - top keys: `per_layer`, `raw_units`
  - per-row keys: `layer`, `rewrite_scores_mean`, `rewrite_scores_median`, `rewrite_scores_trimmed_mean`, `rewrite_scores_topk_mean`, `rewrite_scores_num_units`
- Schema of each artifact pickle:
  - `token_labels`: list (len ≈ 55, e.g. `['<|im_start|>_0', 'system_1', ...]`)
  - `layer_labels`: list (len 28, `[0..27]`)
  - `metadata`: dict with `cohort`, `prompt_id`, `patch_gender`, `patch_target='residual'`, `condition_name`, `corrupted_prob`, `corrupted_logprob`, `num_layers=28`, `num_tokens` (variable), `score_keys=['rewrite_scores']`
  - `rewrite_scores`: ndarray shape `(28, num_tokens)` dtype float
- Model fingerprint: `token_labels[0] = '<|im_start|>_0'` → **Qwen chat template**; `num_layers=28` → **Qwen 2.5-7B-Instruct**
- `patch_target` field in every artifact metadata = `'residual'` (not MLP)

### Coverage

Conditions: asthma, depression, multiple_sclerosis, rheumatoid_arthritis, sarcoidosis (5 cohorts).
Prompts: 1–31 each.
Total = 155 units (matches `num_units=155` in every aggregate row).
Patch direction: `patch_gender='Male'` (Male activations patched into corrupted prompts where baseline P(Male) is low; matches Sam's setup).
Sample `corrupted_prob` from one unit: 0.000805 — consistent with "baseline P(Male) ~ 0.001".

### Per-layer rewrite_score (Qwen, residual-stream, all 155 units)

| layer | mean | median | trimmed_mean | topk_mean |
|---:|---:|---:|---:|---:|
| 0 | 0.7961 | 0.9877 | 0.8647 | 0.9941 |
| 1 | 0.8329 | 0.9838 | 0.9054 | 0.9927 |
| 2 | 0.8440 | 0.9823 | 0.9156 | 0.9921 |
| 3 | 0.8669 | 0.9877 | 0.9429 | 0.9941 |
| 4 | 0.8712 | 0.9899 | 0.9545 | 0.9948 |
| 5 | 0.8800 | 0.9778 | 0.9498 | 0.9891 |
| 6 | 0.8859 | 0.9838 | 0.9658 | 0.9919 |
| 7 | 0.9020 | 0.9921 | 0.9796 | 0.9960 |
| 8 | 0.9042 | 0.9931 | 0.9815 | 0.9963 |
| 9 | 0.9054 | 0.9929 | 0.9817 | 0.9963 |
| 10 | 0.9149 | 0.9919 | 0.9827 | 0.9953 |
| 11 | 0.9265 | 0.9895 | 0.9843 | 0.9938 |
| 12 | 0.9275 | 0.9829 | 0.9786 | 0.9884 |
| 13 | 0.9426 | 0.9802 | 0.9754 | 0.9848 |
| 14 | 0.9391 | 0.9725 | 0.9686 | 0.9774 |
| 15 | 0.9414 | 0.9722 | 0.9687 | 0.9759 |
| 16 | 0.9411 | 0.9695 | 0.9668 | 0.9737 |
| 17 | 0.9395 | 0.9689 | 0.9658 | 0.9727 |
| **18** | **0.9417** | 0.9675 | 0.9649 | 0.9714 |
| 19 | 0.9216 | 0.9434 | 0.9425 | 0.9515 |
| 20 | 0.5458 | 0.5504 | 0.5520 | 0.6024 |
| **21** | **0.6281** | 0.6412 | 0.6356 | 0.6749 |
| **22** | **0.0045** | 0.0055 | 0.0046 | 0.0121 |
| 23 | 0.0003 | 0.0017 | 0.0009 | 0.0065 |
| 24 | -0.0088 | -0.0081 | -0.0081 | -0.0045 |
| 25 | 0.0035 | 0.0049 | 0.0046 | 0.0061 |
| 26 | 0.0056 | 0.0064 | 0.0063 | 0.0078 |
| 27 | -0.0008 | 0.0000 | 0.0000 | 0.0000 |

### Plateau band

- **L0–L19**: rewrite_scores_mean ranges 0.796 → 0.942 (continuously high, slight upward drift then ridge plateau L7–L19 at 0.90–0.94)
- **L20**: 0.5458 (sharp drop)
- **L21**: 0.6281 (partial recovery)
- **L22**: 0.0045 (collapse to near-zero)
- **L23–L27**: near-zero or slightly negative (-0.0088 at L24, 0.0056 at L26)

So the plateau ends sharply after L19; L20–L21 form a transition step; L22 onwards is collapsed.

### Layer-18 specifically (for side-by-side with simple-MLP)

| model + patch target | layer-18 rewrite_score |
|---|---:|
| Qwen simple-MLP (`female5_patch_male` condition_token_mean) | 0.3975 |
| OLMo simple-MLP (`olmo31_rewrite_only` condition_token_mean) | 0.1297 |
| **Qwen simple-residual (this folder, mean)** | **0.9417** |
| Qwen simple-residual (median) | 0.9675 |
| Qwen simple-residual (trimmed_mean) | 0.9649 |
| Qwen simple-residual (topk_mean) | 0.9714 |

Note: the residual aggregate is mean **across all token positions** within each unit, then averaged across 155 units. Restricting to condition-token positions (analogous to the simple-MLP method): **layer-18 mean = 0.9797** (n=160 condition-token cells across 155 units).

### Residual score at the forced "Gender:" position (final token of each prompt)

| layer | rewrite_score at final token, mean over 155 units | median |
|---:|---:|---:|
| 0–27 (uniform) | -0.0382 | -0.0012 |

The final token's rewrite_score is constant across layers within each unit (this is the patched-vs-corrupt logit-shift at the position the next token is generated for; layer-by-layer patching does not propagate to this token because of the structure of the corrupted prompt). Cross-unit mean = -0.0382. This matches Sam's "~-0.04 at the final Gender: token" exactly.

### Per-prompt content-band scores (layers 5-21, content tokens only, across 5 cohorts)

| prompt_id | mean across 5 cohorts of L5–L21 content-token mean |
|---:|---:|
| 1 | 0.9539 |
| 2 | 0.9033 |
| 3 | 0.9327 |
| 4 | 0.9394 |
| 5 | 0.9671 |
| 6 | 0.9709 |
| 7 | 0.9419 |
| 8 | 0.9513 |
| 9 | 0.9448 |
| 10 | 0.9822 |
| 11 | 0.9600 |
| 12 | 0.8588 |
| 13 | 0.9237 |
| 14 | 0.9868 |
| 15 | 0.9327 |
| 16 | 0.9286 |
| 17 | 0.9936 |
| 18 | 0.9641 |
| 19 | 0.8758 |
| 20 | 0.9332 |
| 21 | 0.9589 |
| 22 | 0.9885 |
| 23 | 0.9857 |
| 24 | 0.8706 |
| 25 | 0.9699 |
| **26** | **0.4828** |
| **27** | **0.7425** |
| 28 | 0.9849 |
| 29 | 0.9093 |
| 30 | 0.9749 |
| 31 | 0.9859 |

**Prompts 26 and 27 are the only two below 0.85**: prompt 26 = 0.4828, prompt 27 = 0.7425. Per-cohort breakdown for these two:

| cohort | prompt 26 | prompt 27 |
|---|---:|---:|
| asthma | 0.7807 | 0.4083 |
| depression | 0.5045 | 0.6388 |
| multiple_sclerosis | 0.5347 | 0.9081 |
| rheumatoid_arthritis | 0.2867 | 0.8642 |
| sarcoidosis | 0.3075 | 0.8930 |

### Logprob_delta in residual file?

`score_keys` in every artifact metadata: `['rewrite_scores']` only. **No logprob_delta or logprob_scores** are present in the residual bundle. The aggregate file confirms: only rewrite_scores_{mean, median, trimmed_mean, topk_mean, num_units}.

### Confirm/correct prior summary numbers

| claim | extracted | match? |
|---|---|---|
| baseline P(Male) ~ 0.001 | corrupted_prob = 0.000805 in sample artifact | match |
| mean rewrite ~ 0.89 across content tokens in L5–L21 | content-token mean over L5–L21 across units ranges 0.9417–0.9843 (overall mean 0.9692) | claim was conservative; actual mean is ~0.97 |
| all cells reaching ≥ 0.99 somewhere in that band | every unit's max across L5–L21 content positions exceeds 0.99 in our sample (median = 0.9931 at L8) | match |
| near-zero at final Gender: token (~ -0.04) | -0.0382 mean / -0.0012 median across units | match exactly |
| collapse after layer 21 | L22 = 0.0045 (mean drops from 0.6281 at L21) | match — sharp collapse at L22 |
| prompts 26-27 dropping to ~ 0.51-0.62 | prompt 26 = 0.4828, prompt 27 = 0.7425 (cross-cohort means of L5–L21 content) | prompt 26 lower than claimed (0.48 vs 0.51); prompt 27 higher (0.74 vs 0.62) |

No contradictions with Qwen simple-MLP 0.398 or OLMo simple-MLP 0.130 — those are different aggregations (MLP, condition-token-restricted) than this residual content-band number (0.94 at L18 mean over all tokens, 0.98 over condition tokens).

---

## Cross-check on simple-MLP reference numbers

| model + target + aggregation | layer 18 value | source |
|---|---:|---|
| Qwen simple-MLP, condition_token_mean | 0.3975 | `female5_patch_male/condition_token_analysis/aggregate_layer_condition_token_summary.csv` |
| OLMo simple-MLP, condition_token_mean | 0.1297 | `olmo31_rewrite_only/condition_token_analysis/aggregate_layer_condition_token_summary.csv` |

Both match the user-stated reference figures (0.398 / 0.130) exactly. Nothing in either uploaded folder contradicts them.
