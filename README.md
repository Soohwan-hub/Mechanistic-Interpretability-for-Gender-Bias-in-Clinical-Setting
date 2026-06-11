# Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting

> **Branch: `om`** — this README describes the layout of the `om` working branch. The original project README content from `main` is preserved at the bottom.

The `om` branch adds three things on top of `main`:

1. **Aggregated numbers and analysis docs** for §2.5, §2.6, §2.7 of the writeup (Qwen vs OLMo, simple-MLP vs CoT-MLP vs simple-residual)
2. **Raw uploaded result bundles** from collaborators (Soohwan's Qwen log_delta, OLMo CoT MLP, Qwen CoT MLP, Sam's simple-residual)
3. **The Qwen log_delta rerun notebook** + instructions for running on Lambda

Nothing here mutates files under `main` paths (`activation_patching/`, `cot_vignette/`, `localize_bias/`, `build_dataset/`, `config.py`, `model_runner.py`). Everything new is in three folders: `docs/`, `raw_uploads/`, `om_comparison/`, plus one notebook + one instructions file under `activation_patching/simple_patching/`.

---

## Top-level layout (om branch)

```
.
├── README.md                                  # this file
├── docs/                                      # all reports / extracted-number markdown
│   ├── BRANCH_CHANGES.md                      # what each remote branch contains
│   ├── BUNDLE_DIGEST.md                       # contents of the female5/olmo31 committed bundles
│   ├── results_for_writeup.md                 # §2.5/§2.6/§2.7 numbers from log_delta + OLMo CoT
│   └── results_qwen_cot_and_simple_residual.md # §2.5/§2.6/§2.7 numbers from Qwen CoT + Qwen residual
│
├── om_comparison/                             # per-prompt Qwen vs OLMo (from rewrite_score CSVs)
│   ├── per_prompt_layer_qwen.csv
│   ├── per_prompt_layer_olmo.csv
│   ├── per_prompt_layer_merged.csv            # long table on (prompt_id, layer)
│   ├── per_prompt_cohort_layer_qwen.csv
│   ├── per_prompt_cohort_layer_olmo.csv
│   ├── qwen_peak_per_prompt.csv
│   ├── olmo_peak_per_prompt.csv
│   └── figures/
│       ├── fig1_per_layer_cross_model.png     # cross-model per-layer line, L18 marked
│       ├── fig2a_heatmap_qwen.png             # prompt × layer heatmap (Qwen)
│       ├── fig2b_heatmap_olmo.png             # prompt × layer heatmap (OLMo) — shared color scale
│       └── fig3_layer18_by_cohort.png         # L18 by cohort, OLMo sarcoidosis flagged NaN
│
├── raw_uploads/                               # raw bundles from collaborators (large, not regenerated here)
│   ├── log_delta/                             # Soohwan: Qwen logprob_delta token×layer CSVs
│   │   ├── fig_token_layer_heatmap_all155_logdelta.{csv,pdf,png}
│   │   ├── layer_mean_logdelta_bar_all155.{pdf,png}
│   │   ├── per_cohort/                        # 5 × {csv,pdf,png}
│   │   └── per_cell/                          # 155 × {csv,pdf,png}
│   ├── cot_patching_olmo/                     # OLMo-7B-0424-Instruct CoT MLP results
│   │   ├── manifest.json
│   │   └── mlp_cot_patching_results/
│   │       ├── pred_gender_tracker_{A,C}_*.json   # 12 files
│   │       ├── prompt_A/{condition}/*.pkl         # safe_var_matrix_* aggregates + per-cell pickles
│   │       └── prompt_C/{condition}/*.pkl
│   ├── cot_patching_qwen/                     # Qwen 2.5-7B-Instruct CoT MLP results
│   │   └── mlp_cot_patching_results/          # same shape as cot_patching_olmo/
│   └── simple_prompt_residual/                # Sam: Qwen simple-prompt RESIDUAL-stream patching
│       ├── aggregate_per_layer.{json,csv}     # 28-layer mean/median/trimmed/topk
│       ├── progress.json
│       └── artifacts/                         # 155 per-unit pickles (5 cohorts × 31 prompts)
│
├── activation_patching/simple_patching/       # (mostly main; om adds two files)
│   ├── run_qwen_logdelta.ipynb                # om: Lambda-ready rerun with --score-keys all
│   └── RUN_INSTRUCTIONS.md                    # om: how to run the notebook
│
└── (everything else)                          # unchanged from main
```

---

## What's in `docs/`

| File | What it covers | Sections it feeds |
|---|---|---|
| `BRANCH_CHANGES.md` | Per-remote-branch divergence audit (what every branch contains vs main) | meta / handoff |
| `BUNDLE_DIGEST.md` | Schema and per-layer numbers from the committed `female5_patch_male/` (Qwen) and `olmo31_rewrite_only/` (OLMo) bundles brought in via PR #6 | §2.6 |
| `results_for_writeup.md` | Extraction from `raw_uploads/log_delta/` (Qwen logprob_delta) + `raw_uploads/cot_patching_olmo/` (OLMo CoT MLP); plus cross-checks against committed Qwen 0.398 / OLMo 0.130 L18 numbers | §2.5, §2.6, §2.7 (OLMo CoT side) |
| `results_qwen_cot_and_simple_residual.md` | Extraction from `raw_uploads/cot_patching_qwen/` (Qwen CoT MLP) + `raw_uploads/simple_prompt_residual/` (Qwen residual stream) | §2.5, §2.6, §2.7 (Qwen CoT side + residual side) |

Each results-doc starts with a Sources block and a Schema block, then per-layer tables, then dominance summaries. Numbers only, no interpretation.

---

## What's in `raw_uploads/`

Four bundles. All are READ-ONLY references — the docs in `docs/` were derived from these, so if you want to re-derive or audit, this is the source.

| Folder | Producer | Model | Patch site | Scale | What it has |
|---|---|---|---|---|---|
| `log_delta/` | Soohwan | Qwen 2.5-7B-Instruct | MLP `down_proj.output` | 5 cohorts × 31 prompts (155 units) | token×layer logprob_delta CSVs (overall + per-cohort + per-cell), PNG/PDF heatmaps; **layer 0 already excluded by the producer**; 31 layers L1–L31 |
| `cot_patching_olmo/` | Soohwan | OLMo-7B-0424-Instruct | MLP (CoT vignette flow) | 6 conditions × {A, C} × 1–5 variants = 43 `safe_var_matrix` files | per-(prompt_type, condition, variant) `safe_var_matrix_*.pkl` of shape `(n_patch_targets × 32_layers)` + 3,510 per-cell pickles + 12 `pred_gender_tracker_*.json` |
| `cot_patching_qwen/` | Soohwan | Qwen 2.5-7B-Instruct | MLP (CoT vignette flow) | 6 conditions × {A, C} × 1–5 variants = 32 `safe_var_matrix` files | same shape, 28-layer matrices |
| `simple_prompt_residual/` | Sam | Qwen 2.5-7B-Instruct | **residual stream** (not MLP) | 5 cohorts × 31 prompts (155 units) | committed `aggregate_per_layer.{json,csv}` (28-layer mean/median/trimmed/topk) + 155 artifact pickles with `(28, ntok)` rewrite_scores |

Model fingerprints in each bundle were verified independently (token IDs against tokenizer vocab; chat-template token; matrix second dim against expected n_layers). See the Source blocks in the docs for details.

---

## What's in `om_comparison/`

The cross-model per-prompt comparison that fed Figure-1/Figure-2/Figure-3 of the writeup. Built from the committed `female5_patch_male/` and `olmo31_rewrite_only/` condition-token-mean CSVs.

| File | What it is |
|---|---|
| `per_prompt_layer_qwen.csv` | Qwen prompt × layer (cohort-averaged), 837 rows (31 prompts × 27 layers, L0 excluded) |
| `per_prompt_layer_olmo.csv` | OLMo prompt × layer (cohort-averaged), 961 rows (31 prompts × 31 layers, L0 excluded) |
| `per_prompt_layer_merged.csv` | Long table `(prompt_id, layer, qwen_score, olmo_score, diff_qwen_minus_olmo)` |
| `per_prompt_cohort_layer_qwen.csv`, `per_prompt_cohort_layer_olmo.csv` | Cohort-resolved versions |
| `qwen_peak_per_prompt.csv`, `olmo_peak_per_prompt.csv` | Peak layer + peak score per prompt |
| `figures/fig1_per_layer_cross_model.png` | Cross-model per-layer line, L18 marked |
| `figures/fig2a_heatmap_qwen.png`, `figures/fig2b_heatmap_olmo.png` | Prompt × layer heatmaps, shared color scale |
| `figures/fig3_layer18_by_cohort.png` | Grouped bar of L18 by cohort, OLMo sarcoidosis drawn as NaN placeholder |

---

## Notebook: `activation_patching/simple_patching/run_qwen_logdelta.ipynb`

End-to-end Lambda-ready notebook to produce the Qwen `--score-keys all` (rewrite + logprob + logprob_delta) sweep that closes the §2.6 asymmetry (OLMo has all three score keys committed via PR #8; Qwen does not).

5 code cells: sanity → deps+HF_TOKEN check → dry-run guard (refuses to launch if the unit count is not 155) → real run streamed to log → verification (asserts all three score families present in the produced `aggregate_per_layer.json`).

Companion: `activation_patching/simple_patching/RUN_INSTRUCTIONS.md`.

**Status:** notebook is ready, not yet executed. Run produces `activation_patching/simple_patching/female5_patch_male_logdelta/` (not in repo yet).

---

## How to use this branch

| Task | Where to look |
|---|---|
| Read the §2.5/§2.6/§2.7 numbers | `docs/results_for_writeup.md` and `docs/results_qwen_cot_and_simple_residual.md` |
| Audit the cross-model per-prompt comparison | `om_comparison/per_prompt_layer_merged.csv` + `om_comparison/figures/` |
| Re-derive any number in the docs | Open the matching bundle in `raw_uploads/`; the doc's Sources block tells you which file |
| Understand what's on other contributors' branches | `docs/BRANCH_CHANGES.md` |
| Understand what came in via PR #6 and PR #8 | `docs/BUNDLE_DIGEST.md` + the `activation_patching/simple_patching/{female5_patch_male,olmo31_rewrite_only,olmo31_derived_logprob_metrics}/` dirs (these are upstream, not under `raw_uploads/`) |
| Run the missing Qwen log_delta sweep | `activation_patching/simple_patching/run_qwen_logdelta.ipynb` + `RUN_INSTRUCTIONS.md` |

---

## Status of paper deliverables

| Section | Numbers needed | Status |
|---|---|---|
| §2.5 Qwen CoT MLP | per-layer + L18 + dominance | ✓ extracted to `docs/results_qwen_cot_and_simple_residual.md` |
| §2.5 OLMo CoT MLP | per-layer + L18 + dominance | ✓ extracted to `docs/results_for_writeup.md` |
| §2.6 Qwen simple MLP rewrite | L18 condition_token_mean = 0.398 | ✓ committed in `female5_patch_male/` (referenced in both docs) |
| §2.6 OLMo simple MLP rewrite | L18 condition_token_mean = 0.130 | ✓ committed in `olmo31_rewrite_only/` (referenced in both docs) |
| §2.6 Qwen logprob_delta | peak layer + per-cohort L18 | ✓ extracted from `raw_uploads/log_delta/` in `docs/results_for_writeup.md` |
| §2.6 OLMo logprob_delta | peak layer = L18, mean 0.7314 | ✓ committed via PR #8 (`olmo31_derived_logprob_metrics/aggregate_per_layer.json`) |
| §2.6 Qwen logprob_delta with full `--score-keys all` aggregate | full 28-layer rewrite + logprob + logprob_delta in one bundle | ✗ pending Lambda run via `run_qwen_logdelta.ipynb` |
| §2.7 Qwen simple-residual (Sam) | content-band 0.94 plateau + L18 + final-token -0.04 | ✓ extracted to `docs/results_qwen_cot_and_simple_residual.md` |

---

## Original project README content (from `main`)

The text below is the README content that exists on `main`. It is preserved here for context; the `om` branch does not modify any file under the paths referenced below.

> # Files Structure
>
> ```
> localize_bias/
>     cot_vignette/
>     simple/
>
> build_dataset/
>     dataset/
>     - BHCs of depression and heart failure in json
> ```
>
> # Research Stages
>
> ## 1. Localize Bias
>
> ### Number of Runs
> #### CoT:
> (3 Prompt Types × 5 Prompt Variations × 30 BHC cases) × 2 conditions = 900 runs
> #### Simple:
>
> ### Metric: Average Rewrite Score Per Layer
>
> ## CoT patch target switch
>
> For `cot_vignette/activation_patching.ipynb`, import:
>
> `from cot_vignette.patch_targets import get_patch_tensor, set_patch_tensor, patch_token_vector, extract_head_slice, patch_token_head_slice`
>
> and replace hardcoded residual-stream lines (`llm.model.layers[layer_idx].output[0]`) with patch-target-aware calls.
>
> Supported patch targets:
> - `residual`
> - `mlp`
> - `attn`
> - `attn_head`
