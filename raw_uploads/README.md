# raw_uploads/

Raw result bundles from collaborators. **Read-only references** — everything
in `docs/` was derived from these; if you want to re-derive or audit a
number, this is the source.

| Folder | Producer | Model | Patch site | Scale | What it has |
|---|---|---|---|---|---|
| `log_delta/` | Soohwan | Qwen 2.5-7B-Instruct | MLP `down_proj.output` | 5 cohorts × 31 prompts (155 units) | token×layer logprob_delta CSVs (overall + per-cohort + per-cell), PNG/PDF heatmaps; layer 0 already excluded by the producer; 31 layers L1–L31 |
| `cot_patching_olmo/` | Soohwan | OLMo-7B-0424-Instruct | MLP (CoT vignette flow) | 6 conditions × {A, C} × 1–5 variants = 43 `safe_var_matrix` files | per-(prompt_type, condition, variant) `safe_var_matrix_*.pkl` `(n_patch_targets × 32_layers)` + 3,510 per-cell pickles + 12 `pred_gender_tracker_*.json` |
| `cot_patching_qwen/` | Soohwan | Qwen 2.5-7B-Instruct | MLP (CoT vignette flow) | 6 conditions × {A, C} × 1–5 variants = 32 `safe_var_matrix` files | same shape, 28-layer matrices |
| `simple_prompt_residual/` | Sam | Qwen 2.5-7B-Instruct | **residual stream** (not MLP) | 5 cohorts × 31 prompts (155 units) | `aggregate_per_layer.{json,csv}` (28-layer mean/median/trimmed/topk) + 155 artifact pickles with `(28, ntok)` rewrite_scores |
| `cot_residual_fig9/` | (extracted, see its own README) | Qwen 2.5-7B-Instruct | **residual stream**, CoT prompts | 8 (condition × var) panels | `rewrite_matrix` + `token_labels` behind `paper_figures/fig9*.png`; ~550 KB, distilled from a 14 GB Lambda run that is not committed |

Model fingerprints in each bundle were verified independently (token IDs
against tokenizer vocab; chat-template token; matrix second dim against
expected n_layers) — see the Source blocks in `docs/*.md`.

Do not confuse `cot_patching_{qwen,olmo}/` (**MLP** patch site) with
`simple_prompt_residual/` or `cot_residual_fig9/` (**residual-stream** patch
site) — mixing these up produced several incorrect figures earlier in this
project; see `paper_figures/README.md` for the details.
