# docs/

Reports and extracted-number markdown. Each starts with a Sources block and
a Schema block, then per-layer tables, then dominance summaries. Numbers
only, no interpretation.

| File | What it covers | Sections it feeds |
|---|---|---|
| `BRANCH_CHANGES.md` | Per-remote-branch divergence audit (what every branch contains vs `main`) | meta / handoff |
| `BUNDLE_DIGEST.md` | Schema and per-layer numbers from the committed `female5_patch_male/` (Qwen) and `olmo31_rewrite_only/` (OLMo) bundles brought in via PR #6 | §2.6 |
| `results_for_writeup.md` | Extraction from `raw_uploads/log_delta/` (Qwen logprob_delta) + `raw_uploads/cot_patching_olmo/` (OLMo CoT MLP); cross-checks against committed Qwen 0.398 / OLMo 0.130 L18 numbers | §2.5, §2.6, §2.7 (OLMo CoT side) |
| `results_qwen_cot_and_simple_residual.md` | Extraction from `raw_uploads/cot_patching_qwen/` (Qwen CoT MLP) + `raw_uploads/simple_prompt_residual/` (Qwen residual stream) | §2.5, §2.6, §2.7 (Qwen CoT side + residual side) |

To re-derive any number here, open the matching bundle in `raw_uploads/` —
each doc's Sources block names the exact file.
