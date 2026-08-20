# activation_patching/

Model code and committed result bundles for activation-patching experiments.
Mostly `main`-branch content; the `om` branch adds two files under
`simple_patching/` (see root README).

| Path | What it is |
|---|---|
| `simple_patching/` | Simple (non-CoT) prompt patching — scripts, and the committed `female5_patch_male/` (Qwen), `olmo31_rewrite_only/` (OLMo), `olmo31_derived_logprob_metrics/` bundles referenced throughout `docs/` and `paper_figures/` |
| `simple_patching_results/` | Older/superseded run outputs (see individual folder names, some marked `_old`) |
| `cot_patching/` | SAE localization / linear-probe / scrubbing tooling for CoT patching (`sae_localization*.py`, `linear probe/`) |
| `simple_prompts_patching/` | Simple-prompt patching variant |
| `simple_prompts_patching_scaled_metrics.py` | Scaled-metrics aggregation script; source of the `plot_top_layers_bar` logic used to build `paper_figures/fig2*.png` |

For the Qwen/OLMo bundles specifically referenced by committed figures, see
`docs/BUNDLE_DIGEST.md` and `paper_figures/README.md`.
