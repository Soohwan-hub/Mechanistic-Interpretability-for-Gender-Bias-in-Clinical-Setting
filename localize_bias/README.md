# localize_bias/

Early bias-localization notes and the CoT residual-stream patching notebook.

| Path | What it is |
|---|---|
| `localize_bias.md` | Notes on localizing gender bias in the model |
| `cot_vignette/cot_patching_resdStream.ipynb` | **Source notebook for `paper_figures/fig9*.png`.** Runs CoT + residual-stream patching (`BASE_SAVE_DIR = /home/ubuntu/patching_results1_*` on Lambda). Its full output bundle (~14 GB) is not committed; the extracted numbers behind the figures live in `raw_uploads/cot_residual_fig9/` instead |

If you need to re-run this notebook (e.g. to get conditions/variants beyond
what `cot_residual_fig9/` covers), budget ~10+ GPU-hours per condition — see
the notebook's own markdown cells for the run breakdown.
