# om_comparison/

Cross-model per-prompt comparison that fed Figure-1/Figure-2/Figure-3 of the
writeup. Built from the committed `female5_patch_male/` and
`olmo31_rewrite_only/` condition-token-mean CSVs (under
`activation_patching/simple_patching/`).

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

These are the original (pre-palette) exploratory figures. The
publication-styled versions live in `paper_figures/`.
