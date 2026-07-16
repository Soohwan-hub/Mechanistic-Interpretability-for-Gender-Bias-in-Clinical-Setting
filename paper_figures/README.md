# paper_figures/

Publication-ready figure renders for the writeup: standardized palette
(`#488f31` male-green / `#de3e00` female-orange / divergent green↔orange
for rewrite-score heatmaps), larger fonts, white background, horizontal
gridlines only. All values were loaded directly from existing committed
result bundles — nothing here was recomputed or invented; see "Source
data" per figure below.

| File | What it is | Source data |
|---|---|---|
| `fig1_gender_probs_by_condition.png` | Grouped bar: mean P(Male) vs P(Female) at the forced "Gender:" next-token, one pair per condition (7 conditions, x-order alphabetical) | `activation_patching/simple_patching/female_bias_run1/summary.json` → `by_condition` |
| `fig2a_toplayers_asthma.png` | Top-15 layers by mean rewrite score, asthma cohort, descending, layers 0/1/2 muted as artifact bars | `activation_patching/simple_patching/female5_patch_male/aggregate_per_layer.json` → `raw_units` (cohort=asthma), same aggregation as `plot_top_layers_bar` |
| `fig2b_toplayers_depression.png` | Same as above, depression cohort | same file, cohort=depression |
| `fig2_toplayers_combined_asthma_depression.png` | Side-by-side asthma / depression version of the two above, shared y-axis | same as fig2a/fig2b |
| `fig4_layer18_condition_token_table.png` | Booktabs-style table: layer-18 condition-token rewrite score, OLMo-7B vs Qwen2.5-7B, 5 conditions. Max/min cell bolded. | Values supplied directly by the user (team's canonical numbers — the values committed in `olmo31_rewrite_only/` and `female5_patch_male/condition_token_analysis/` in this repo are an older/different run and do **not** match; see caveat below) |
| `fig5a_mlp_heatmap_asthma_prompt1.png` | Layer × token heatmap of rewrite score, Qwen simple-prompt MLP patching, asthma prompt 1, divergent green(+)/orange(−) scale centered at 0 | `activation_patching/simple_patching/female5_patch_male/artifacts/asthma_prompt1.pkl` |
| `fig5b_mlp_heatmap_rheumatoid_arthritis_prompt1.png` | Same, rheumatoid arthritis prompt 1 | `activation_patching/simple_patching/female5_patch_male/artifacts/rheumatoid_arthritis_prompt1.pkl` |
| `fig5_mlp_heatmaps_combined_asthma_ra.png` | Stacked combined version of fig5a (top) + fig5b (bottom), shared colorbar/z-scale | same two `.pkl` files |

## Caveat on `fig4`

The layer-18 condition-token-mean values currently computable from this
repo's committed bundles (`olmo31_rewrite_only/` and `female5_patch_male/`
`condition_token_analysis/aggregate_by_cohort_layer_condition_token_summary.csv`)
do **not** match the numbers used in `fig4_layer18_condition_token_table.png`
— most values are close but not identical, and OLMo-sarcoidosis is `NaN`
in this repo's data (a known tokenizer-splitting issue, documented in
`docs/BUNDLE_DIGEST.md`) versus a real value in the table. Per the user,
their team's copy of the canonical numbers is more recent than what's
committed here; the table was built from those team-supplied values
directly, not re-derived from repo data. If a corrected/updated bundle is
committed later, regenerate `fig4` from it and remove this note.

## Regenerating

The scripts used to produce these are not currently checked into the
repo (they were run ad hoc from a scratch directory). If these figures
need to be regenerated with new data, rebuild against the source files
listed above using the same styling conventions (palette, fonts, rules)
documented in this table.
