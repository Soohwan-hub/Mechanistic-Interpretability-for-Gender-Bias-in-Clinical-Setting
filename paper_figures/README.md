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
| `fig2c_toplayers_all_conditions.png` | Same as above, averaged across **all 5 cohorts** (asthma, depression, multiple sclerosis, rheumatoid arthritis, sarcoidosis; 155 units) | same file, all `raw_units` |
| `fig2_toplayers_combined_asthma_depression.png` | Side-by-side asthma / depression version, shared y-axis | same as fig2a/fig2b |
| `fig4_layer18_condition_token_table.png` | Booktabs-style table: layer-18 condition-token rewrite score, OLMo-7B vs Qwen2.5-7B, 5 conditions. Max/min cell bolded. | Values supplied directly by the user (team's canonical numbers — the values committed in `olmo31_rewrite_only/` and `female5_patch_male/condition_token_analysis/` in this repo are an older/different run and do **not** match; see caveat below) |
| `fig5a_mlp_heatmap_asthma_prompt1.png` | Layer × token heatmap of rewrite score, Qwen simple-prompt MLP patching, asthma prompt 1, divergent green(+)/orange(−) scale centered at 0 | `activation_patching/simple_patching/female5_patch_male/artifacts/asthma_prompt1.pkl` |
| `fig5b_mlp_heatmap_rheumatoid_arthritis_prompt1.png` | Same, rheumatoid arthritis prompt 1 | `activation_patching/simple_patching/female5_patch_male/artifacts/rheumatoid_arthritis_prompt1.pkl` |
| `fig5_mlp_heatmaps_combined_asthma_ra.png` | Stacked combined version of fig5a (top) + fig5b (bottom), shared colorbar/z-scale | same two `.pkl` files |
| `fig9_cot_residual_by_condition_promptA_var1.png` | CoT **residual-stream** rewrite score, layer × token, four conditions (depression, multiple sclerosis, rheumatoid arthritis, sarcoidosis), Qwen Prompt A / var1. 28 layers; 15-token window ending just before the "You must start with" instruction, matching the source figure. Shared symmetric [-1, 1] scale. | `cot_resdstream_results_20260609_222032/patching_results1_<condition>/VIGNETTE_PROMPT_A/result_var1.pkl` → `rewrite_matrix` + `token_labels` |
| `fig9b_cot_residual_by_condition_promptA_var2.png` | Same, Prompt A / var2. | same bundle, `result_var2.pkl` |
| `fig11_residual_plateau_qwen.png` | Per-layer rewrite score (mean / median / top-k mean) for Qwen simple-prompt **residual-stream** patching, 28 layers × 155 units. Mid-layer plateau L5–L21 shaded; L18 and the L22 collapse marked. The "mean across token positions × 155 units" aggregation detail belongs in the caption, not the axis label. | `raw_uploads/simple_prompt_residual/aggregate_per_layer.csv` → `rewrite_scores_{mean,median,topk_mean}` |

## Display filtering (fig5 only)

The heatmaps are cropped for legibility in a paper column — **no values were
altered**, only rows/columns hidden:

- **Layers:** restricted to **L3–L21**, the interpretable band. Layers 0–2 are
  the early-layer saturating artifact the paper already treats as noise (the
  same bars muted in fig2), and layers 22–27 carry no signal in either
  condition.
- **Tokens:** kept the **top 18 tokens by peak |rewrite score| within L3–L21**.
  Ranking inside the band matters: layer 0 saturates near 1.0 on almost every
  token, so ranking that included it readmitted 47 of 55 columns and defeated
  the decluttering. Asthma 55 → 18 tokens; rheumatoid arthritis 58 → 18.
- Both panels use the same layer range and an identical symmetric color scale,
  so the two conditions stay directly comparable.
- Token tick labels are reformatted from `' patient_16'` to `patient (16)`.

**Known limitation — the rheumatoid-arthritis panel renders nearly blank.**
That is a faithful result, not a plotting bug: RA's entire effect sits in the
excluded artifact layers 0 and 2, and its largest in-band value is 0.085
(vs 0.99 for asthma). This is consistent with RA's near-zero scores in fig2 and
fig4. If a visually informative RA panel is needed, either annotate the blank
panel explicitly or show RA on its own rescaled colorbar — do not silently
reintroduce layers 0–2 for RA only, which would make the two panels
incomparable.


## Note on `fig9` provenance

The source bundle (`cot_resdstream_results_20260609_222032/`, ~14 GB) is **not
committed** — it was produced by `localize_bias/cot_vignette/cot_patching_resdStream.ipynb`
on a Lambda instance at `/home/ubuntu/patching_results1_*` and shared as a zip.
It is CoT + residual-stream, distinct from the MLP CoT bundles under
`raw_uploads/cot_patching_{qwen,olmo}/` — do not substitute those.

Token selection is the 15 prompt tokens ending just before the "You must start
with" instruction, plotted in **descending** position order left-to-right
(`._156` -> `consistent_142`) exactly as the source figure does (the source figure's title says "Top-15 tokens (max|RS|)", but
ranking by max|RS| does not reproduce its axes; the contiguous window does, and
was verified against all four conditions).

Verified: 28 layers, ~180-token prompts, signal saturating near 1.0 across
L0-21 and collapsing at L22 (max |RS| 0.0002-0.045 for L22-27).

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
