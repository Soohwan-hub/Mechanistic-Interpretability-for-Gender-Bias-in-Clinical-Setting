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
| `fig2_token_association_table_nosubtitle.png` | Same table as `fig4_layer18_condition_token_table.png`, regenerated with the grey source-bundle subtitles (`olmo31_rewrite_only` / `female5_patch_male`) removed from under the column headers. Values, bolding, zebra shading, rules and 2538px width are unchanged; height drops 1335 -> 1045 because the header is one line instead of two. | Same team-supplied values as fig4, transcribed rather than recomputed (see caveat below) |
| `fig5a_mlp_heatmap_asthma_prompt1.png` | Layer × token heatmap of rewrite score, Qwen simple-prompt MLP patching, asthma prompt 1, divergent green(+)/orange(−) scale centered at 0 | `activation_patching/simple_patching/female5_patch_male/artifacts/asthma_prompt1.pkl` |
| `fig5b_mlp_heatmap_rheumatoid_arthritis_prompt1.png` | Same, rheumatoid arthritis prompt 1 | `activation_patching/simple_patching/female5_patch_male/artifacts/rheumatoid_arthritis_prompt1.pkl` |
| `fig5_mlp_heatmaps_combined_asthma_ra.png` | Stacked combined version of fig5a (top) + fig5b (bottom), shared colorbar/z-scale | same two `.pkl` files |
| `fig5_mlp_heatmaps_combined_asthma_ra_redpink.png` | Alternate-palette render of the combined fig5 heatmap: divergent pink(+)/dark-red(−) scale centered at 0, replacing the green/orange ramp. **Produced by recolouring the unfiltered fig5 render (commit `f0a910e`) pixel-by-pixel** via `recolor_fig5_palette.py` — nothing is re-plotted, so tokens, values, layer range, layout and fonts are preserved bit-for-bit; only heatmap and colorbar pixels differ. Note this means it shows the **unfiltered** L0–L21 / all-token view, not the L3–L21 / top-18 filtering that the current `fig5*.png` files carry. | recoloured from `f0a910e:paper_figures/fig5_mlp_heatmaps_combined_asthma_ra.png` |
| `fig9_cot_residual_by_condition_promptA_var1.png` | CoT **residual-stream** rewrite score, layer × token, four conditions (depression, multiple sclerosis, rheumatoid arthritis, sarcoidosis), Qwen Prompt A / var1. 28 layers; 15-token window ending just before the "You must start with" instruction, matching the source figure. Shared symmetric [-1, 1] scale, **orange positive / green negative** (ramp reversed relative to the other figures). | `cot_resdstream_results_20260609_222032/patching_results1_<condition>/VIGNETTE_PROMPT_A/result_var1.pkl` → `rewrite_matrix` + `token_labels` |
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



## Alternate palette (`_redpink`)

`fig5_mlp_heatmaps_combined_asthma_ra_redpink.png` restates the combined fig5
heatmap in a divergent red/pink scale. The mapping mirrors the original ramp's
structure, keeping a neutral near-white midpoint so zero cells still read as
empty:

| Score | Original | `_redpink` |
|---|---|---|
| +1 | `#488f31` green | `#de6e8c` pink |
| 0 | `#f1f1f1` | `#f1f1f1` |
| −1 | `#de3e00` orange | `#8f0707` dark red |

The palette's own `#d7c2c1` midpoint was **not** used at zero: it tints every
near-zero cell, so the grid stops reading as empty-vs-signal. It sits at the
−1/3 stop instead.

Because the source data is almost entirely positive, the dark-red end is
effectively unused — the figure reads as pink-on-white and `#8f0707` appears
mainly in the colorbar.

Regenerate with:

```
git show f0a910e:paper_figures/fig5_mlp_heatmaps_combined_asthma_ra.png > /tmp/src.png
python paper_figures/recolor_fig5_palette.py /tmp/src.png \
    paper_figures/fig5_mlp_heatmaps_combined_asthma_ra_redpink.png
```

Verified bit-for-bit reproducible, with the white-background and text masks
pixel-identical to the source.

**No generating script for fig5 exists in this repo** — searched across every
commit in history; only the PNG outputs were ever committed. That is why this
figure is produced by recolouring the render rather than by re-running a plot.

## Note on `fig9` provenance

The source bundle (`cot_resdstream_results_20260609_222032/`, ~14 GB) is **not
committed** — it was produced by `localize_bias/cot_vignette/cot_patching_resdStream.ipynb`
on a Lambda instance at `/home/ubuntu/patching_results1_*` and shared as a zip.
It is CoT + residual-stream, distinct from the MLP CoT bundles under
`raw_uploads/cot_patching_{qwen,olmo}/` — do not substitute those.

**Panel sources are not uniform.** Despite the source figure's "PROMPT_A" title,
the panels come from different prompt types, verified label-for-label:

| Panel | var1 figure | var2 figure |
|---|---|---|
| depression | PROMPT_A / var1, tokens 142-156 | PROMPT_A / var2, 139-153 |
| multiple sclerosis | **PROMPT_C** / var1, 133-147 | **PROMPT_C** / var2, 136-150 |
| rheumatoid arthritis | PROMPT_A / var1, 148-162 | PROMPT_A / var2, 145-159 |
| sarcoidosis | PROMPT_A / var1, 146-160 | PROMPT_A / var2, 143-157 |

Each panel shows a contiguous 15-token window plotted in **descending** position
order left-to-right, matching the source figure. The window is hard-coded per
panel in the generating script rather than derived, since no single rule
reproduces all four.

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
