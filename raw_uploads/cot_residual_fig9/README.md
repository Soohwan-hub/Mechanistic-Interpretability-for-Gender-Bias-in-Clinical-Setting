# cot_residual_fig9/

Extracted numbers behind **Figure 9** (CoT residual-stream rewrite scores,
`paper_figures/fig9_*.png`). Self-contained: the figures can be regenerated
from this folder alone.

## Why this exists

The source bundle `cot_resdstream_results_20260609_222032/` is ~14 GB, because
each `result_var*.pkl` carries a ~70 MB `layer_hidden_states` blob of raw
activations. The heatmaps use **only** `rewrite_matrix` and `token_labels`.
Those are extracted here at **548 KB total**, so the 14 GB bundle no longer
needs to be kept on disk.

Values were verified bit-for-bit against the originals (`np.allclose`,
atol=1e-9) for all 8 panels before the bundle was discarded.

## Contents — `data/`

Per (condition, prompt, variant), three files:

| Suffix | What |
|---|---|
| `_rewrite_matrix.csv` | Full 28 × n_tokens matrix; row = layer 0-27, col = token position |
| `_token_labels.json` | Token strings, index = column index in the matrix |
| `_fig9_var{1,2}_window.csv` | Just the 15 columns drawn in the figure, in plot order (descending position), with `layer` as the first column |

`index.json` maps each panel to its prompt, variant, and window positions.

## Panel sources

Despite the source figure's "PROMPT_A" title, multiple sclerosis comes from
PROMPT_C. Verified label-for-label; see `paper_figures/README.md`.

| Panel | var1 | var2 |
|---|---|---|
| depression | PROMPT_A / var1, tokens 142-156 | PROMPT_A / var2, 139-153 |
| multiple sclerosis | **PROMPT_C** / var1, 133-147 | **PROMPT_C** / var2, 136-150 |
| rheumatoid arthritis | PROMPT_A / var1, 148-162 | PROMPT_A / var2, 145-159 |
| sarcoidosis | PROMPT_A / var1, 146-160 | PROMPT_A / var2, 143-157 |

## Provenance

Produced by `localize_bias/cot_vignette/cot_patching_resdStream.ipynb` on a
Lambda instance (`/home/ubuntu/patching_results1_*`), shared as
`cot_resdstream_results_20260609_222032.zip`. Qwen 2.5-7B, CoT vignette
prompts, **residual-stream** patching, 28 layers.

Not to be confused with the MLP CoT bundles in
`raw_uploads/cot_patching_{qwen,olmo}/` — different patch site.

## Regenerate the figures

    python regenerate_fig9.py

Reads only this folder; writes to `paper_figures/`.
