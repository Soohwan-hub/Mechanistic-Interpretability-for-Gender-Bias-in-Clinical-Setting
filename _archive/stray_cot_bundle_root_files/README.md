# Stray files relocated from the repo root

These were left at the repo root while investigating Figure 9 and are **not**
part of the project layout. Relocated rather than deleted, per the
never-delete rule.

| File | What it is | Why it moved |
|---|---|---|
| `manifest.json` | Byte-identical copy of the manifest from `cot_resdstream_results_20260609_222032/` (the CoT residual-stream bundle) | Duplicate of a file that lives with its bundle; at the root it read as project metadata |
| `patching_results1/aggregated_scores.pkl` | **Empty (0 bytes)** stub | Never written to; an empty artifact at the root implies committed CoT results that do not exist |

The real CoT residual-stream bundle (~14 GB) is not committed. See the
`fig9` provenance note in `paper_figures/README.md`.
