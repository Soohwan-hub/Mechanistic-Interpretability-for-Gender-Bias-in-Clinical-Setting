# Cot patching `results/` — what each bundle is

Short map of artifact folders here. **Large tensors / pickles are usually git‑ignored**; what you commit is mostly **figures** (`**/plots/`). See repo root `.gitignore`.

| Folder | What it is |
|--------|------------|
| **`gh200_full_run`** | Full **`sae_localization.py`** run (GH200 profile): causal sweep over sparse SAE features on **residual stream**, Qwen2.5‑7B‑Instruct, **early layer set** (e.g. 3–27 subset). Contains `artifacts/`, **`progress.json`**, parquet/CSVs where present. |
| **`gh200_full_artifacts`** | Same run as **`gh200_full`**, flattened (artifact files top‑level, no nested `artifacts/` wrapper). Convenience copy for sharing/analysis. |
| **`gh200_full_artifacts_x86`** | Duplicate of **`gh200_full`** outputs from another machine/sync; identical metrics; historically had the only exported **heatmap PNG** among the three full bundles. |
| **`gh200_signaware_superset_artifacts`** | **`sae_localization.py`** with **sign‑aware gating** (positive vs negative‑activation thresholds) **+ expanded SAE layer superset** (Andyrdt + Geaming coverage). Larger sweep than **`gh200_full`**. |
| **`rep_top5_plots_artifacts`** | **`sae_replication.py`**: replicate **early top shortlisted** `(layer, feature)` pairs (e.g. from localization shortlists) across conditions / prompt variants / temps / seeds — **figures** + `replication_*.csv` / parquet. |
| **`rep_layer18_signaware_csv_artifacts`** | **`sae_replication.py`** focused on **layer 18** shortlist coordinates with **`sign_aware`** gating; multi‑temperature/seed replication and plots. |
| **`qwen_bronch_promptA`** | **`attention_head_cot_patching`** (or sibling) — **attention head**/patching‑style diagnostics for **bronchitis** under **prompt style A**, Qwen traces; trackers + **`plots`** (collapsed / per‑layer head viz). |
| **`Olmo_head_patching`** | Same **attention head patching** tooling, **`Olmo`** model; nests multiple **`head_onecond_*`** condition runs (e.g. RA/bronchitis/asthma × prompt **A**/ **C**, layer‑block ranges in names). Heavy `.pkl` usually ignored by git.

**Contrast:** **`gh200_*`** = **SAE latent** interventions (sparse features); **`qwen_bronch_*`** / **`Olmo_head_*`** = **attention‑head**/tensor‑level patching (different mechanism).
