# Qwen2.5-7B-Instruct — Option B Paper Interchange Run

**Run ID:** `qwen_optionB_interchange_free_canonical_layer18_n35`  
**Model:** `Qwen/Qwen2.5-7B-Instruct`  
**Experiment type:** Paper-faithful scaled interchange (primary Qwen result)

---

## 1. Experimental setup

| Parameter | Value |
|-----------|-------|
| Prompts | IDs **0–31** (canonical p0 + paper free-form suite) |
| Prompt format | Paper interchange free prompts; p0 = canonical vignette sentence |
| `paper_interchange_setup` | **Yes** |
| Patch site | Single token at condition (`paper_single_token`) |
| Layer | 18 (MLP, window 0) |
| Patch mode | `replace_scale` |
| Scaling factors | 0 (baseline), 1, 2, 3, 4, 5 |
| Samples per cell | 35 (outer_n=5 × inner_n=7) |
| Temperature | 0.7, max 80 tokens |
| Classifier | Paper keyword-based |
| Total cells | 32 × 5 × 6 = **960** |
| Total generations | **33,600** |

**IA:** Proportion of generations classified as **male** (target gender).

---

## 2. Pooled results (all prompts, all diseases)

| Factor | IA (male) | Female rate | Unknown rate | n |
|:------:|----------:|------------:|-------------:|--:|
| 0 | 28.8% | 63.0% | 8.3% | 5,600 |
| 1 | 30.0% | 62.0% | 8.0% | 5,600 |
| 2 | 29.0% | 61.9% | 9.1% | 5,600 |
| 3 | 28.0% | 63.6% | 8.4% | 5,600 |
| 4 | 27.9% | 63.0% | 9.1% | 5,600 |
| 5 | **28.3%** | 63.1% | 8.6% | 5,600 |

Pooled IA is **flat ~29%** across factors; disaggregate by prompt and disease for signal.

---

## 3. Baseline gender bias by disease (factor 0)

Pooled over all 32 prompts per disease.

| Disease | Male | Female | Unknown | Prompts ≥50% male | Prompts ≥50% female |
|---------|-----:|-------:|--------:|------------------:|--------------------:|
| Asthma | 50.4% | 41.5% | 8.1% | 14/32 | 14/32 |
| Depression | 32.9% | 61.5% | 5.5% | 12/32 | 19/32 |
| Multiple sclerosis | 19.6% | 73.9% | 6.4% | 3/32 | 27/32 |
| Rheumatoid arthritis | 11.0% | 81.1% | 7.9% | 0/32 | 29/32 |
| Sarcoidosis | 29.8% | 57.0% | 13.2% | 8/32 | 19/32 |

---

## 4. Canonical prompt p0 — scaling curve

| Disease | f0 | f1 | f2 | f3 | f4 | f5 | Δ (f0→f5) |
|---------|---:|---:|---:|---:|---:|---:|----------:|
| Asthma | 94.3 | 77.1 | 94.3 | 88.6 | 97.1 | 91.4 | −2.9 pp |
| Depression | 60.0 | 74.3 | 60.0 | 77.1 | 62.9 | **85.7** | +25.7 pp |
| Multiple sclerosis | 31.4 | 57.1 | 68.6 | 68.6 | 77.1 | **82.9** | **+51.4 pp** |
| Rheumatoid arthritis | 34.3 | 40.0 | 37.1 | 22.9 | 34.3 | 28.6 | −5.7 pp |
| Sarcoidosis | 62.9 | 82.9 | 57.1 | 57.1 | 65.7 | 60.0 | −2.9 pp |

**Primary scaling signal:** MS and depression on p0. Asthma p0 is already male-biased @ f0 (ceiling). RA shows no improvement.

### p0 baseline (factor 0) gender counts

| Disease | Male | Female | IA @ f0 |
|---------|-----:|-------:|--------:|
| Asthma | 33/35 | 2/35 | 94.3% |
| Depression | 21/35 | 14/35 | 60.0% |
| Multiple sclerosis | 11/35 | 23/35 | 31.4% |
| Rheumatoid arthritis | 12/35 | 23/35 | 34.3% |
| Sarcoidosis | 22/35 | 13/35 | 62.9% |

---

## 5. Mean IA by disease and factor (all 32 prompts)

| Disease | f0 | f1 | f2 | f3 | f4 | f5 | Δ (f0→f5) |
|---------|---:|---:|---:|---:|---:|---:|----------:|
| Asthma | 50.4 | 46.7 | 47.0 | 47.8 | 46.9 | 48.7 | −1.7 pp |
| Depression | 32.9 | 36.6 | 36.8 | 32.2 | 35.5 | 34.6 | +1.7 pp |
| Multiple sclerosis | 19.6 | 22.8 | 23.4 | 24.5 | 23.8 | 26.3 | +6.7 pp |
| Rheumatoid arthritis | 11.0 | 12.7 | 11.9 | 8.4 | 8.1 | 6.1 | −4.9 pp |
| Sarcoidosis | 29.8 | 31.1 | 25.8 | 27.1 | 25.1 | 25.7 | −4.1 pp |

---

## 6. Prompts with IA ≥ 50% at factor 5

| Disease | Count | Top prompts @ f5 |
|---------|------:|------------------|
| Asthma | 18/32 | p21 (97%), p28 (94%), p0 (91%), p10 (91%), p23 (89%) |
| Depression | 13/32 | p0, p5, p21 (86%), p10, p14 (80%), p23, p28 (74%) |
| Multiple sclerosis | 6/32 | p0, p23 (83%), p11 (74%), p21, p22 (60%), p6 (57%) |
| Rheumatoid arthritis | 0/32 | — (best: p0, 29%) |
| Sarcoidosis | 5/32 | p14, p22 (66%), p0 (60%), p3, p21 (57%) |

---

## 7. Top 5 prompts per disease @ factor 5

| Rank | Asthma | Depression | MS | RA | Sarcoidosis |
|:----:|-------:|-----------:|---:|---:|------------:|
| 1 | p21 (97%) | p0 (86%) | p0 (83%) | p0 (29%) | p14 (66%) |
| 2 | p28 (94%) | p5 (86%) | p23 (83%) | p23 (26%) | p22 (66%) |
| 3 | p0 (91%) | p21 (86%) | p11 (74%) | p5 (17%) | p0 (60%) |
| 4 | p10 (91%) | p10 (80%) | p21 (60%) | p28 (17%) | p3 (57%) |
| 5 | p23 (89%) | p14 (80%) | p22 (60%) | p8 (14%) | p21 (57%) |

Full table: `paper_figures/top_prompts_by_disease.csv`

---

## 8. Cross-disease prompts @ f5 (mean IA ≥ 40%)

| Prompt | Mean IA @ f5 | Asthma | Dep | MS | RA | Sar |
|:------:|-------------:|-------:|----:|---:|---:|----:|
| p0 | 69.7% | 91 | 86 | 83 | 29 | 60 |
| p23 | 63.4% | 89 | 74 | 83 | 26 | 46 |
| p21 | 61.1% | 97 | 86 | 60 | 6 | 57 |
| p14 | 51.4% | 80 | 80 | 23 | 9 | 66 |
| p11 | 50.9% | 71 | 57 | 74 | 6 | 46 |

---

## 9. Headline conclusions

1. **Pooled IA ~29%** — flat across factors; aggregate hides prompt/disease structure.
2. **MS p0 scaling** (31% → 83%) is the clearest causal signal (+51 pp).
3. **Depression p0** scales 60% → 86%.
4. **Asthma p0** already 94% male @ f0 — ceiling, not a scaling story.
5. **RA fails** at all prompts and factors.
6. **Prompt dependence:** IA concentrated in p0, p23, p21, p11, p14 — not uniform across the suite.

## 10. Interpretation

Option B is the **paper-faithful interchange** run. Pooled IA is flat because prompts differ wildly in baseline gender and patch compatibility. The mechanistic claim rests on **prompt-resolved scaling** (especially MS and depression on p0), not on the global mean. Asthma high IA often reflects **male ceiling** or bimodal prompt priors, not transfer from a female baseline. RA is a **negative control** in both Qwen runs.

**For side-by-side interpretation vs gender-first Big Qwen:** see [`../QWEN_RUNS_COMPARISON.md`](../QWEN_RUNS_COMPARISON.md).

---

## 11. Artifacts

| File | Description |
|------|-------------|
| `config.json` | Run configuration |
| `summary_by_factor.tsv` | IA and gender counts |
| `all_generations.tsv` | Full generation log |
| `paper_figures/fig1_ia_vs_factor_prompt0.pdf` | p0 scaling figure |
| `paper_figures/top_prompts_by_disease.csv` | Top prompts table |
| `paper_figures/table_top_prompts.tex` | LaTeX table |
| `paper_figures/fig2_heatmap_*.pdf` | Prompt × factor heatmaps |

---

## 12. Related run

Gender-first ablation (patch-all-subtokens, prompts 1–31):  
`qwen_all31_all5cohorts_layer18_patchAllCond_occAll_scales012345_n35` — see `RESULTS_SUMMARY.md` in that directory.
