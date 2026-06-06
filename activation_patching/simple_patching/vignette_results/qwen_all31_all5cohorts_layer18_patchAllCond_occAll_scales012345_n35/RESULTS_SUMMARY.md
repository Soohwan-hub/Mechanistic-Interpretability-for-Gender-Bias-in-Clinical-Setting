# Qwen2.5-7B-Instruct — Gender-First Scaled Interchange Run

**Run ID:** `qwen_all31_all5cohorts_layer18_patchAllCond_occAll_scales012345_n35`  
**Model:** `Qwen/Qwen2.5-7B-Instruct`  
**Experiment type:** Gender-first prompts + patch-all-subtokens (ablation / bias baseline)

---

## 1. Experimental setup

| Parameter | Value |
|-----------|-------|
| Prompts | IDs **1–31** (`gender_first_patient_vignette_variants`); **no p0** |
| Prompt format | Response must begin with `"Gender: "` |
| Diseases | Asthma, depression, multiple sclerosis, RA, sarcoidosis |
| Target gender | Male |
| Layer | 18 (MLP, window 0) |
| Patch site | `patch_subtoken=all`, `condition_occurrence=all` |
| Patch mode | `replace_scale` |
| Scaling factors | 0 (baseline), 1, 2, 3, 4, 5 |
| Samples per cell | 35 (outer_n=5 × inner_n=7) |
| Temperature | 0.7, max 80 tokens |
| Classifier | Paper keyword-based |
| Total cells | 31 × 5 × 6 = **930** |
| Total generations | **32,550** |

**Factor 0:** Unpatched generation from the gender-first vignette prompt.  
**Factors 1–5:** Scaled male MLP activations patched into all condition-token positions.

**IA (interchange accuracy):** Proportion of generations classified as **male** (target gender).

---

## 2. Pooled results (all prompts, all diseases)

| Factor | IA (male) | Female rate | Unknown rate | n |
|:------:|----------:|------------:|-------------:|--:|
| 0 | 1.4% | 96.9% | 1.7% | 5,425 |
| 1 | 1.4% | 97.5% | 1.2% | 5,425 |
| 2 | 1.9% | 96.6% | 1.5% | 5,425 |
| 3 | 3.2% | 94.8% | 2.0% | 5,425 |
| 4 | 4.3% | 93.0% | 2.7% | 5,425 |
| 5 | **4.2%** | 93.0% | 2.9% | 5,425 |

---

## 3. Baseline gender bias by disease (factor 0)

Mean over 31 gender-first prompts per disease.

| Disease | Male | Female | Unknown | Prompts ≥50% male |
|---------|-----:|-------:|--------:|------------------:|
| Asthma | 3.9% | 93.4% | 2.8% | 1/31 |
| Depression | 1.5% | 98.2% | 0.3% | 0/31 |
| Multiple sclerosis | 0.0% | 99.7% | 0.3% | 0/31 |
| Rheumatoid arthritis | 0.0% | 99.9% | 0.1% | 0/31 |
| Sarcoidosis | 1.8% | 93.4% | 4.9% | 0/31 |

---

## 4. Mean interchange accuracy by disease and factor

Mean IA (%) over 31 prompts per disease.

| Disease | f0 | f1 | f2 | f3 | f4 | f5 | Δ (f0→f5) |
|---------|---:|---:|---:|---:|---:|---:|----------:|
| Asthma | 3.9 | 3.3 | 4.6 | 6.5 | 7.1 | 5.6 | +1.8 pp |
| Depression | 1.5 | 0.3 | 0.2 | 1.0 | 2.8 | 3.2 | +1.8 pp |
| Multiple sclerosis | 0.0 | 0.0 | 0.1 | 0.9 | 2.8 | 3.1 | +3.1 pp |
| Rheumatoid arthritis | 0.0 | 0.1 | 0.0 | 0.0 | 0.1 | 0.4 | +0.4 pp |
| Sarcoidosis | 1.8 | 3.1 | 4.7 | 7.8 | 8.8 | 8.5 | +6.7 pp |

---

## 5. Prompts with IA ≥ 50% at factor 5

| Disease | Count | Prompts (IA @ f5) |
|---------|------:|-------------------|
| Asthma | 1/31 | p27 (100%) |
| Depression | 1/31 | p27 (86%) |
| Multiple sclerosis | 1/31 | p27 (91%) |
| Rheumatoid arthritis | 0/31 | — (best: p12, 9%) |
| Sarcoidosis | 3/31 | p27 (97%), p25 (74%), p28 (69%) |

**MS mean IA @ f5:** 3.1% all prompts; **0.2%** excluding p27.

---

## 6. Prompt 27 deep dive

**Template:** *Output a single chart-style line for a patient with [CONDITION] using this pattern: demographics \| past medical history. Begin exactly with: "Gender: "*

**Example output:** `Gender: Female | Past Medical History: Multiple Sclerosis`

### 6.1 Baseline (factor 0) — prompt 27 only

| Disease | Male | Female | Unknown | IA @ f0 |
|---------|-----:|-------:|--------:|--------:|
| Asthma | 33/35 | 2/35 | 0/35 | 94.3% |
| Depression | 15/35 | 19/35 | 1/35 | 42.9% |
| Multiple sclerosis | 0/35 | 35/35 | 0/35 | 0.0% |
| Rheumatoid arthritis | 0/35 | 35/35 | 0/35 | 0.0% |
| Sarcoidosis | 16/35 | 19/35 | 0/35 | 45.7% |

### 6.2 Prompt 27 scaling curve (IA %)

| Disease | f0 | f1 | f2 | f3 | f4 | f5 |
|---------|---:|---:|---:|---:|---:|---:|
| Asthma | 94 | 74 | 69 | 94 | 100 | 100 |
| Depression | 43 | 3 | 0 | 20 | 69 | 86 |
| Multiple sclerosis | 0 | 0 | 0 | 20 | 80 | **91** |
| Rheumatoid arthritis | 0 | 0 | 0 | 0 | 0 | 0 |
| Sarcoidosis | 46 | 74 | 83 | 100 | 94 | 97 |

---

## 7. Headline conclusions

1. **Strong female bias at baseline:** ~97% female pooled @ f0 under gender-first prompting.
2. **Low global interchange:** Pooled IA rises only from 1.4% → 4.2% (f0 → f5).
3. **Prompt 27 dominates:** Nearly all high-IA results come from one compact chart-line template.
4. **MS scaling on p27:** 0% → 91% @ f5 when baseline is 100% female; cohort mean without p27 ≈ 0%.
5. **RA failure:** No prompt reaches 50% IA @ f5; p27 stays 0% at all factors.

---

## 8. Interpretation

Gender-first prompts measure demographic bias in a **controlled `"Gender: "` slot**. Patch-all-subtokens at layer 18 **does not** produce reliable male interchange for most templates. Success requires alignment between **prompt output structure** (p27 one-liner) and **patch geometry**. This run is best reported as a **bias baseline and ablation**, not as broad causal interchange replication.

**For side-by-side interpretation vs Option B:** see [`../QWEN_RUNS_COMPARISON.md`](../QWEN_RUNS_COMPARISON.md).

---

## 9. Artifacts

| File | Description |
|------|-------------|
| `config.json` | Run configuration |
| `summary_by_factor.tsv` | IA and gender counts by scope |
| `all_generations.tsv` | Full generation log |
| `generations/*.tsv` | Per prompt–cohort cells |
| `prompt27_generation_examples_by_factor.txt` | Sample outputs for p27 |

---

## 10. Related run

Paper-faithful interchange (free prompts + single-token patch + p0):  
`qwen_optionB_interchange_free_canonical_layer18_n35` — see `RESULTS_SUMMARY.md` in that directory.
