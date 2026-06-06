# Qwen2.5-7B-Instruct — Run Comparison

Side-by-side comparison of the two main scaled-interchange runs on the same model and layer.

---

## Setup comparison

| | Gender-first (Big Qwen) | Option B (Paper interchange) |
|--|-------------------------|------------------------------|
| **Run ID** | `qwen_all31_..._n35` | `qwen_optionB_..._n35` |
| **Prompts** | 1–31 | 0–31 (includes canonical p0) |
| **Prompt style** | Gender-first (`"Gender: "` lead-in) | Paper free + canonical p0 |
| **Patch site** | All subtokens, all occurrences | Single token (paper interchange) |
| **Paper interchange setup** | No | Yes |
| **Generations** | 32,550 | 33,600 |

---

## Pooled IA and female rate

| Factor | Big Qwen IA | Big Qwen female | Option B IA | Option B female |
|:------:|------------:|----------------:|------------:|----------------:|
| 0 | 1.4% | 96.9% | 28.8% | 63.0% |
| 5 | 4.2% | 93.0% | 28.3% | 63.1% |
| Δ (f0→f5) | +2.8 pp | −3.9 pp | −0.5 pp | +0.1 pp |

---

## Mean IA by disease @ f0 and f5

| Disease | Big f0 | Big f5 | OptB f0 | OptB f5 |
|---------|-------:|-------:|--------:|--------:|
| Asthma | 3.9% | 5.6% | 50.4% | 48.7% |
| Depression | 1.5% | 3.2% | 32.9% | 34.6% |
| Multiple sclerosis | 0.0% | 3.1% | 19.6% | 26.3% |
| Rheumatoid arthritis | 0.0% | 0.4% | 11.0% | 6.1% |
| Sarcoidosis | 1.8% | 8.5% | 29.8% | 25.7% |

---

## Prompts ≥50% IA @ f5

| Disease | Big Qwen | Option B |
|---------|:--------:|:--------:|
| Asthma | 1/31 | 18/32 |
| Depression | 1/31 | 13/32 |
| Multiple sclerosis | 1/31 | 6/32 |
| Rheumatoid arthritis | 0/31 | 0/32 |
| Sarcoidosis | 3/31 | 5/32 |

---

## MS — best signal comparison

| Metric | Big Qwen | Option B |
|--------|----------|----------|
| Mean IA @ f5 | 3.1% | 26.3% |
| Best prompt @ f5 | p27 (91%) | p0, p23 (83%) |
| p0 / canonical scaling | N/A (no p0) | 31% → 83% |
| MS mean @ f5 excl. top outlier | 0.2% (excl. p27) | — |

---

## How to use in the paper

| Question | Report from |
|----------|-------------|
| Female demographic bias (gender slot) | **Big Qwen** @ f0 |
| Paper-faithful interchange | **Option B** |
| Prompt-format ablation | **This comparison** |
| RA negative result | **Both** (neither works) |

---

## Interpretation and explanation

### What each run is actually measuring

Both runs use the same model (Qwen2.5-7B-Instruct), the same layer (18), the same target (male), and the same classifier. They differ in **prompt framing** and **patch geometry**. Those two choices change both (a) what gender the model produces without patching, and (b) whether patching can override that output.

| Dimension | Gender-first (Big Qwen) | Option B (Paper interchange) |
|-----------|---------------------------|------------------------------|
| **Generation task** | Fill an explicit `"Gender: "` field first, then demographics | Write a normal clinical vignette; gender may appear anywhere |
| **What f0 measures** | Default gender in a **controlled slot** | Default gender in **open clinical prose** |
| **Patching** | Replace **all** condition subtokens at **all** occurrences | Replace **one** token at the paper interchange site |
| **Closest paper analogue** | Bias-check / ablation (gender-slot probing) | Authors’ `get_interchange_accuracy.py` setup |

Neither run alone answers every question. Together they separate **demographic bias** from **causal interchange under patching**.

---

### Baseline bias: why female rates differ so much (f0)

**Big Qwen @ f0: ~97% female (pooled)**

Gender-first templates force the model to commit to gender in the first tokens of its reply. Under this constraint, Qwen overwhelmingly outputs **Female** for all five diseases (93–99% female per disease when averaged over 31 prompts). This matches the expectation from prior bias work: when asked to populate an explicit gender field for these conditions, the model shows strong **female demographic priors**.

**Option B @ f0: ~63% female (pooled)**

Free-form and canonical prompts do not force an immediate gender slot. The model can weave demographics into narrative structure, omit gender keywords (8.3% Unknown), or assign male or female depending on template wording. Disease-level patterns are mixed:

- **MS and RA** remain strongly female-leaning on free prompts (74–81% female).
- **Asthma is bimodal:** 14/32 prompts male-biased, 14/32 female-biased @ f0.
- **Canonical p0 breaks the pattern:** asthma p0 is **94% male** @ f0; MS p0 is **66% female**.

**Interpretation:** Strong female bias is **real but not universal**. It appears most clearly under gender-first slot prompting. Open vignette wording exposes **prompt-dependent** priors that cohort averages hide. Asthma p0 is not a good stand-in for “asthma bias” in Option B—it is one template with a strong male prior.

---

### Interchange: why pooled IA is 4% vs 29%

**Big Qwen: IA 1.4% → 4.2% (f0 → f5)**

Patch-all-subtokens at layer 18 **fails to flip** most gender-first generations toward male. Pooled IA stays near zero because:

1. The model is already ~97% female @ f0, so success requires overcoming a strong prior.
2. Patching **every** condition subtoken may **disrupt** generation for most template formats rather than cleanly transfer gender.
3. Almost all high-IA cells come from **prompt 27** alone—a minimal one-line format (`Gender: X | PMH: …`) where gender sits in an isolated, patchable position.

Excluding p27, MS mean IA @ f5 is **0.2%**. The run does **not** demonstrate broad causal interchange; it demonstrates one **prompt–patch interaction**.

**Option B: IA 28.8% → 28.3% (f0 → f5)**

Pooled IA is ~7× higher than Big Qwen but **flat across factors** when averaged over all prompts. That flat curve is misleading: it mixes templates where patching already works @ f0 (asthma p0 at 94% male) with templates where scaling helps (MS p0: 31% → 83%). The signal lives in **disaggregated** prompt × disease cells, not in the global mean.

**Interpretation:** Higher Option B IA reflects **better alignment** between paper single-token patching and free-form generation—not simply “more bias.” Big Qwen’s lower IA reflects **patch-all failure** on most templates, not necessarily weaker internal gender representations.

---

### Multiple sclerosis: two different “success stories”

| | Big Qwen | Option B |
|--|----------|----------|
| Baseline (best case) | p27: **100% female** @ f0 | p0: **66% female** @ f0 (11/35 male) |
| Best IA @ f5 | p27: **91%** | p0, p23: **83%** |
| Scaling shape | Sharp jump f3→f5 on p27 only | Monotonic p0: 31%→57%→69%→69%→77%→83% |
| Breadth | 1/31 prompts ≥50% | 6/32 prompts ≥50% |

**Big Qwen MS story:** One gender-first chart-line template (p27), fully female @ f0, flips to ~91% male at f5 when **all** condition subtokens are scaled. Cohort mean MS IA without p27 ≈ 0%.

**Option B MS story:** Canonical paper prompt (p0) and free templates (p23, p11) show scaling from a **partly female** baseline to ~83% male. This is the cleaner **dose–response** curve for a paper Results figure.

**Interpretation:** Both runs show MS can be causally manipulated at layer 18, but **Option B generalizes across prompts**; Big Qwen’s MS result is **fragile and template-specific**.

---

### Rheumatoid arthritis: shared failure mode

| | Big Qwen @ f5 | Option B @ f5 |
|--|---------------|---------------|
| Mean IA | 0.4% | 6.1% |
| Prompts ≥50% | 0/31 | 0/32 |
| p27 @ f5 | **0%** (despite 100% female @ f0) | p27 not in suite |

RA is female-biased at baseline in **both** setups (Big Qwen ~100% female; Option B ~81% female pooled @ f0), yet **neither** patch protocol achieves reliable male interchange. Even Big Qwen’s p27—which reaches 91% on MS—stays at **0% on RA** at all factors.

**Interpretation:** RA is a **robust negative result**. Gender for this condition may be less localized at layer 18 in Qwen, may require different patch sites, or may not couple to generation in a way the keyword classifier captures. RA should be reported as a **boundary case**, not as noise.

---

### Asthma: ceiling vs bimodality

**Big Qwen:** Cohort mean IA low (3.9%→5.6%), but **p27 @ f0 is already 94% male**—patching adds little because the “target” is already met.

**Option B:** Cohort mean IA high (~50%) because many templates are male-leaning or easily flipped; **p0 @ f0 is 94% male** with no real scaling gain (94%→91%).

**Interpretation:** Asthma is a **ceiling-effect disease** on high-performing templates, not a scaling showcase. Do not interpret high Option B asthma IA as strong evidence for causal patching from a female baseline—often the model was already male-typed before scaling.

---

### Prompt 27: why it matters in Big Qwen only

Prompt 27 asks for a **single chart line** beginning with `"Gender: "`. Outputs look like:

`Gender: Female | Past Medical History: Multiple Sclerosis`

This format isolates gender in token 1 of the assistant reply. Patch-all-subtokens on the condition name can then swap internal representations at high scale factors—**but only for some diseases** (MS, sarcoidosis, depression; not RA; asthma already male @ f0).

Option B uses different free prompts plus canonical p0; **p27 is not the main carrier** of signal there. Option B success concentrates in **p0, p23, p21, p11** under single-token patching.

**Interpretation:** p27 is an **artifact of prompt format × patch-all**, not evidence that gender-first prompting generally enables interchange.

---

### Side-by-side conclusions

| Claim | Supported by Big Qwen? | Supported by Option B? |
|-------|:----------------------:|:----------------------:|
| Qwen assigns female demographics in clinical vignettes | **Yes** (~97% @ f0) | **Partially** (63% @ f0; disease/prompt dependent) |
| Layer-18 patching can cause male generations | **Only on p27** (and similar compact templates) | **Yes, on selected prompts** (esp. MS/depression p0) |
| Scaling factor increases IA monotonically (pooled) | Weakly (+3 pp pooled) | No (flat pooled; yes on p0 MS/depression) |
| Interchange is disease-universal | **No** (RA fails) | **No** (RA fails) |
| Interchange is prompt-universal | **No** (1–3 prompts) | **No** (6–18 prompts per disease) |
| Paper-faithful replication target | No | **Yes** |

---

### Recommended narrative for your docs / paper

**Paragraph 1 — Bias (Big Qwen):**  
Under gender-first prompting, Qwen2.5-7B-Instruct produced overwhelmingly female-typed clinical vignettes (~97% at baseline), consistent with female demographic bias in an explicit gender field across asthma, depression, MS, RA, and sarcoidosis.

**Paragraph 2 — Interchange failure (Big Qwen):**  
Scaled patch-all-subtoken interchange at layer 18 yielded pooled male IA of only ~4% at factor 5. Successful flips were confined to prompt 27, a compact chart-line template; multiple sclerosis cohort-level IA without this prompt was ~0.2%. Rheumatoid arthritis showed no interchange at any factor.

**Paragraph 3 — Interchange partial success (Option B):**  
Under the paper-faithful protocol (free prompts, canonical p0, single-token patch), pooled IA was ~29% and flat across factors, but prompt-resolved analysis revealed strong scaling on the canonical prompt for multiple sclerosis (31%→83%) and depression (60%→86%). Six MS templates exceeded 50% IA at factor 5. Rheumatoid arthritis again failed.

**Paragraph 4 — Joint interpretation:**  
The two runs show that **observed gender bias and successful causal gender transfer depend jointly on prompt framing and patching geometry**, not on disease identity alone. Gender-first prompts best measure bias in a controlled slot; the paper free-prompt protocol best measures interchange. Neither setup supports a claim that layer 18 universally encodes interchangeable gender for all clinical entities in Qwen—RA remains a negative control in both.

---

### Limitations (both runs)

- **n = 35** per cell: wide confidence intervals; small count changes move IA by several percentage points.
- **Keyword classifier:** Unknown generations count as failure (Option B ~8.6%, Big Qwen ~3%).
- **Single layer, single model:** Results may not transfer to OLMo or other layers without further experiments.
- **Not independent replicates:** Different protocols; do not average or merge IA across runs.

---

## Full summaries

- [Gender-first run](qwen_all31_all5cohorts_layer18_patchAllCond_occAll_scales012345_n35/RESULTS_SUMMARY.md)
- [Option B run](qwen_optionB_interchange_free_canonical_layer18_n35/RESULTS_SUMMARY.md)
