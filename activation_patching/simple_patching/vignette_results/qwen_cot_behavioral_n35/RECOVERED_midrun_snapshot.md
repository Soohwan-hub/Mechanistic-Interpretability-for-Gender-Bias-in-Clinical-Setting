# Recovered mid-run snapshot (NOT full n=35)

**Status:** Full Lambda disk was terminated before results were copied.  
The tables below are reconstructed from an SSH terminal capture on the GPU host while the run was still in progress. They are **real numbers from that moment**, but **not** the final 3500-gen run.

## Snapshot metadata

| Field | Value |
|-------|--------|
| Model | Qwen/Qwen2.5-7B-Instruct |
| Run id | `qwen_cot_behavioral_n35` |
| Design | 5 conditions × 20 CoT prompts × 35 gens = 3500 |
| Progress at snapshot | **75/100 cells done** (`n_rows=2275`) |
| Later log progress | seen at **81/100** (rheumatoid_arthritis p16 / sarcoidosis p1) |
| Source | Mac SSH session capture (`terminals/31.txt`) reading `all_generations.tsv` on Lambda |
| Full TSVs / generations | **Lost** with instance terminate |

## Gender rates at 2275 gens (75 cells)

| Gender | Count | Rate |
|--------|------:|-----:|
| Female | 2064 | **90.7%** |
| Male | 210 | **9.2%** |
| Unknown | 1 | ~0.0% |
| **Total** | **2275** | 100% |

## Format checks (same snapshot)

| Check | Result |
|-------|--------|
| Presentation starts with `Gender:` | **2275/2275 (100%)** |
| Assistant text contains `<thinking>` | **0/2275 (0%)** |
| Presentation word count | mean ≈ 147, p50 ≈ 137, p90 ≈ 221 |

## Cells known complete at resume (1–75)

When the job was resumed with `--resume`, these cells already existed on disk:

- asthma prompts 1–20  
- depression prompts 1–20  
- multiple_sclerosis prompts 1–20  
- rheumatoid_arthritis prompts 1–15  

→ **75 cells × 35 = 2275** rows (matches `n_rows`).

Remaining at that moment: RA prompts 16–20 + all sarcoidosis prompts 1–20 = **25 cells**.

## What friends can / cannot conclude

**Can say (with caveat):** under CoT free generation on Qwen, mid-run output was strongly Female-skewed (~91% Female / ~9% Male) across the first 75 cells; model did not emit visible `<thinking>` tags.

**Cannot say:** final per-condition / per-prompt rates for the full 100 cells, or share raw vignettes — those files are gone.

## How to get the real full results

Re-launch a GPU instance, re-run `./run_cot_behavioral_n35.sh`, then **rsync before terminate**:

```bash
rsync -avz --progress \
  ubuntu@NEW_IP:~/Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting/activation_patching/simple_patching/vignette_results/qwen_cot_behavioral_n35/ \
  activation_patching/simple_patching/vignette_results/qwen_cot_behavioral_n35/
```

Prefer writing outputs under an attached Lambda filesystem (`/lambda/nfs/...`) so terminate does not wipe them.
