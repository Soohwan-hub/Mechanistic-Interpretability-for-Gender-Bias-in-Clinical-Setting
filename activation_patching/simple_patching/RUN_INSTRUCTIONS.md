# Running `run_qwen_logdelta.ipynb` on Lambda Cloud

Re-runs the Qwen 5-cohort × 31-prompt simple-prompt MLP patching sweep with `--score-keys all`, so the output contains `logprob_delta_scores` (a.k.a. `log_delta`) alongside `rewrite_scores` in a single aggregate file. The original [`female5_patch_male/`](female5_patch_male/) bundle is left untouched — output goes to a fresh sibling dir [`female5_patch_male_logdelta/`](female5_patch_male_logdelta/).

---

## What you'll get

```
activation_patching/simple_patching/female5_patch_male_logdelta/
├── progress.json                    # 155 (cohort, prompt_id) entries when complete
├── run.log                          # tee'd stdout/stderr from the script
├── artifacts/                       # per-unit pickles, one per (cohort, prompt_id)
│   ├── asthma_prompt1.pkl
│   ├── asthma_prompt2.pkl
│   └── ...  (155 files)
├── aggregate_per_layer.csv          # 28 layers × 16 columns (4 stats × 4 score families × num_units)
└── aggregate_per_layer.json         # same content as JSON
```

Each per-unit pickle contains all three score matrices: `rewrite_scores`, `logprob_scores`, `logprob_delta_scores`, each shaped `(num_layers, num_tokens)`. The aggregate file has `{rewrite_scores,logprob_scores,logprob_delta_scores}_{mean,median,trimmed_mean,topk_mean,num_units}` per layer.

The downstream `condition_token_analysis/` directory you saw in `female5_patch_male/` is **not** produced by this script — that comes from a separate post-processing pass (not committed to main; lives on Soohwan's machine).

---

## What the notebook does (cells 1–5)

1. **Cell 1** — prints cwd, git branch, the simple-patching directory contents, and the explicitly resolved cohort + prompt-id lists. Asserts the expected unit count is exactly 155.
2. **Cell 2** — imports `torch`, `transformers`, `nnsight`, `numpy` and prints versions + CUDA device + free VRAM. **Also asserts `HF_TOKEN` is set** (Qwen weights are gated) and raises with a clear message if not.
3. **Cell 3** — runs the script with `--dry-run` and the full flag set. Refuses to proceed if the dry-run mentions `20` (the trap value = 5 cohorts × 4 default prompt-ids). This is the 30-second guard against a misconfigured run.
4. **Cell 4** — runs the full sweep with `--score-keys all`, streams stdout to the notebook and tees to `run.log`. Asserts exit code 0.
5. **Cell 5** — loads the produced `aggregate_per_layer.json`, asserts `rewrite_scores_mean`, `logprob_scores_mean`, `logprob_delta_scores_mean` are all present, prints layer-18 side-by-side values.

If any cell fails, the notebook stops there. Push the cells in order; no manual intervention needed.

---

## Lambda setup (A10 recommended, not A100)

The original Qwen run was done on Lambda; an **A10 (24 GB)** is the right tier for this workload. A100 is overkill; A10 fits Qwen 2.5-7B-Instruct in bf16 comfortably and matches the GPU class used for the existing bundle.

### Launch

1. Create a Lambda instance with **1× A10**, Ubuntu 22.04, CUDA 12.x.
2. SSH in.
3. Clone or upload this repo:
   ```bash
   git clone https://github.com/Soohwan-hub/Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting.git
   cd Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting
   git checkout om
   ```
4. Launch Jupyter:
   ```bash
   jupyter lab --ip=0.0.0.0 --no-browser
   ```
   Tunnel the printed URL through `ssh -L 8888:localhost:8888 ubuntu@<lambda-ip>` or use Lambda's built-in port forwarding.
5. Open `activation_patching/simple_patching/run_qwen_logdelta.ipynb` and run the cells top-to-bottom.

### Authentication for Qwen weights

`Qwen/Qwen2.5-7B-Instruct` is gated. **Cell 2 will raise if `HF_TOKEN` is not set** — better to fail there in 2 seconds than during model download 5 minutes into the real run.

Preferred: export the token in the shell before launching Jupyter:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
jupyter lab --ip=0.0.0.0 --no-browser
```

Alternative: paste the token into Cell 2 by uncommenting the `os.environ['HF_TOKEN'] = ...` line before running it. Do not commit the token. Cell 2 also accepts `HUGGING_FACE_HUB_TOKEN` as a fallback variable name.

---

## Expected runtime

The original `female5_patch_male/` bundle ran 155 units (5 cohorts × 31 prompts) with all 28 Qwen layers and full token sweeps. The added log_delta cost per token is a single `log_softmax` lookup and a subtraction — within ~10 % of rewrite-only wall-clock. **Plan for the same total runtime as the original sweep**; if Soohwan's run took roughly N hours, this one will take ~1.05–1.10 × N.

If you don't have a recorded baseline, a useful smoke check: run a single cohort first (e.g. `--cohorts depression`) to estimate ~31× the wall-clock of one prompt, then scale to 155 units. The script logs per-unit progress so you'll see this quickly.

---

## This is Qwen-only

`MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"` is hardcoded at [`simple_patching_without_BHCs.py:35`](simple_patching_without_BHCs.py). There is **no `--model-name` CLI flag**. This notebook does **not** produce OLMo log_delta. Producing the OLMo equivalent requires either:

- editing line 35 to `MODEL_NAME = "allenai/OLMo-...-Instruct"` (whichever string matches the original OLMo bundle — confirm with Soohwan before running) and re-launching, or
- copying the script to a `simple_patching_without_BHCs_olmo.py` sibling with the OLMo model name and running that separately.

Either way, the OLMo run will need its own fresh `--run-id` (e.g. `olmo31_logdelta`) and its own pass.

---

## Kernel disconnect warning — strongly consider tmux / nohup

Cell 3 runs the patching subprocess inline and streams output to the notebook's stdout. **A Jupyter tab disconnect or browser close will kill the kernel and may interrupt the subprocess.** The script is resume-safe via `progress.json` — if it dies mid-sweep, re-running Cell 3 with `--resume` (already set) picks up at the next unfinished `(cohort, prompt_id)` unit. But you'll lose any partial work on the in-flight unit.

### Safer: run from a terminal under tmux

If you can reach a shell on the Lambda box, prefer this:

```bash
tmux new -s logdelta
cd /home/ubuntu/Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting
export HF_TOKEN=hf_xxx
python -u activation_patching/simple_patching/simple_patching_without_BHCs.py \
  --score-keys all \
  --cohorts asthma,depression,multiple_sclerosis,rheumatoid_arthritis,sarcoidosis \
  --prompt-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  --run-id female5_patch_male_logdelta \
  --output-dir activation_patching/simple_patching \
  --resume \
  2>&1 | tee activation_patching/simple_patching/female5_patch_male_logdelta/run.log
```

Detach with `Ctrl-b d`; reattach later with `tmux attach -t logdelta`. The run survives SSH disconnects.

Alternative — `nohup`:

```bash
nohup python -u activation_patching/simple_patching/simple_patching_without_BHCs.py \
  --score-keys all --cohorts ... --prompt-ids ... \
  --run-id female5_patch_male_logdelta \
  --output-dir activation_patching/simple_patching \
  > activation_patching/simple_patching/female5_patch_male_logdelta/run.log 2>&1 &
```

Use the notebook's Cell 4 from the local machine afterwards to verify the aggregate output. Cell 4 only reads the JSON — no GPU needed.

### If you must run from Jupyter

Keep the browser tab visible and the laptop awake. Lambda Cloud's hosted JupyterLab can survive brief disconnects, but a closed tab will reliably take down the subprocess.

---

## Resume rules (the trap to know)

`--resume` reads `progress.json` and skips `(cohort, prompt_id)` units already in the `completed` list. It does **not** introspect the score keys of existing pickles. Consequences:

- **First run of `--run-id female5_patch_male_logdelta`** with `--resume`: nothing to resume, nothing to skip, all 155 units run with all 3 score keys. ✓
- **Subsequent re-run with same run-id** after a partial completion: skips finished units, picks up where it left off — partial pickles already have all 3 score keys, so the final aggregate is correct. ✓
- **Reusing the existing `--run-id female5_patch_male`** (the rewrite-only bundle) with `--score-keys all --resume`: the 155 pickles are already in `completed`, all units are skipped, the aggregator reads the rewrite-only pickles and writes an aggregate with **only** `rewrite_scores_*` columns. **You will not get log_delta.** ✗

This is why the notebook uses a fresh `--run-id`. Do not change it.

---

## Verifying the output before celebrating

Cell 4 in the notebook checks:
1. `aggregate_per_layer.json` exists at the expected path.
2. Each `per_layer` row has all three score-key families (`rewrite_scores_mean`, `logprob_scores_mean`, `logprob_delta_scores_mean`).
3. Row for `layer == 18` prints both the rewrite score mean and the log_delta mean side by side.

If the assertion fails, the most common cause is that the run terminated before the aggregator stage. Check `run.log`'s last lines — the aggregator runs after the last patching unit completes.
