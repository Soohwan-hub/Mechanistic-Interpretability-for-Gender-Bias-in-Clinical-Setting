# Commands to Run (SAE CoT Patching)

These commands mirror your `successful_setup_commands.md` style, but are tailored for the SAE CoT script workflow in `cot_patching`.

## SSH and directory setup

```bash
ssh ubuntu@YOUR_SERVER_IP
mkdir -p ~/sae_cot_patching
cd ~/sae_cot_patching
```

## SCP uploads (run from Windows PowerShell)

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519" "C:\Users\soohw\rtar_parent\Simple_Patching\Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting\activation_patching\cot_patching\sae_localization.py" ubuntu@YOUR_SERVER_IP:~/sae_cot_patching/
scp -i "$env:USERPROFILE\.ssh\id_ed25519" "C:\Users\soohw\rtar_parent\Simple_Patching\Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting\activation_patching\cot_patching\commands_to_run.md" ubuntu@YOUR_SERVER_IP:~/sae_cot_patching/
```

## Virtual environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
which python
python -V
```

## Packaging tools upgrade

```bash
python -m pip install --upgrade pip setuptools wheel
```

## Hugging Face token (recommended before model/SAE download)

```bash
export HF_TOKEN="hf_xxx_your_token_here"
```

## Dependency installation (GH200/A10-safe baseline)

```bash
pip install "numpy<2.0" pandas scipy scikit-learn tqdm plotly pyarrow kaleido huggingface_hub
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformer_lens sae_lens
```

## Quick environment checks

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpu_count:', torch.cuda.device_count())"
python -c "import numpy, scipy, sklearn; print(numpy.__version__, scipy.__version__, sklearn.__version__)"
ls
```

## Dry run

```bash
python sae_localization.py --stage all --run-id gh200_full --runtime-profile gh200 --dry-run
```

## Stage-by-stage run (recommended first pass)

```bash
python sae_localization.py --stage 1 --run-id gh200_full --runtime-profile gh200
python sae_localization.py --stage 2 --run-id gh200_full --runtime-profile gh200 --resume
python sae_localization.py --stage 3 --run-id gh200_full --runtime-profile gh200 --resume
python sae_localization.py --stage 4 --run-id gh200_full --runtime-profile gh200 --resume
python sae_localization.py --stage 5 --run-id gh200_full --runtime-profile gh200 --resume
python sae_localization.py --stage 6 --run-id gh200_full --runtime-profile gh200 --resume --save-plots --plot-format png
```

## Full pipeline one-shot (GH200)

```bash
python sae_localization.py --stage all --run-id gh200_full --runtime-profile gh200 --resume --save-plots --plot-format png
```

## A10 fallback run (more conservative)

```bash
python sae_localization.py --stage all --run-id a10_safe --runtime-profile a10 --resume --max-sweep-coords 25000 --max-control-rows 10000
```

## Output location

Artifacts are written under:

```bash
sae_localization_runs/<run-id>/artifacts/
```

Useful outputs:
- `baseline_traces.json`
- `run_index.json`
- `contrastive_pairs.json`
- `top_latents_per_layer.json`
- `sweep_coords.json`
- `sweep_results.parquet`
- `controls_results.parquet`
- `causal_shortlist.csv`
- `control_stats.csv`
- `timeline_summary.csv`

## Replication-only script (`sae_replication.py`)

### Dry run

```bash
python sae_replication.py --run-id rep_top5 --dry-run
```

### Top-N shortlist replication (recommended)

```bash
python sae_replication.py \
  --run-id rep_top5 \
  --shortlist-csv gh200_full_artifacts_x86/causal_shortlist.csv \
  --top-n 5 \
  --seeds 42,43,44,45,46 \
  --resume
```

### Top-N shortlist replication + in-depth plots

```bash
python sae_replication.py \
  --run-id rep_top5_plots \
  --shortlist-csv gh200_full_artifacts_x86/causal_shortlist.csv \
  --top-n 5 \
  --seeds 42,43,44,45,46 \
  --save-plots \
  --plot-format png \
  --max-plot-tokens 40 \
  --max-plot-features-for-box 20 \
  --resume
```

### Manual feature list replication

```bash
python sae_replication.py \
  --run-id rep_manual \
  --feature-list 11:87380,3:87385 \
  --seeds 42,43,44,45,46 \
  --resume
```

### Replication output location

```bash
sae_replication_runs/<run-id>/artifacts/
```

Replication outputs:
- `replication_targets.csv`
- `replication_baselines.json`
- `replication_trace_index.json`
- `replication_activation_cache.json`
- `replication_raw.parquet`
- `replication_summary.csv`
- `replication_pass_fail.csv`
- `plots/feature_token_mean_effect_heatmap.png`
- `plots/feature_token_sign_consistency_heatmap.png`
- `plots/replication_decision_scatter.png`
- `plots/feature_temperature_profile.png`
- `plots/feature_temperature_sign_consistency.png`
- `plots/feature_seed_effect_stability.png`
- `plots/feature_seed_sign_stability.png`
- `plots/feature_norm_effect_boxplot.png` (when feature count is not too large)

## Sign-aware gating + full global in-depth sweep

### Localization full pipeline (sign-aware, global)

```bash
python sae_localization.py \
  --stage all \
  --run-id gh200_full_signaware_global \
  --runtime-profile gh200 \
  --gating-mode sign_aware \
  --max-sweep-coords 0 \
  --max-control-rows 0 \
  --resume \
  --save-plots \
  --plot-format png
```

Notes:
- `--gating-mode sign_aware` runs both `f_k > z` and `f_k < -z` separately.
- `--max-sweep-coords 0` keeps full global sweep (no truncation).
- Default `SAE_LAYERS` now use a combined andyrdt+Geaming superset:
  `3,4,7,11,12,15,18,19,20,23,25,27`.

### Replication full-global sign-aware sweep

```bash
python sae_replication.py \
  --run-id rep_signaware_global \
  --shortlist-csv gh200_full_artifacts_x86/causal_shortlist.csv \
  --top-n 99999 \
  --gating-mode sign_aware \
  --seeds 42,43,44,45,46 \
  --save-plots \
  --plot-format png \
  --resume
```

Notes:
- Use a large `--top-n` (e.g. `99999`) to include all shortlist rows in a full global replication pass.
- Outputs are sign-separated via `gate_sign` in CSV/parquet artifacts.

## Attention-head CoT script (`attention_head_cot_patching.py`)

### Upload script (run from Windows PowerShell)

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519" "C:\Users\soohw\rtar_parent\Simple_Patching\Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting\activation_patching\cot_patching\attention_head_cot_patching.py" ubuntu@YOUR_SERVER_IP:~/sae_cot_patching/
```

### Install additional dependencies (inside venv on server)

```bash
pip install nnsight transformers accelerate bitsandbytes
pip install matplotlib seaborn
```

### Dry run

```bash
python attention_head_cot_patching.py \
  --run-id head_dryrun \
  --base-save-dir head_cot_patching_runs \
  --dry-run \
  --load-in-4bit
```

### Smoke test first (recommended)

```bash
python attention_head_cot_patching.py \
  --run-id head_smoke \
  --base-save-dir head_cot_patching_runs \
  --do-smoke-test \
  --smoke-condition "rheumatoid arthritis" \
  --smoke-prompt-label A \
  --smoke-var var2 \
  --max-new-tokens 700 \
  --num-layers-batch 2 \
  --num-heads-batch 8 \
  --load-in-4bit
```

### Full parity run (all configured conditions x prompt types x variations)

```bash
python attention_head_cot_patching.py \
  --run-id gh200_head_full \
  --base-save-dir head_cot_patching_runs \
  --conditions "rheumatoid arthritis,asthma,bronchitis,essential hypertension,depression" \
  --prompt-labels "A,C" \
  --do-smoke-test \
  --smoke-condition "rheumatoid arthritis" \
  --smoke-prompt-label A \
  --smoke-var var2 \
  --max-new-tokens 700 \
  --num-layers-batch 2 \
  --num-heads-batch 8 \
  --save-plots \
  --load-in-4bit
```

### Plot-only rerun from saved artifacts

```bash
python attention_head_cot_patching.py \
  --run-id gh200_head_full \
  --base-save-dir head_cot_patching_runs \
  --conditions "rheumatoid arthritis,asthma,bronchitis,essential hypertension,depression" \
  --prompt-labels "A,C" \
  --plot-only \
  --save-plots
```
