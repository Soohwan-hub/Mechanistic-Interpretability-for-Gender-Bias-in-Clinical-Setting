# Qwen CoT behavioral gender rates (n=35)

Free-generation (no patching) Male/Female rates under CoT prompts for Qwen2.5-7B-Instruct.

## Setup

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Conditions: asthma, depression, MS, RA, sarcoidosis (5)
- Prompts: CoT Type A (ids 1–10) + Type C (ids 11–20) from `cot_thinking_prompts.py`
- Generations per cell: 35 → **3500** total
- Classifier: paper Male/Female line parser

## Code

- `../../generate_cot_behavioral_vignettes.py`
- `../../run_cot_behavioral_n35.sh`
- `../../run_cot_behavioral_smoke.sh`
- `../../cot_thinking_prompts.py`

## Expected artifacts (after full run)

- `config.json` — full run config
- `generations/*.jsonl` — one file per condition×prompt cell
- `summary_gender_rates.tsv` — Male/Female rates by condition and prompt
- `summary_by_condition.tsv` — rates aggregated by condition (if written)

## Note

Full n=35 outputs are produced on the GPU host. If this folder only has a stub `config.json`, pull from the Lambda run directory before sharing.
