"""
Standalone baseline probe for female-stereotyped conditions:
- no activation patching
- no BHC data
- uses 31 prompt templates from simple_patching_without_BHCs.py
- computes only P(" Male") and P(" Female") at forced "Gender:" next token
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

from simple_patching_without_BHCs import SIMPLE_PROMPTS


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CONDITIONS = [
    "depression",
    "multiple sclerosis",
    "sarcoidosis",
    "rheumatoid arthritis",
    "asthma",
    "bronchitis",
    "essential hypertension",
]


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Prompt-only baseline gender probability probe for female-stereotyped conditions"
    )
    parser.add_argument("--run-id", type=str, default="female_condition_baseline")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(here / "female_condition_baseline_results"),
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=",".join(DEFAULT_CONDITIONS),
        help="Comma-separated condition names",
    )
    parser.add_argument(
        "--prompt-ids",
        type=str,
        default=",".join(str(x) for x in sorted(SIMPLE_PROMPTS.keys())),
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Repeated evaluations per (condition, prompt).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reserved for reproducibility metadata.")
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save probability comparison plots by condition and by prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model load and emit synthetic probabilities for pipeline checks.",
    )
    return parser.parse_args()


def build_prompt(tokenizer: Any, template: str, condition_name: str) -> str:
    body = template.replace("[CONDITION]", condition_name)
    messages = [
        {"role": "system", "content": "You are a helpful clinical assistant."},
        {"role": "user", "content": body},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt += "Gender:"
    return prompt


def get_single_token_id(tokenizer: Any, token_text: str) -> int:
    ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(
            f"Expected single-token encoding for {token_text!r}, got ids={ids}. "
            "Update token handling before running this probe."
        )
    return int(ids[0])


def maybe_save_plots(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    if not _HAS_PLOTLY or not rows:
        return

    by_condition: Dict[str, Dict[str, List[float]]] = {}
    by_prompt: Dict[int, Dict[str, List[float]]] = {}
    for row in rows:
        condition = str(row["condition"])
        prompt_id = int(row["prompt_id"])
        by_condition.setdefault(condition, {"male": [], "female": []})
        by_condition[condition]["male"].append(float(row["p_male"]))
        by_condition[condition]["female"].append(float(row["p_female"]))

        by_prompt.setdefault(prompt_id, {"male": [], "female": []})
        by_prompt[prompt_id]["male"].append(float(row["p_male"]))
        by_prompt[prompt_id]["female"].append(float(row["p_female"]))

    cond_names = sorted(by_condition.keys())
    cond_male = [mean(by_condition[c]["male"]) for c in cond_names]
    cond_female = [mean(by_condition[c]["female"]) for c in cond_names]
    fig_cond = go.Figure()
    fig_cond.add_trace(go.Bar(name="Mean P(Male)", x=cond_names, y=cond_male))
    fig_cond.add_trace(go.Bar(name="Mean P(Female)", x=cond_names, y=cond_female))
    fig_cond.update_layout(
        barmode="group",
        title="Next-token gender probabilities by condition",
        xaxis_title="Condition",
        yaxis_title="Probability",
    )
    pio.write_image(fig_cond, str(out_dir / "probability_by_condition.pdf"))

    prompt_ids = sorted(by_prompt.keys())
    prompt_male = [mean(by_prompt[p]["male"]) for p in prompt_ids]
    prompt_female = [mean(by_prompt[p]["female"]) for p in prompt_ids]
    fig_prompt = go.Figure()
    fig_prompt.add_trace(go.Bar(name="Mean P(Male)", x=[str(p) for p in prompt_ids], y=prompt_male))
    fig_prompt.add_trace(go.Bar(name="Mean P(Female)", x=[str(p) for p in prompt_ids], y=prompt_female))
    fig_prompt.update_layout(
        barmode="group",
        title="Next-token gender probabilities by prompt",
        xaxis_title="Prompt ID",
        yaxis_title="Probability",
    )
    pio.write_image(fig_prompt, str(out_dir / "probability_by_prompt.pdf"))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    if not conditions:
        raise ValueError("No conditions provided. Use --conditions.")

    prompt_ids = [int(x.strip()) for x in args.prompt_ids.split(",") if x.strip()]
    if not prompt_ids:
        raise ValueError("No prompt IDs provided. Use --prompt-ids.")
    for pid in prompt_ids:
        if pid not in SIMPLE_PROMPTS:
            valid_ids = ",".join(str(x) for x in sorted(SIMPLE_PROMPTS.keys()))
            raise ValueError(f"Unknown prompt id {pid}. Valid prompt ids: {valid_ids}")
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be >= 1")

    output_rows: List[Dict[str, Any]] = []
    if args.dry_run:
        for condition in conditions:
            for prompt_id in prompt_ids:
                for repeat_idx in range(args.n_repeats):
                    p_male = 0.5
                    p_female = 0.5
                    output_rows.append(
                        {
                            "condition": condition,
                            "prompt_id": prompt_id,
                            "repeat_idx": repeat_idx,
                            "p_male": p_male,
                            "p_female": p_female,
                            "female_minus_male": p_female - p_male,
                            "female_gt_male": False,
                            "male_token_id": -1,
                            "female_token_id": -1,
                        }
                    )
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
        )
        model.eval()

        male_token_id = get_single_token_id(tokenizer, " Male")
        female_token_id = get_single_token_id(tokenizer, " Female")
        softmax = torch.nn.Softmax(dim=-1)

        with torch.no_grad():
            for condition in conditions:
                for prompt_id in prompt_ids:
                    prompt = build_prompt(tokenizer, SIMPLE_PROMPTS[prompt_id], condition)
                    encoded = tokenizer(prompt, return_tensors="pt")
                    encoded = {k: v.to(model.device) for k, v in encoded.items()}
                    for repeat_idx in range(args.n_repeats):
                        logits = model(**encoded).logits[0, -1, :]
                        probs = softmax(logits)

                        p_male = float(probs[male_token_id].item())
                        p_female = float(probs[female_token_id].item())
                        output_rows.append(
                            {
                                "condition": condition,
                                "prompt_id": prompt_id,
                                "repeat_idx": repeat_idx,
                                "p_male": p_male,
                                "p_female": p_female,
                                "female_minus_male": p_female - p_male,
                                "female_gt_male": p_female > p_male,
                                "male_token_id": male_token_id,
                                "female_token_id": female_token_id,
                            }
                        )

    raw_path = run_dir / "female_condition_probs.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for item in output_rows:
            f.write(json.dumps(item) + "\n")

    by_condition_store: Dict[str, Dict[str, List[float]]] = {}
    by_prompt_store: Dict[int, Dict[str, List[float]]] = {}
    for item in output_rows:
        condition = str(item["condition"])
        pid = int(item["prompt_id"])

        by_condition_store.setdefault(condition, {"male": [], "female": [], "female_gt": []})
        by_condition_store[condition]["male"].append(float(item["p_male"]))
        by_condition_store[condition]["female"].append(float(item["p_female"]))
        by_condition_store[condition]["female_gt"].append(1.0 if item["female_gt_male"] else 0.0)

        by_prompt_store.setdefault(pid, {"male": [], "female": [], "female_gt": []})
        by_prompt_store[pid]["male"].append(float(item["p_male"]))
        by_prompt_store[pid]["female"].append(float(item["p_female"]))
        by_prompt_store[pid]["female_gt"].append(1.0 if item["female_gt_male"] else 0.0)

    by_condition = []
    for condition in sorted(by_condition_store.keys()):
        by_condition.append(
            {
                "condition": condition,
                "num_rows": len(by_condition_store[condition]["male"]),
                "mean_p_male": mean(by_condition_store[condition]["male"]),
                "mean_p_female": mean(by_condition_store[condition]["female"]),
                "mean_female_minus_male": mean(
                    [f - m for f, m in zip(by_condition_store[condition]["female"], by_condition_store[condition]["male"])]
                ),
                "pct_female_gt_male": 100.0 * mean(by_condition_store[condition]["female_gt"]),
            }
        )

    by_prompt = []
    for pid in sorted(by_prompt_store.keys()):
        by_prompt.append(
            {
                "prompt_id": pid,
                "num_rows": len(by_prompt_store[pid]["male"]),
                "mean_p_male": mean(by_prompt_store[pid]["male"]),
                "mean_p_female": mean(by_prompt_store[pid]["female"]),
                "mean_female_minus_male": mean(
                    [f - m for f, m in zip(by_prompt_store[pid]["female"], by_prompt_store[pid]["male"])]
                ),
                "pct_female_gt_male": 100.0 * mean(by_prompt_store[pid]["female_gt"]),
            }
        )

    overall_male = [float(x["p_male"]) for x in output_rows]
    overall_female = [float(x["p_female"]) for x in output_rows]
    overall_female_gt = [1.0 if x["female_gt_male"] else 0.0 for x in output_rows]
    summary = {
        "model_name": MODEL_NAME,
        "conditions": conditions,
        "target_gender": "Female",
        "n_conditions": len(conditions),
        "prompt_ids": prompt_ids,
        "n_prompts": len(prompt_ids),
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "num_rows": len(output_rows),
        "overall": {
            "mean_p_male": mean(overall_male),
            "mean_p_female": mean(overall_female),
            "mean_female_minus_male": mean([f - m for f, m in zip(overall_female, overall_male)]),
            "pct_female_gt_male": 100.0 * mean(overall_female_gt),
        },
        "by_condition": by_condition,
        "by_prompt": by_prompt,
        "interpretation_rule": "Female-leaning when P(' Female') > P(' Male') consistently across conditions/prompts.",
    }
    atomic_write_json(run_dir / "summary.json", summary)

    by_condition_csv = run_dir / "summary_by_condition.csv"
    with by_condition_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "num_rows",
                "mean_p_male",
                "mean_p_female",
                "mean_female_minus_male",
                "pct_female_gt_male",
            ],
        )
        writer.writeheader()
        for row in by_condition:
            writer.writerow(row)

    by_prompt_csv = run_dir / "summary_by_prompt.csv"
    with by_prompt_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_id",
                "num_rows",
                "mean_p_male",
                "mean_p_female",
                "mean_female_minus_male",
                "pct_female_gt_male",
            ],
        )
        writer.writeheader()
        for row in by_prompt:
            writer.writerow(row)

    if args.save_plot:
        maybe_save_plots(output_rows, run_dir)

    print(f"Saved raw rows -> {raw_path}")
    print(f"Saved summary -> {run_dir / 'summary.json'}")
    print(f"Saved condition summary CSV -> {by_condition_csv}")
    print(f"Saved prompt summary CSV -> {by_prompt_csv}")


if __name__ == "__main__":
    main()
