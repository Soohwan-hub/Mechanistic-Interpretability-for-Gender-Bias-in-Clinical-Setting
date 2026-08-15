#!/usr/bin/env python3
"""
CoT next-token gender probability probe (Fig-1 style, NOT free vignettes).

For each CoT prompt (Type A/C) × condition:
  - build chat-templated CoT user message
  - force assistant suffix ending at "Gender:"
  - read P(" Male") and P(" Female") at the next token
  - no generation, no patching

Score points (match cot_patching_without_BHCs.py):
  forced_suffix       : assistant starts with "Gender:"
  after_thinking_stub : assistant "<thinking></thinking>\\nGender:"

Prompt ids: 1–10 = Type A, 11–20 = Type C (cot_simple_prompts.py).
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

from cot_simple_prompts import COT_SIMPLE_PROMPTS, cot_prompt_label

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CONDITIONS = [
    "asthma",
    "depression",
    "multiple sclerosis",
    "rheumatoid arthritis",
    "sarcoidosis",
]
SCORE_POINT_CHOICES = ("forced_suffix", "after_thinking_stub")
AFTER_THINKING_STUB_SUFFIX = "<thinking></thinking>\nGender:"


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="CoT next-token P(Male)/P(Female) baseline (no free generation)"
    )
    p.add_argument("--run-id", type=str, default="qwen_cot20_gender_probs")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(here / "cot_gender_baseline_results"),
    )
    p.add_argument("--model-name", type=str, default=MODEL_NAME)
    p.add_argument(
        "--conditions",
        type=str,
        default=",".join(DEFAULT_CONDITIONS),
        help="Comma-separated condition names (display strings).",
    )
    p.add_argument(
        "--prompt-ids",
        type=str,
        default=",".join(str(i) for i in range(1, 21)),
        help="CoT prompt ids 1–20 (A=1–10, C=11–20).",
    )
    p.add_argument(
        "--score-point",
        type=str,
        default="forced_suffix",
        choices=SCORE_POINT_CHOICES,
        help="Where Gender: is forced before reading next-token probs.",
    )
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-plot", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model load; emit synthetic probs to validate pipeline.",
    )
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load full precision / bf16 instead of 4-bit (needs more VRAM).",
    )
    return p.parse_args()


def build_prompt(tokenizer: Any, template: str, condition_name: str, score_point: str) -> str:
    body = template.replace("[CONDITION]", condition_name).replace(
        "[CONDITION_NAME]", condition_name
    )
    messages = [
        {"role": "system", "content": "You are a helpful clinical assistant."},
        {"role": "user", "content": body},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if score_point == "after_thinking_stub":
        prompt += AFTER_THINKING_STUB_SUFFIX
    else:
        prompt += "Gender:"
    return prompt


def get_single_token_id(tokenizer: Any, token_text: str) -> int:
    ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(
            f"Expected single-token encoding for {token_text!r}, got ids={ids}."
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
    fig_cond = go.Figure()
    fig_cond.add_trace(
        go.Bar(name="Mean P(Male)", x=cond_names, y=[mean(by_condition[c]["male"]) for c in cond_names])
    )
    fig_cond.add_trace(
        go.Bar(
            name="Mean P(Female)",
            x=cond_names,
            y=[mean(by_condition[c]["female"]) for c in cond_names],
        )
    )
    fig_cond.update_layout(
        barmode="group",
        title="CoT next-token gender probabilities by condition",
        xaxis_title="Condition",
        yaxis_title="Probability",
    )
    pio.write_image(fig_cond, str(out_dir / "probability_by_condition.pdf"))

    prompt_ids = sorted(by_prompt.keys())
    labels = [cot_prompt_label(p) for p in prompt_ids]
    fig_prompt = go.Figure()
    fig_prompt.add_trace(
        go.Bar(name="Mean P(Male)", x=labels, y=[mean(by_prompt[p]["male"]) for p in prompt_ids])
    )
    fig_prompt.add_trace(
        go.Bar(
            name="Mean P(Female)",
            x=labels,
            y=[mean(by_prompt[p]["female"]) for p in prompt_ids],
        )
    )
    fig_prompt.update_layout(
        barmode="group",
        title="CoT next-token gender probabilities by prompt",
        xaxis_title="Prompt",
        yaxis_title="Probability",
    )
    pio.write_image(fig_prompt, str(out_dir / "probability_by_prompt.pdf"))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    if not conditions:
        raise ValueError("No conditions provided.")

    prompt_ids = [int(x.strip()) for x in args.prompt_ids.split(",") if x.strip()]
    if not prompt_ids:
        raise ValueError("No prompt ids provided.")
    for pid in prompt_ids:
        if pid not in COT_SIMPLE_PROMPTS:
            valid = ",".join(str(x) for x in sorted(COT_SIMPLE_PROMPTS.keys()))
            raise ValueError(f"Unknown prompt id {pid}. Valid: {valid}")
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be >= 1")

    output_rows: List[Dict[str, Any]] = []
    if args.dry_run:
        for condition in conditions:
            for prompt_id in prompt_ids:
                for repeat_idx in range(args.n_repeats):
                    output_rows.append(
                        {
                            "condition": condition,
                            "prompt_id": prompt_id,
                            "prompt_label": cot_prompt_label(prompt_id),
                            "score_point": args.score_point,
                            "repeat_idx": repeat_idx,
                            "p_male": 0.5,
                            "p_female": 0.5,
                            "female_minus_male": 0.0,
                            "female_gt_male": False,
                            "male_token_id": -1,
                            "female_token_id": -1,
                        }
                    )
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        load_kwargs: Dict[str, Any] = {"device_map": "auto"}
        if not args.no_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
        model.eval()

        male_token_id = get_single_token_id(tokenizer, " Male")
        female_token_id = get_single_token_id(tokenizer, " Female")
        softmax = torch.nn.Softmax(dim=-1)

        n_total = len(conditions) * len(prompt_ids)
        done = 0
        with torch.no_grad():
            for condition in conditions:
                for prompt_id in prompt_ids:
                    done += 1
                    template = COT_SIMPLE_PROMPTS[prompt_id]
                    prompt = build_prompt(
                        tokenizer, template, condition, args.score_point
                    )
                    encoded = tokenizer(prompt, return_tensors="pt")
                    encoded = {k: v.to(model.device) for k, v in encoded.items()}
                    print(
                        f"[{done}/{n_total}] {condition} {cot_prompt_label(prompt_id)} "
                        f"({args.score_point}) tokens={encoded['input_ids'].shape[1]}",
                        flush=True,
                    )
                    for repeat_idx in range(args.n_repeats):
                        logits = model(**encoded).logits[0, -1, :]
                        probs = softmax(logits)
                        p_male = float(probs[male_token_id].item())
                        p_female = float(probs[female_token_id].item())
                        output_rows.append(
                            {
                                "condition": condition,
                                "prompt_id": prompt_id,
                                "prompt_label": cot_prompt_label(prompt_id),
                                "score_point": args.score_point,
                                "repeat_idx": repeat_idx,
                                "p_male": p_male,
                                "p_female": p_female,
                                "female_minus_male": p_female - p_male,
                                "female_gt_male": p_female > p_male,
                                "male_token_id": male_token_id,
                                "female_token_id": female_token_id,
                            }
                        )
                    print(
                        f"  P(Male)={p_male:.4f}  P(Female)={p_female:.4f}",
                        flush=True,
                    )

    raw_path = run_dir / "cot_gender_probs.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for item in output_rows:
            f.write(json.dumps(item) + "\n")

    by_condition_store: Dict[str, Dict[str, List[float]]] = {}
    by_prompt_store: Dict[int, Dict[str, List[float]]] = {}
    by_family_store: Dict[str, Dict[str, List[float]]] = {}
    for item in output_rows:
        condition = str(item["condition"])
        pid = int(item["prompt_id"])
        family = "A" if pid <= 10 else "C"

        by_condition_store.setdefault(condition, {"male": [], "female": [], "female_gt": []})
        by_condition_store[condition]["male"].append(float(item["p_male"]))
        by_condition_store[condition]["female"].append(float(item["p_female"]))
        by_condition_store[condition]["female_gt"].append(
            1.0 if item["female_gt_male"] else 0.0
        )

        by_prompt_store.setdefault(pid, {"male": [], "female": [], "female_gt": []})
        by_prompt_store[pid]["male"].append(float(item["p_male"]))
        by_prompt_store[pid]["female"].append(float(item["p_female"]))
        by_prompt_store[pid]["female_gt"].append(1.0 if item["female_gt_male"] else 0.0)

        by_family_store.setdefault(family, {"male": [], "female": [], "female_gt": []})
        by_family_store[family]["male"].append(float(item["p_male"]))
        by_family_store[family]["female"].append(float(item["p_female"]))
        by_family_store[family]["female_gt"].append(1.0 if item["female_gt_male"] else 0.0)

    by_condition = []
    for condition in sorted(by_condition_store.keys()):
        m = by_condition_store[condition]["male"]
        f = by_condition_store[condition]["female"]
        by_condition.append(
            {
                "condition": condition,
                "num_rows": len(m),
                "mean_p_male": mean(m),
                "mean_p_female": mean(f),
                "mean_female_minus_male": mean([a - b for a, b in zip(f, m)]),
                "pct_female_gt_male": 100.0 * mean(by_condition_store[condition]["female_gt"]),
            }
        )

    by_prompt = []
    for pid in sorted(by_prompt_store.keys()):
        m = by_prompt_store[pid]["male"]
        f = by_prompt_store[pid]["female"]
        by_prompt.append(
            {
                "prompt_id": pid,
                "prompt_label": cot_prompt_label(pid),
                "num_rows": len(m),
                "mean_p_male": mean(m),
                "mean_p_female": mean(f),
                "mean_female_minus_male": mean([a - b for a, b in zip(f, m)]),
                "pct_female_gt_male": 100.0 * mean(by_prompt_store[pid]["female_gt"]),
            }
        )

    by_family = []
    for family in sorted(by_family_store.keys()):
        m = by_family_store[family]["male"]
        f = by_family_store[family]["female"]
        by_family.append(
            {
                "prompt_family": family,
                "num_rows": len(m),
                "mean_p_male": mean(m),
                "mean_p_female": mean(f),
                "mean_female_minus_male": mean([a - b for a, b in zip(f, m)]),
                "pct_female_gt_male": 100.0 * mean(by_family_store[family]["female_gt"]),
            }
        )

    overall_male = [float(x["p_male"]) for x in output_rows]
    overall_female = [float(x["p_female"]) for x in output_rows]
    overall_female_gt = [1.0 if x["female_gt_male"] else 0.0 for x in output_rows]
    summary = {
        "model_name": args.model_name,
        "score_point": args.score_point,
        "conditions": conditions,
        "prompt_ids": prompt_ids,
        "n_conditions": len(conditions),
        "n_prompts": len(prompt_ids),
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "num_rows": len(output_rows),
        "overall": {
            "mean_p_male": mean(overall_male),
            "mean_p_female": mean(overall_female),
            "mean_female_minus_male": mean(
                [f - m for f, m in zip(overall_female, overall_male)]
            ),
            "pct_female_gt_male": 100.0 * mean(overall_female_gt),
        },
        "by_condition": by_condition,
        "by_prompt": by_prompt,
        "by_family": by_family,
        "interpretation_rule": (
            "Female-leaning when P(' Female') > P(' Male') at forced Gender: next token."
        ),
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
                "prompt_label",
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

    by_family_csv = run_dir / "summary_by_family.csv"
    with by_family_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_family",
                "num_rows",
                "mean_p_male",
                "mean_p_female",
                "mean_female_minus_male",
                "pct_female_gt_male",
            ],
        )
        writer.writeheader()
        for row in by_family:
            writer.writerow(row)

    if args.save_plot:
        maybe_save_plots(output_rows, run_dir)

    print(f"Saved raw rows -> {raw_path}")
    print(f"Saved summary -> {run_dir / 'summary.json'}")
    print(f"Saved condition summary CSV -> {by_condition_csv}")
    print(f"Saved prompt summary CSV -> {by_prompt_csv}")
    print(f"Saved family summary CSV -> {by_family_csv}")
    print(
        f"OVERALL  P(Male)={summary['overall']['mean_p_male']:.4f}  "
        f"P(Female)={summary['overall']['mean_p_female']:.4f}"
    )


if __name__ == "__main__":
    main()
