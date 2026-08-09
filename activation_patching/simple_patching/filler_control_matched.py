"""
Length-matched filler control — all 31 simple prompts.

For each condition × prompt_id in SIMPLE_PROMPTS:
  1. Build the exact chat-templated simple prompt (same as rewrite sweeps)
  2. Pad with neutral filler to --target-length tokens
  3. Append "Gender:" and run layer-18 MLP activation patching
  4. Record condition-token rewrite score

Usage (GPU required):
    python filler_control_matched.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --target-length 210 \
        --output-dir filler_control_matched_results

Optional: pass --artifact-dir to also record original (unpadded) lengths
from existing simple-patching pkls for side-by-side comparison.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig

from simple_patching_without_BHCs import SIMPLE_PROMPTS


FILLER_SENTENCE = (
    "The weather today is mild and the sky is partly cloudy. "
    "Birds are singing in the trees outside the window. "
    "A gentle breeze moves through the open field. "
    "The clock on the wall ticks steadily. "
)

CONDITIONS = [
    "asthma",
    "depression",
    "multiple_sclerosis",
    "rheumatoid_arthritis",
    "sarcoidosis",
]

CONDITION_DISPLAY = {
    "asthma": "asthma",
    "depression": "depression",
    "multiple_sclerosis": "multiple sclerosis",
    "rheumatoid_arthritis": "rheumatoid arthritis",
    "sarcoidosis": "sarcoidosis",
}

CONDITION_KEYWORDS = {
    "asthma": "asthma",
    "depression": "depression",
    "multiple_sclerosis": "sclerosis",
    "rheumatoid_arthritis": "arthritis",
    "sarcoidosis": "oid",
}

DEFAULT_PROMPT_IDS = ",".join(str(i) for i in sorted(SIMPLE_PROMPTS.keys()))


def _resolve(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


def get_token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, return_tensors="pt")["input_ids"][0])


def build_corrupt_body(tokenizer, template: str, condition_name: str) -> str:
    """Chat-templated prompt body (before forced Gender: suffix)."""
    body = template.replace("[CONDITION]", condition_name)
    messages = [
        {"role": "system", "content": "You are a helpful clinical assistant."},
        {"role": "user", "content": body},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def pad_to_length(tokenizer, text: str, target_length: int) -> str:
    """Append filler sentences until token count >= target_length."""
    padded = text
    while get_token_count(tokenizer, padded) < target_length:
        padded = padded + " " + FILLER_SENTENCE
    return padded


def run_layer18_sweep(
    llm: LanguageModel,
    clean_text: str,
    patch_token_from: int,
    corrupted_text: str,
    target_gender_token_id: int,
) -> Tuple[np.ndarray, List[str], float]:
    softmax = torch.nn.Softmax(dim=-1)
    corrupted_tokens = llm.tokenizer(
        corrupted_text, return_tensors="pt"
    )["input_ids"][0]
    n_tokens = len(corrupted_tokens)
    token_labels = [
        f"{llm.tokenizer.decode([corrupted_tokens[i].item()])}_{i}"
        for i in range(n_tokens)
    ]

    with torch.no_grad():
        with llm.generate(max_new_tokens=1) as tracer:
            with tracer.invoke(clean_text):
                saved = llm.model.layers[18].mlp.down_proj.output[
                    :, patch_token_from, :
                ].save()
    z_clean = _resolve(saved).detach().clone()

    with torch.no_grad():
        with llm.generate(max_new_tokens=1) as tracer:
            with tracer.invoke(corrupted_text):
                logits = llm.lm_head.output
                probs = softmax(logits[0, -1, :])
                p_proxy = probs[target_gender_token_id].save()
    corrupted_prob = float(_resolve(p_proxy).cpu().float().item())
    denom = 1.0 - corrupted_prob + 1e-8

    scores = []
    for tok_idx in range(n_tokens):
        with torch.no_grad():
            with llm.generate(max_new_tokens=1) as tracer:
                with tracer.invoke(corrupted_text):
                    z = llm.model.layers[18].mlp.down_proj.output
                    z[:, tok_idx, :] = z_clean
                    llm.model.layers[18].mlp.down_proj.output = z
                    patched_logits = llm.lm_head.output
                    patched_prob = softmax(
                        patched_logits[0, -1, :]
                    )[target_gender_token_id]
                    rs = (patched_prob - corrupted_prob) / denom
                    rs_proxy = rs.save()
        scores.append(float(_resolve(rs_proxy).cpu().float().item()))

    return np.array(scores, dtype=float).reshape(1, -1), token_labels, corrupted_prob


def find_token_idx(token_labels: List[str], keyword: str) -> int:
    for i, lab in enumerate(token_labels):
        if keyword.lower() in lab.lower():
            return i
    return -1


def build_clean_source(llm, gender: str):
    msg = {"role": "user", "content": f"The patient is {gender}"}
    clean_text = llm.tokenizer.apply_chat_template(
        [msg], tokenize=False, add_generation_prompt=True
    )
    clean_tokens = llm.tokenizer(clean_text, return_tensors="pt")["input_ids"][0]
    gender_token_ids = llm.tokenizer(
        " " + gender, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    target_id = int(gender_token_ids[-1].item())
    matches = torch.argwhere(clean_tokens == gender_token_ids[-1])
    patch_token_from = int(matches[0][0].item())
    return clean_text, patch_token_from, target_id


def optional_orig_length(artifact_dir: Optional[Path], cond: str, pid: int) -> Optional[int]:
    if artifact_dir is None:
        return None
    pkl_path = artifact_dir / f"{cond}_prompt{pid}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    labels = data.get("token_labels")
    if labels is None:
        return None
    return len(labels)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filler-pad all 31 simple prompts and run layer-18 MLP patching"
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--target-length", type=int, default=210)
    p.add_argument(
        "--prompt-ids",
        type=str,
        default=DEFAULT_PROMPT_IDS,
        help=f"Comma-separated prompt IDs (default: all {len(SIMPLE_PROMPTS)} SIMPLE_PROMPTS)",
    )
    p.add_argument(
        "--conditions",
        type=str,
        default=",".join(CONDITIONS),
        help="Comma-separated condition keys",
    )
    p.add_argument(
        "--artifact-dir",
        type=str,
        default="",
        help="Optional path to original simple-patching pkls (for orig_length only)",
    )
    p.add_argument("--output-dir", type=str, default="filler_control_matched_results")
    p.add_argument("--patch-gender", type=str, default="Male", choices=["Male", "Female"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir.strip() else None

    prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x.strip()]
    invalid = [pid for pid in prompt_ids if pid not in SIMPLE_PROMPTS]
    if invalid:
        valid = ",".join(str(x) for x in sorted(SIMPLE_PROMPTS.keys()))
        raise ValueError(f"Invalid prompt ids {invalid}. Valid: {valid}")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for cond in conditions:
        if cond not in CONDITION_DISPLAY:
            raise ValueError(f"Unknown condition {cond!r}. Valid: {list(CONDITION_DISPLAY)}")

    print(f"Loading model {args.model} ...")
    print(f"Prompts: {len(prompt_ids)}  Conditions: {len(conditions)}  "
          f"Target length: {args.target_length}")
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    llm = LanguageModel(args.model, quantization_config=quant_cfg, device_map="auto")

    clean_text, patch_token_from, target_id = build_clean_source(llm, args.patch_gender)
    all_results: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for cond in conditions:
        display = CONDITION_DISPLAY[cond]
        kw = CONDITION_KEYWORDS[cond]
        print(f"\n{'='*60}\nCondition: {cond}")
        all_results[cond] = {}

        for pid in prompt_ids:
            template = SIMPLE_PROMPTS[pid]
            body = build_corrupt_body(llm.tokenizer, template, display)
            orig_len = get_token_count(llm.tokenizer, body + "Gender:")
            artifact_len = optional_orig_length(artifact_dir, cond, pid)

            padded_body = pad_to_length(llm.tokenizer, body, args.target_length)
            corrupted_text = padded_body + "Gender:"
            padded_len = get_token_count(llm.tokenizer, corrupted_text)

            scores, labels, cp = run_layer18_sweep(
                llm, clean_text, patch_token_from, corrupted_text, target_id
            )

            cond_idx = find_token_idx(labels, kw)
            cond_score = float(scores[0, cond_idx]) if cond_idx >= 0 else None
            max_score = float(np.max(scores))
            cs_str = f"{cond_score:.4f}" if cond_score is not None else "N/A"

            print(
                f"  prompt{pid:2d}: orig_len={orig_len:3d}  padded_len={padded_len:3d}  "
                f"cond_tok_score={cs_str}  max_score={max_score:.4f}"
            )

            out_pkl = out_dir / f"{cond}_prompt{pid}_padded.pkl"
            with open(out_pkl, "wb") as f:
                pickle.dump(
                    {
                        "rewrite_scores": scores,
                        "scores": scores,
                        "token_labels": labels,
                        "corrupted_prob": cp,
                        "orig_length": orig_len,
                        "artifact_orig_length": artifact_len,
                        "padded_length": padded_len,
                        "condition": cond,
                        "prompt_id": pid,
                        "target_length": args.target_length,
                        "patch_gender": args.patch_gender,
                        "condition_token_idx": cond_idx,
                    },
                    f,
                )

            all_results[cond][pid] = {
                "orig_length": orig_len,
                "artifact_orig_length": artifact_len,
                "padded_length": padded_len,
                "condition_token_score": cond_score,
                "condition_token_idx": cond_idx,
                "max_score": max_score,
            }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n=== FINAL SUMMARY ===")
    print(
        f"{'Condition':<25} {'Prompt':<8} {'Orig len':<10} "
        f"{'Pad len':<10} {'Cond tok L18':<15} {'Max L18'}"
    )
    print("-" * 80)
    for cond, prompts in all_results.items():
        for pid, r in sorted(prompts.items()):
            cs = (
                f"{r['condition_token_score']:.4f}"
                if r["condition_token_score"] is not None
                else "N/A"
            )
            print(
                f"{cond:<25} {pid:<8} {r['orig_length']:<10} "
                f"{r['padded_length']:<10} {cs:<15} {r['max_score']:.4f}"
            )

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
