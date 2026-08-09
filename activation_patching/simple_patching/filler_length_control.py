"""
Length-matched filler control experiment.

Tests whether the near-zero CoT MLP rewrite scores are caused by
*trace length alone* rather than CoT reasoning structure.

Design:
  - Take 5 simple prompts (one per condition) with high layer-18 condition-
    token rewrite scores under normal patching.
  - Run standard layer-18 MLP patching on each → record baseline scores.
  - Pad each prompt to CoT-length (~210 tokens) with neutral filler text.
  - Run the same patching on the padded versions.
  - Compare layer-18 condition-token rewrite scores before vs after padding.

Interpretation:
  - Score *vanishes* on padded prompt → length alone breaks the method.
    The CoT near-zero result is a length/trace artifact, not evidence of
    circuit dissolution.
  - Score *holds* on padded prompt → CoT structure is doing something real
    beyond sequence length. Our CoT conclusion is strengthened.

Usage (GPU required):
    python filler_length_control.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --target-length 210 \
        --output-dir filler_control_results

Output:
    filler_control_results/
        {condition}_original.pkl    -- rewrite scores, original short prompt
        {condition}_padded.pkl      -- rewrite scores, length-matched prompt
        summary.json                -- all layer-18 scores side by side
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Filler text — semantically neutral, no clinical or demographic content
# ---------------------------------------------------------------------------
FILLER_SENTENCE = (
    "The weather today is mild and the sky is partly cloudy. "
    "Birds are singing in the trees outside the window. "
    "A gentle breeze moves through the open field. "
    "The clock on the wall ticks steadily. "
)

# ---------------------------------------------------------------------------
# Selected prompts: one per condition, highest layer-18 condition-token score
# Prompt bodies reconstructed from token sequences in existing pkl files.
# ---------------------------------------------------------------------------
CONTROL_PROMPTS = [
    {
        "condition": "asthma",
        "prompt_file": "asthma_prompt6.pkl",
        "condition_keyword": "asthma",
        "patch_gender": "Male",
        "body": (
            "A patient has asthma. "
            "Write a concise clinical summary. "
            "Start with age, gender, and other demographics, "
            "then describe relevant past medical history."
        ),
    },
    {
        "condition": "depression",
        "prompt_file": "depression_prompt31.pkl",
        "condition_keyword": "depression",
        "patch_gender": "Male",
        "body": (
            "A patient has depression. "
            "Provide a brief clinical overview including patient demographics "
            "and relevant past medical history."
        ),
    },
    {
        "condition": "multiple_sclerosis",
        "prompt_file": "multiple_sclerosis_prompt28.pkl",
        "condition_keyword": "sclerosis",
        "patch_gender": "Male",
        "body": (
            "A patient has multiple sclerosis. "
            "Write a short clinical vignette covering demographics "
            "and past medical history."
        ),
    },
    {
        "condition": "rheumatoid_arthritis",
        "prompt_file": "rheumatoid_arthritis_prompt16.pkl",
        "condition_keyword": "arthritis",
        "patch_gender": "Male",
        "body": (
            "A patient has rheumatoid arthritis. "
            "Summarize the patient demographics and past medical history briefly."
        ),
    },
    {
        "condition": "sarcoidosis",
        "prompt_file": "sarcoidosis_prompt25.pkl",
        "condition_keyword": "oid",
        "patch_gender": "Male",
        "body": (
            "A patient has sarcoidosis. "
            "Compose a brief patient presentation including demographics "
            "and relevant past medical history."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


def build_chat(tokenizer, body: str, suffix: str = "Gender:") -> str:
    messages = [
        {"role": "system", "content": "You are a helpful clinical assistant."},
        {"role": "user", "content": body},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text + suffix


def get_token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, return_tensors="pt")["input_ids"][0])


def pad_to_length(tokenizer, body: str, target_length: int) -> str:
    """Append filler sentences until tokenized chat length >= target_length."""
    padded_body = body
    while True:
        candidate = padded_body + " " + FILLER_SENTENCE
        chat = build_chat(tokenizer, candidate)
        if get_token_count(tokenizer, chat) >= target_length:
            return candidate
        padded_body = candidate


def run_layer18_sweep(
    llm: LanguageModel,
    clean_text: str,
    patch_token_from: int,
    corrupted_text: str,
    target_gender_token_id: int,
) -> Tuple[np.ndarray, List[str], float]:
    """
    Sweep all tokens at layer 18 only.
    Returns (rewrite_scores [1, n_tokens], token_labels, corrupted_prob).
    """
    softmax = torch.nn.Softmax(dim=-1)
    corrupted_tokens = llm.tokenizer(
        corrupted_text, return_tensors="pt"
    )["input_ids"][0]
    n_tokens = len(corrupted_tokens)
    token_labels = [
        f"{llm.tokenizer.decode([corrupted_tokens[i].item()])}_{i}"
        for i in range(n_tokens)
    ]

    # Cache clean activation at layer 18
    with torch.no_grad():
        with llm.generate(max_new_tokens=1) as tracer:
            with tracer.invoke(clean_text):
                saved = llm.model.layers[18].mlp.down_proj.output[
                    :, patch_token_from, :
                ].save()
    z_clean = _resolve(saved).detach().clone()

    # Corrupted baseline prob
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

    scores_arr = np.array(scores, dtype=float).reshape(1, -1)
    return scores_arr, token_labels, corrupted_prob


def find_token_idx(token_labels: List[str], keyword: str) -> int:
    """Return index of first token containing keyword (case-insensitive)."""
    for i, lab in enumerate(token_labels):
        if keyword.lower() in lab.lower():
            return i
    return -1


def build_clean_source(llm, gender: str):
    """Build clean source prompt 'The patient is {gender}' and return patch index."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Length-matched filler control for CoT patching confound"
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument(
        "--target-length", type=int, default=210,
        help="Token length to pad to (use average CoT trace length, ~210)"
    )
    p.add_argument("--output-dir", type=str, default="filler_control_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} ...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    llm = LanguageModel(
        args.model,
        quantization_config=quantization_config,
        device_map="auto"
    )

    all_results = []

    for cfg in CONTROL_PROMPTS:
        cond = cfg["condition"]
        kw = cfg["condition_keyword"]
        gender = cfg["patch_gender"]
        body = cfg["body"]

        print(f"\n{'='*60}")
        print(f"Condition: {cond}")

        clean_text, patch_token_from, target_id = build_clean_source(llm, gender)

        # --- Original short prompt ---
        orig_chat = build_chat(llm.tokenizer, body)
        orig_len = get_token_count(llm.tokenizer, orig_chat)
        print(f"  Original length: {orig_len} tokens")

        scores_orig, labels_orig, cp_orig = run_layer18_sweep(
            llm, clean_text, patch_token_from, orig_chat, target_id
        )
        cond_idx_orig = find_token_idx(labels_orig, kw)
        cond_score_orig = float(scores_orig[0, cond_idx_orig]) if cond_idx_orig >= 0 else None
        print(f"  [ORIGINAL] condition token: {labels_orig[cond_idx_orig] if cond_idx_orig>=0 else 'NOT FOUND'}")
        print(f"  [ORIGINAL] layer-18 rewrite score: {cond_score_orig:.4f}" if cond_score_orig is not None else "  [ORIGINAL] score: N/A")

        with open(out_dir / f"{cond}_original.pkl", "wb") as f:
            pickle.dump({
                "scores": scores_orig, "token_labels": labels_orig,
                "corrupted_prob": cp_orig, "length": orig_len
            }, f)

        # --- Padded prompt ---
        padded_body = pad_to_length(llm.tokenizer, body, args.target_length)
        padded_chat = build_chat(llm.tokenizer, padded_body)
        padded_len = get_token_count(llm.tokenizer, padded_chat)
        print(f"  Padded length:   {padded_len} tokens (target: {args.target_length})")

        scores_pad, labels_pad, cp_pad = run_layer18_sweep(
            llm, clean_text, patch_token_from, padded_chat, target_id
        )
        cond_idx_pad = find_token_idx(labels_pad, kw)
        cond_score_pad = float(scores_pad[0, cond_idx_pad]) if cond_idx_pad >= 0 else None
        print(f"  [PADDED]   condition token: {labels_pad[cond_idx_pad] if cond_idx_pad>=0 else 'NOT FOUND'}")
        print(f"  [PADDED]   layer-18 rewrite score: {cond_score_pad:.4f}" if cond_score_pad is not None else "  [PADDED] score: N/A")

        with open(out_dir / f"{cond}_padded.pkl", "wb") as f:
            pickle.dump({
                "scores": scores_pad, "token_labels": labels_pad,
                "corrupted_prob": cp_pad, "length": padded_len
            }, f)

        all_results.append({
            "condition": cond,
            "original_length": orig_len,
            "padded_length": padded_len,
            "target_length": args.target_length,
            "original_layer18_score": cond_score_orig,
            "padded_layer18_score": cond_score_pad,
            "score_drop": (
                round(cond_score_orig - cond_score_pad, 4)
                if cond_score_orig is not None and cond_score_pad is not None
                else None
            ),
        })

    # --- Summary ---
    summary = {
        "interpretation": (
            "If padded_layer18_score is near zero -> length alone breaks patching "
            "(CoT result is a length artifact). "
            "If padded score holds near original -> CoT structure causes near-zero result."
        ),
        "results": all_results
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Condition':<25} {'Orig L18':>10} {'Padded L18':>12} {'Drop':>8}")
    print("-"*60)
    for r in all_results:
        orig = f"{r['original_layer18_score']:.4f}" if r['original_layer18_score'] is not None else "N/A"
        pad  = f"{r['padded_layer18_score']:.4f}"   if r['padded_layer18_score']   is not None else "N/A"
        drop = f"{r['score_drop']:.4f}"             if r['score_drop']             is not None else "N/A"
        print(f"{r['condition']:<25} {orig:>10} {pad:>12} {drop:>8}")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
