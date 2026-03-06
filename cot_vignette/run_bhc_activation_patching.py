import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig

from cot_vignette.patch_targets import (
    extract_head_slice,
    get_patch_tensor,
    patch_token_head_slice,
    patch_token_vector,
)
from cot_vignette.prompts_config import BHC_PROMPT_KEYS_5_PER_TYPE, CORRUPT_PROMPTS


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

COHORT_TO_FILE = {
    "depression": "data/depression_cases.jsonl",
    "hf": "data/hf_cases.jsonl",
}

COHORT_TO_CONDITION_NAME = {
    "depression": "depression",
    "hf": "heart failure",
}


@dataclass
class RunConfig:
    patch_target: str
    n_cases: int
    prompt_keys: List[str]
    cohorts: List[str]
    output_dir: str
    seed: int
    original_gender: str
    target_gender: str
    max_tokens: int
    head_idx: int


def _resolve(x: Any):
    return x.value if hasattr(x, "value") else x


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _prepare_clean_prompt(llm: LanguageModel, target_gender: str) -> Dict[str, Any]:
    message = {"role": "user", "content": f"The patient is {target_gender}"}
    clean_prompt = llm.tokenizer.apply_chat_template(
        [message], tokenize=False, add_generation_prompt=True
    )
    clean_tokens = llm.tokenizer(clean_prompt, return_tensors="pt")["input_ids"][0]
    target_token_ids = llm.tokenizer(
        " " + target_gender,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    patch_token_from = torch.argwhere(clean_tokens == target_token_ids[-1])[0][0].tolist()
    return {
        "clean_prompt": clean_prompt,
        "clean_tokens": clean_tokens,
        "patch_token_from": patch_token_from,
    }


def _prepare_corrupt_prompt(
    llm: LanguageModel,
    prompt_template: str,
    condition_name: str,
    bhc_text: str,
) -> Dict[str, Any]:
    user_text = prompt_template.replace("[CONDITION]", bhc_text)
    user_text = user_text.replace("[CONDITION_NAME]", condition_name)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    corrupted_prompt = llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    corrupted_prompt += "Gender:"
    corrupted_tokens = llm.tokenizer(
        corrupted_prompt,
        return_tensors="pt",
    )["input_ids"][0]
    return {
        "corrupted_prompt": corrupted_prompt,
        "corrupted_tokens": corrupted_tokens,
    }


def _cache_clean_vectors(
    llm: LanguageModel,
    patch_target: str,
    patch_token_from: int,
    num_layers_model: int,
    num_heads: int,
    head_idx: int,
    clean_prompt: str,
) -> Dict[int, torch.Tensor]:
    saved: Dict[int, Any] = {}
    with torch.no_grad():
        with llm.generate(max_new_tokens=1) as tracer:
            with tracer.invoke(clean_prompt):
                for layer_idx in range(num_layers_model):
                    layer = llm.model.layers[layer_idx]
                    act = get_patch_tensor(layer, patch_target)
                    vec = act[patch_token_from, :]
                    if patch_target == "attn_head":
                        vec = extract_head_slice(vec, num_heads=num_heads, head_idx=head_idx)
                    saved[layer_idx] = vec.save()
    return {i: _resolve(proxy).detach().clone() for i, proxy in saved.items()}


def _compute_corrupted_target_prob(
    llm: LanguageModel,
    corrupted_prompt: str,
    target_gender_token: int,
) -> float:
    softmax = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        with llm.generate(max_new_tokens=1) as tracer:
            with tracer.invoke(corrupted_prompt):
                logits = llm.lm_head.output
                probs = softmax(logits[0, -1, :])
                p = probs[target_gender_token].save()
    return float(_resolve(p).cpu().float().item())


def _run_patch_sweep(
    llm: LanguageModel,
    patch_target: str,
    clean_activations_cache: Dict[int, torch.Tensor],
    corrupted_prompt: str,
    corrupted_tokens: torch.Tensor,
    offset: int,
    target_gender_token: int,
    corrupted_prob_target: float,
    num_layers_model: int,
    num_heads: int,
    head_idx: int,
    max_tokens: int,
) -> Tuple[List[float], List[float]]:
    softmax = torch.nn.Softmax(dim=-1)
    per_layer_mean: List[float] = []
    per_layer_max: List[float] = []
    token_count = len(corrupted_tokens) if max_tokens <= 0 else min(max_tokens, len(corrupted_tokens))

    for layer_idx in range(num_layers_model):
        layer_scores: List[float] = []
        for token_idx in range(token_count):
            with torch.no_grad():
                with llm.generate(max_new_tokens=1) as tracer:
                    with tracer.invoke(corrupted_prompt):
                        layer = llm.model.layers[layer_idx]
                        patch_idx = token_idx + offset
                        if patch_target == "attn_head":
                            patch_token_head_slice(
                                layer=layer,
                                token_idx=patch_idx,
                                clean_head_slice=clean_activations_cache[layer_idx],
                                num_heads=num_heads,
                                head_idx=head_idx,
                            )
                        else:
                            patch_token_vector(
                                layer=layer,
                                patch_target=patch_target,
                                token_idx=patch_idx,
                                clean_vector=clean_activations_cache[layer_idx],
                            )

                        patched_logits = llm.lm_head.output
                        patched_prob = softmax(patched_logits[0, -1, :])[target_gender_token]
                        rewrite_score = (patched_prob - corrupted_prob_target) / (1 - corrupted_prob_target)
                        score = rewrite_score.save()
            layer_scores.append(float(_resolve(score).cpu().float().item()))

        if layer_scores:
            per_layer_mean.append(float(sum(layer_scores) / len(layer_scores)))
            per_layer_max.append(float(max(layer_scores)))
        else:
            per_layer_mean.append(float("nan"))
            per_layer_max.append(float("nan"))

    return per_layer_mean, per_layer_max


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_aggregate_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "cohort",
        "prompt_key",
        "layer",
        "mean_rewrite_score",
        "max_rewrite_score",
        "num_cases",
        "patch_target",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-target", default="mlp", choices=["residual", "mlp", "attn", "attn_head"])
    parser.add_argument("--n-cases", type=int, default=30)
    parser.add_argument("--prompts", default=",".join(BHC_PROMPT_KEYS_5_PER_TYPE))
    parser.add_argument("--cohorts", default="depression,hf")
    parser.add_argument("--output-dir", default="/home/ubuntu/patching_results/bhc_prompt_sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--original-gender", default="Female")
    parser.add_argument("--target-gender", default="Male")
    parser.add_argument("--max-tokens", type=int, default=0, help="0 means full token sweep")
    parser.add_argument("--head-idx", type=int, default=0)
    args = parser.parse_args()
    return RunConfig(
        patch_target=args.patch_target,
        n_cases=args.n_cases,
        prompt_keys=[p.strip() for p in args.prompts.split(",") if p.strip()],
        cohorts=[c.strip() for c in args.cohorts.split(",") if c.strip()],
        output_dir=args.output_dir,
        seed=args.seed,
        original_gender=args.original_gender,
        target_gender=args.target_gender,
        max_tokens=args.max_tokens,
        head_idx=args.head_idx,
    )


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    llm = LanguageModel(MODEL_NAME, quantization_config=quantization_config, device_map="auto")
    num_layers_model = len(llm.model.layers)
    num_heads = llm.model.config.num_attention_heads

    clean_prompt_output = _prepare_clean_prompt(llm, cfg.target_gender)
    clean_prompt = clean_prompt_output["clean_prompt"]
    clean_tokens = clean_prompt_output["clean_tokens"]
    patch_token_from = clean_prompt_output["patch_token_from"]
    target_gender_token = llm.tokenizer(" " + cfg.target_gender, add_special_tokens=False)["input_ids"][-1]

    random.seed(cfg.seed)
    all_case_rows: List[Dict[str, Any]] = []

    for cohort in cfg.cohorts:
        if cohort not in COHORT_TO_FILE:
            raise ValueError(f"Unknown cohort: {cohort}")
        rows = _load_jsonl(COHORT_TO_FILE[cohort])
        random.shuffle(rows)
        picked = rows[: cfg.n_cases]
        for idx, row in enumerate(picked):
            all_case_rows.append({
                "cohort": cohort,
                "case_idx": idx,
                "text": row["text"],
            })

    aggregate: Dict[Tuple[str, str, int], List[float]] = {}

    for row in all_case_rows:
        cohort = row["cohort"]
        case_idx = row["case_idx"]
        condition_case = row["text"]
        condition_name = COHORT_TO_CONDITION_NAME[cohort]

        for prompt_key in cfg.prompt_keys:
            prompt_template = CORRUPT_PROMPTS[prompt_key]
            corrupt = _prepare_corrupt_prompt(
                llm=llm,
                prompt_template=prompt_template,
                condition_name=condition_name,
                bhc_text=condition_case,
            )
            corrupted_prompt = corrupt["corrupted_prompt"]
            corrupted_tokens = corrupt["corrupted_tokens"]

            # Align clean-token index to the corrupted prompt timeline.
            # If clean is longer, shift corrupted token positions forward.
            # Example: clean=10, corrupt=8 -> offset=2, patch at token_idx+2.
            diff = len(clean_tokens) - len(corrupted_tokens)
            offset = diff if diff > 0 else 0

            clean_cache = _cache_clean_vectors(
                llm=llm,
                patch_target=cfg.patch_target,
                patch_token_from=patch_token_from,
                num_layers_model=num_layers_model,
                num_heads=num_heads,
                head_idx=cfg.head_idx,
                clean_prompt=clean_prompt,
            )

            corrupted_prob_target = _compute_corrupted_target_prob(
                llm=llm,
                corrupted_prompt=corrupted_prompt,
                target_gender_token=target_gender_token,
            )

            per_layer_mean, per_layer_max = _run_patch_sweep(
                llm=llm,
                patch_target=cfg.patch_target,
                clean_activations_cache=clean_cache,
                corrupted_prompt=corrupted_prompt,
                corrupted_tokens=corrupted_tokens,
                offset=offset,
                target_gender_token=target_gender_token,
                corrupted_prob_target=corrupted_prob_target,
                num_layers_model=num_layers_model,
                num_heads=num_heads,
                head_idx=cfg.head_idx,
                max_tokens=cfg.max_tokens,
            )

            per_case_path = os.path.join(
                cfg.output_dir,
                f"{cfg.patch_target}_{cohort}_{prompt_key}_case{case_idx:02d}.json",
            )
            _write_json(
                per_case_path,
                {
                    "model_name": MODEL_NAME,
                    "patch_target": cfg.patch_target,
                    "head_idx": cfg.head_idx if cfg.patch_target == "attn_head" else None,
                    "cohort": cohort,
                    "prompt_key": prompt_key,
                    "case_idx": case_idx,
                    "num_layers_model": num_layers_model,
                    "num_tokens_swept": len(corrupted_tokens) if cfg.max_tokens <= 0 else min(cfg.max_tokens, len(corrupted_tokens)),
                    "per_layer_mean": per_layer_mean,
                    "per_layer_max": per_layer_max,
                },
            )

            for layer_idx, score in enumerate(per_layer_mean):
                aggregate.setdefault((cohort, prompt_key, layer_idx), []).append(score)

            print(
                f"done cohort={cohort} case={case_idx} prompt={prompt_key} "
                f"target={cfg.patch_target}"
            )

    rows: List[Dict[str, Any]] = []
    for (cohort, prompt_key, layer_idx), scores in sorted(aggregate.items()):
        rows.append(
            {
                "cohort": cohort,
                "prompt_key": prompt_key,
                "layer": layer_idx,
                "mean_rewrite_score": float(sum(scores) / len(scores)),
                "max_rewrite_score": float(max(scores)),
                "num_cases": len(scores),
                "patch_target": cfg.patch_target,
            }
        )

    suffix = f"{cfg.patch_target}" if cfg.patch_target != "attn_head" else f"{cfg.patch_target}_h{cfg.head_idx}"
    aggregate_csv = os.path.join(cfg.output_dir, f"aggregate_{suffix}.csv")
    _write_aggregate_csv(aggregate_csv, rows)
    print(f"saved aggregate csv -> {aggregate_csv}")


if __name__ == "__main__":
    main()
