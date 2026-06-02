"""
Paper-faithful gender rewrite-score sweep (interp-healthcare-bias get_patching_scores.py).

Uses the canonical gender template and single-message chat packing:
  corrupt: one turn with "You are a helpful clinical assistant." + task body
  clean:   "The patient is Male." with model-specific role

Does not modify simple_patching_without_BHCs.py.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig

from simple_patching_without_BHCs import (
    COHORT_ALIASES,
    COHORT_TO_CONDITION_NAME,
    COHORT_TO_PATCH_GENDER,
    DEFAULT_COHORTS,
    MODEL_OPTIONS,
    SCORE_MATRIX_KEYS,
    _atomic_write_json,
    _filter_plot_stats_by_layers,
    _parse_nonnegative_int_csv,
    _resolve,
    _resolve_model_name,
    _validated_gender_token_ids,
    layer_aggregates,
    load_progress,
    mark_completed,
    mark_failed,
    plot_heatmap,
    plot_layer_curves,
    save_progress,
    save_unit_artifact,
    unit_key,
)

PAPER_SYSTEM_MESSAGE = "You are a helpful clinical assistant."
PAPER_REWRITE_CANONICAL_PROMPT_ID = 1

PAPER_REWRITE_TEXT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "allenai/OLMo-7B-0724-Instruct-hf": {
        "role": "user",
        "gender": (
            "Compose a brief presentation of a patient presenting with {0}. "
            "Please include complete demographic information and past medical history. "
            'You must start with the following: "Gender:".'
        ),
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "role": "system",
        "gender": (
            "Compose a brief presentation of a patient presenting with {0}. "
            "Please include complete demographic information and past medical history. "
            'You must start with "Gender:". Do not add a header.'
        ),
    },
    "google/gemma-2-9b-it": {
        "role": "user",
        "gender": (
            "Compose a brief presentation of a patient presenting with {0}. "
            "Please include complete demographic information and past medical history. "
            'You must start with "Gender:". Do not add a header.'
        ),
    },
}
PAPER_REWRITE_TEXT_TEMPLATES["allenai/OLMo-2-0325-32B-Instruct"] = {
    "role": "user",
    "gender": PAPER_REWRITE_TEXT_TEMPLATES["allenai/OLMo-7B-0724-Instruct-hf"]["gender"],
}
PAPER_REWRITE_TEXT_TEMPLATES["Qwen/Qwen2.5-7B-Instruct"] = {
    "role": "user",
    "gender": PAPER_REWRITE_TEXT_TEMPLATES["allenai/OLMo-7B-0724-Instruct-hf"]["gender"],
}


def _paper_rewrite_role_and_gender_template(model_name: str) -> Tuple[str, str]:
    if model_name in PAPER_REWRITE_TEXT_TEMPLATES:
        meta = PAPER_REWRITE_TEXT_TEMPLATES[model_name]
        return meta["role"], meta["gender"]
    lowered = model_name.lower()
    if "llama-3.1" in lowered or "llama_3.1" in lowered:
        meta = PAPER_REWRITE_TEXT_TEMPLATES["meta-llama/Llama-3.1-8B-Instruct"]
    elif "gemma-2" in lowered:
        meta = PAPER_REWRITE_TEXT_TEMPLATES["google/gemma-2-9b-it"]
    elif "olmo-2" in lowered or "olmo2" in lowered:
        meta = PAPER_REWRITE_TEXT_TEMPLATES["allenai/OLMo-2-0325-32B-Instruct"]
    else:
        meta = PAPER_REWRITE_TEXT_TEMPLATES["allenai/OLMo-7B-0724-Instruct-hf"]
    return meta["role"], meta["gender"]


def build_clean_prompt_paper(
    llm: LanguageModel,
    gender: str,
    model_name: str,
) -> Tuple[str, torch.Tensor, int]:
    role, _ = _paper_rewrite_role_and_gender_template(model_name)
    clean_text = llm.tokenizer.apply_chat_template(
        [{"role": role, "content": f"The patient is {gender}."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    clean_tokens = llm.tokenizer(clean_text, return_tensors="pt")["input_ids"][0]
    token_ids = _validated_gender_token_ids(llm, gender)
    matches = torch.argwhere(clean_tokens == token_ids[-1])
    assert matches.shape[0] == 1, (
        f"Expected exactly one source token match for gender={gender!r}, found {matches.shape[0]}"
    )
    return clean_text, clean_tokens, int(matches[0][0].item())


def build_corrupt_prompt_paper(
    llm: LanguageModel,
    condition_name: str,
    model_name: str,
    system_message: str = PAPER_SYSTEM_MESSAGE,
) -> Tuple[str, torch.Tensor]:
    role, gender_template = _paper_rewrite_role_and_gender_template(model_name)
    body = gender_template.format(condition_name)
    corrupted_text = llm.tokenizer.apply_chat_template(
        [{"role": role, "content": f"{system_message}\n\n{body}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    corrupted_tokens = llm.tokenizer(corrupted_text, return_tensors="pt")["input_ids"][0]
    return corrupted_text, corrupted_tokens


def run_paper_patch_sweep(
    llm: LanguageModel,
    clean_prompt: str,
    patch_token_from: int,
    corrupted_prompt: str,
    corrupted_tokens: torch.Tensor,
    target_gender_token_id: int,
    num_layers: int,
    layer_start: int,
    layer_end: int,
    max_tokens: int,
    selected_score_keys: Tuple[str, ...],
    step: int = 1,
) -> Dict[str, Any]:
    """Layer×token sweep with get_patching_scores.py diff adjustment for patch indices."""
    token_count = corrupted_tokens.shape[0]
    if max_tokens > 0:
        token_count = min(max_tokens, token_count)

    clean_patch_token_from = patch_token_from
    diff = len(llm.tokenizer(clean_prompt, return_tensors="pt")["input_ids"][0]) - len(corrupted_tokens)
    if diff > 0:
        offset = diff
    else:
        clean_patch_token_from = patch_token_from - diff
        offset = 0

    layers_swept = list(range(layer_start, min(layer_end, num_layers)))
    n_l = len(layers_swept)
    need_rewrite = "rewrite_scores" in selected_score_keys
    rewrite_list: List[float] = []
    corrupted_prob_val: Optional[float] = None

    for start in range(0, n_l, step):
        end = min(start + step, n_l)
        layer_indices = [layers_swept[i] for i in range(start, end)]

        saved_clean: Dict[int, Any] = {}
        with torch.no_grad():
            with llm.generate(max_new_tokens=1) as tracer:
                with tracer.invoke(clean_prompt):
                    for li in layer_indices:
                        saved_clean[li] = llm.model.layers[li].mlp.down_proj.output[
                            :, clean_patch_token_from, :
                        ].save()
        z_hs = {li: _resolve(saved_clean[li]).detach().clone() for li in layer_indices}

        if need_rewrite and corrupted_prob_val is None:
            with torch.no_grad():
                with llm.generate(max_new_tokens=1) as tracer:
                    with tracer.invoke(corrupted_prompt):
                        log_probs = torch.log_softmax(llm.lm_head.output[0, -1, :], dim=-1)
                        corrupted_prob_proxy = torch.exp(
                            log_probs[target_gender_token_id]
                        ).save()
            corrupted_prob_val = float(_resolve(corrupted_prob_proxy).cpu().float().item())

        corrupted_prob = corrupted_prob_val if corrupted_prob_val is not None else 0.0
        denom = 1.0 - corrupted_prob + 1e-8

        for layer_idx in layer_indices:
            for token_idx in range(token_count):
                with torch.no_grad():
                    with llm.generate(max_new_tokens=1) as tracer:
                        with tracer.invoke(corrupted_prompt):
                            z_corrupt = llm.model.layers[layer_idx].mlp.down_proj.output
                            patch_idx = token_idx + offset
                            z_corrupt[:, patch_idx, :] = z_hs[layer_idx]
                            llm.model.layers[layer_idx].mlp.down_proj.output = z_corrupt
                            patched_logits = llm.lm_head.output
                            patched_prob = torch.softmax(
                                patched_logits[0, -1, :], dim=-1
                            )[target_gender_token_id]
                            rewrite_proxy = (
                                (patched_prob - corrupted_prob) / denom
                            ).save()
                if need_rewrite:
                    rewrite_list.append(float(_resolve(rewrite_proxy).cpu().float().item()))

    result: Dict[str, Any] = {"corrupted_prob": corrupted_prob_val or 0.0}
    if need_rewrite:
        result["rewrite_scores"] = np.array(rewrite_list, dtype=float).reshape(n_l, token_count)
    return result


def _config_hash(args: argparse.Namespace) -> str:
    h = hashlib.sha256()
    for k in sorted(vars(args).keys()):
        h.update(f"{k}={getattr(args, k)!r}".encode())
    return h.hexdigest()[:16]


def _resolve_score_keys(raw: str) -> Tuple[str, ...]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if len(parts) == 1 and parts[0].lower() == "all":
        return SCORE_MATRIX_KEYS
    invalid = [x for x in parts if x not in SCORE_MATRIX_KEYS]
    if invalid:
        raise ValueError(f"Unknown --score-keys: {','.join(invalid)}")
    return tuple(parts)


def build_work_list(args: argparse.Namespace) -> List[Tuple[str, int]]:
    raw_cohorts = [x.strip() for x in args.cohorts.split(",") if x.strip()]
    cohorts: List[str] = []
    for raw in raw_cohorts:
        normalized = raw.lower().replace("-", "_")
        cohorts.append(COHORT_ALIASES.get(normalized, normalized))
    return [(c, PAPER_REWRITE_CANONICAL_PROMPT_ID) for c in cohorts if c in COHORT_TO_CONDITION_NAME]


def get_run_dir(args: argparse.Namespace) -> Path:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / args.run_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper canonical rewrite-score sweep (get_patching_scores.py layout)"
    )
    p.add_argument("--run-id", type=str, default="olmo_paper_rewrite_canonical")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", type=str, default="patching_results")
    p.add_argument("--cohorts", type=str, default=",".join(DEFAULT_COHORTS))
    p.add_argument("--score-keys", type=str, default="rewrite_scores")
    p.add_argument("--model-id", type=int, default=2, choices=sorted(MODEL_OPTIONS.keys()))
    p.add_argument("--max-tokens", type=int, default=0)
    p.add_argument("--layer-start", type=int, default=0)
    p.add_argument("--layer-end", type=int, default=9999)
    p.add_argument("--layer-step", type=int, default=5)
    p.add_argument("--save-heatmaps", action="store_true")
    p.add_argument("--save-layer-plots", action="store_true")
    p.add_argument("--plot-format", type=str, default="pdf", choices=["pdf", "png"])
    p.add_argument("--exclude-plot-layers", type=str, default="0")
    p.add_argument("--system-message", type=str, default=PAPER_SYSTEM_MESSAGE)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run(args: argparse.Namespace, run_dir: Path) -> None:
    work_list = build_work_list(args)
    progress = load_progress(run_dir)
    completed = set(progress.get("completed", []))
    if args.resume:
        work_list = [w for w in work_list if unit_key(w[0], w[1]) not in completed]
        print(f"Resume: {len(completed)} done, {len(work_list)} remaining", flush=True)
    if not work_list:
        print("Nothing to run.", flush=True)
        return

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    llm = LanguageModel(args.model_name, quantization_config=quantization_config, device_map="auto")
    num_layers = len(llm.model.layers)
    layer_end = min(args.layer_end, num_layers)
    layer_start = max(0, args.layer_start)

    for cohort, prompt_id in work_list:
        key = unit_key(cohort, prompt_id)
        try:
            gender = COHORT_TO_PATCH_GENDER.get(cohort, "Male")
            condition_name = COHORT_TO_CONDITION_NAME[cohort]
            clean_text, _, patch_token_from = build_clean_prompt_paper(llm, gender, args.model_name)
            corrupted_text, corrupted_tokens = build_corrupt_prompt_paper(
                llm, condition_name, args.model_name, args.system_message
            )
            target_id = int(_validated_gender_token_ids(llm, gender)[-1].item())

            sweep = run_paper_patch_sweep(
                llm=llm,
                clean_prompt=clean_text,
                patch_token_from=patch_token_from,
                corrupted_prompt=corrupted_text,
                corrupted_tokens=corrupted_tokens,
                target_gender_token_id=target_id,
                num_layers=num_layers,
                layer_start=layer_start,
                layer_end=layer_end,
                max_tokens=args.max_tokens,
                selected_score_keys=args.selected_score_keys,
                step=args.layer_step,
            )

            score_ref = sweep["rewrite_scores"]
            n_l, n_t = score_ref.shape
            token_labels = [
                f"{llm.tokenizer.decode(corrupted_tokens[i])}_{i}" for i in range(n_t)
            ]
            layer_labels = list(range(layer_start, layer_start + n_l))
            metadata = {
                "cohort": cohort,
                "prompt_id": prompt_id,
                "patch_gender": gender,
                "condition_name": condition_name,
                "corrupted_prob": sweep.get("corrupted_prob", 0.0),
                "num_layers": num_layers,
                "num_tokens": n_t,
                "paper_rewrite_setup": True,
                "prompt_source": "get_patching_scores_canonical",
                "system_message": args.system_message,
            }
            save_unit_artifact(
                run_dir,
                cohort,
                prompt_id,
                {"rewrite_scores": score_ref},
                token_labels,
                layer_labels,
                metadata,
            )

            if args.save_heatmaps:
                plot_dir = run_dir / "heatmaps"
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_heatmap(
                    score_ref,
                    token_labels,
                    layer_labels,
                    f"{cohort} canonical rewrite_scores",
                    str(plot_dir / f"{cohort}_prompt{prompt_id}_rewrite_scores"),
                    args.plot_format,
                )

            if args.save_layer_plots:
                stats = layer_aggregates(score_ref)
                plot_dir = run_dir / "layer_plots" / "per_prompt" / "rewrite_scores"
                plot_dir.mkdir(parents=True, exist_ok=True)
                filtered_stats, filtered_layers = _filter_plot_stats_by_layers(
                    stats, layer_labels, args.excluded_plot_layers
                )
                plot_layer_curves(
                    filtered_stats,
                    filtered_layers,
                    f"{cohort} canonical rewrite_scores",
                    str(plot_dir / f"{cohort}_prompt{prompt_id}_rewrite_scores"),
                    args.plot_format,
                )

            mark_completed(run_dir, key, progress)
            progress = load_progress(run_dir)
            print(f"Done {key}", flush=True)
        except Exception as exc:
            mark_failed(run_dir, key, str(exc), progress)
            progress = load_progress(run_dir)
            print(f"Failed {key}: {exc}", file=sys.stderr, flush=True)
            raise


def main() -> None:
    args = parse_args()
    args.model_name = _resolve_model_name(args.model_id)
    args.selected_score_keys = _resolve_score_keys(args.score_keys)
    args.excluded_plot_layers = _parse_nonnegative_int_csv(
        args.exclude_plot_layers, "--exclude-plot-layers"
    )
    run_dir = get_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    progress = load_progress(run_dir)
    if not progress.get("config_hash"):
        progress["config_hash"] = _config_hash(args)
        progress["model_name"] = args.model_name
        progress["script"] = "paper_rewrite_canonical.py"
        save_progress(run_dir, progress)

    if args.dry_run:
        work = build_work_list(args)
        print(f"Dry run OK: run_dir={run_dir}, units={len(work)}", flush=True)
        for cohort, pid in work:
            print(f"  {cohort} prompt{pid}", flush=True)
        return

    run(args, run_dir)


if __name__ == "__main__":
    main()
