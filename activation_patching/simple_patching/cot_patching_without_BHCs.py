"""
CoT gender activation patching (Qwen 2.5 7B or OLMo 7B Instruct).

Uses <thinking>-tag CoT prompt variants (Types A and C, 10 each) with rewrite
scores from simple_patching_without_BHCs.py:
  - clean: "The patient is Male" (patch Male activations → flip female bias)
  - corrupt: CoT template (+ optional frozen mode)
  - score point (--score-point):
      forced_suffix       : append "Gender:" after user message (legacy / simple parity)
      after_thinking_stub : assistant prefix "<thinking></thinking>\\nGender:" then score
  - patch site: MLP down_proj (default) or residual stream

Prompt ids 1–10 = Type A, 11–20 = Type C (see cot_simple_prompts.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig

import simple_patching_without_BHCs as sp
from cot_simple_prompts import COT_SIMPLE_PROMPTS, cot_prompt_label
from cot_thinking_prompts import FROZEN_PROMPT

COT_CORRUPT_MODE_CHOICES = ("full", "frozen")
SCORE_POINT_CHOICES = ("forced_suffix", "after_thinking_stub")
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OLMO_MODEL_NAME = "allenai/OLMo-7B-0724-Instruct-hf"
# Fixed empty thinking block in the assistant turn, then Gender: (Option B).
AFTER_THINKING_STUB_SUFFIX = "<thinking></thinking>\nGender:"


def build_corrupt_prompt(
    llm: LanguageModel,
    template: str,
    condition_name: str,
    score_point: str = "forced_suffix",
) -> Tuple[str, torch.Tensor]:
    """Build corrupt prompt and fix the assistant suffix used for rewrite scoring."""
    if score_point not in SCORE_POINT_CHOICES:
        valid = ",".join(SCORE_POINT_CHOICES)
        raise ValueError(f"Unknown score_point={score_point!r}. Valid values: {valid}")

    body = template.replace("[CONDITION]", condition_name).replace("[CONDITION_NAME]", condition_name)
    messages = [
        {"role": "system", "content": "You are a helpful clinical assistant."},
        {"role": "user", "content": body},
    ]
    corrupted_text = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if score_point == "after_thinking_stub":
        corrupted_text += AFTER_THINKING_STUB_SUFFIX
    else:
        corrupted_text += "Gender:"
    corrupted_tokens = llm.tokenizer(corrupted_text, return_tensors="pt")["input_ids"][0]
    return corrupted_text, corrupted_tokens


def resolve_corrupt_template(prompt_id: int, corrupt_mode: str) -> str:
    if corrupt_mode == "frozen":
        return FROZEN_PROMPT
    if corrupt_mode == "full":
        return COT_SIMPLE_PROMPTS[prompt_id]
    valid = ",".join(COT_CORRUPT_MODE_CHOICES)
    raise ValueError(f"Unknown corrupt_mode={corrupt_mode!r}. Valid values: {valid}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CoT gender activation patching (Qwen/OLMo, <thinking> A/C prompts)"
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"HuggingFace model id (default: {DEFAULT_MODEL_NAME}; OLMo: {OLMO_MODEL_NAME})",
    )
    p.add_argument("--run-id", type=str, default="qwen_cot20_mlp_rewrite", help="Stable run folder for resume")
    p.add_argument("--resume", action="store_true", help="Skip completed units")
    p.add_argument("--output-dir", type=str, default="patching_results", help="Base output directory")
    p.add_argument(
        "--patch-target",
        type=str,
        default="mlp",
        choices=list(sp.PATCH_TARGET_CHOICES),
        help="Where to read/write activations during the rewrite sweep (default: mlp).",
    )
    p.add_argument(
        "--corrupt-mode",
        type=str,
        default="full",
        choices=COT_CORRUPT_MODE_CHOICES,
        help="full=CoT variant template; frozen=FROZEN_PROMPT (Gender-first, no <thinking>).",
    )
    p.add_argument(
        "--score-point",
        type=str,
        default="forced_suffix",
        choices=SCORE_POINT_CHOICES,
        help="forced_suffix=user+Gender:; after_thinking_stub=assistant <thinking></thinking> then Gender:.",
    )
    p.add_argument(
        "--cohorts",
        type=str,
        default=",".join(sp.DEFAULT_COHORTS),
        help="Comma-separated cohorts",
    )
    p.add_argument(
        "--prompt-ids",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
        help="Comma-separated CoT prompt ids (1–10=A, 11–20=C)",
    )
    p.add_argument(
        "--score-keys",
        type=str,
        default="rewrite_scores",
        help="Comma-separated score matrix keys to compute/use",
    )
    p.add_argument("--max-tokens", type=int, default=0, help="0 = full token sweep")
    p.add_argument("--layer-start", type=int, default=0, help="First layer index (inclusive)")
    p.add_argument("--layer-end", type=int, default=9999, help="Last layer index (exclusive)")
    p.add_argument("--layer-step", type=int, default=1, help="Layer step in sweep (memory tuning)")
    p.add_argument("--top-k", type=int, default=30, help="Top-k used for top-k mean aggregation")
    p.add_argument("--trim-frac", type=float, default=0.10, help="Trim fraction for trimmed mean aggregation")
    p.add_argument("--save-heatmaps", action="store_true", help="Save token×layer heatmap per unit")
    p.add_argument("--save-layer-plots", action="store_true", help="Save per-layer plots and top-layer summaries")
    p.add_argument("--plot-format", type=str, default="pdf", choices=["pdf", "png"])
    p.add_argument(
        "--heatmap-mode",
        type=str,
        default="single",
        choices=["single", "full_suite"],
        help="single: one all-layer heatmap; full_suite: split layers + token windows + overview",
    )
    p.add_argument("--heatmap-token-window", type=int, default=180, help="Token columns per heatmap tile (0 = no split)")
    p.add_argument(
        "--heatmap-overview-bin-size",
        type=int,
        default=10,
        help="Bin size for compact overview heatmap (1 disables binning)",
    )
    p.add_argument("--rebuild-plots-only", action="store_true", help="Regenerate plots from saved artifacts only")
    p.add_argument("--dry-run", action="store_true", help="Only validate config, work list, and progress (no model load)")
    return p.parse_args()


def get_run_dir(args: argparse.Namespace) -> Path:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / args.run_id


def build_work_list(args: argparse.Namespace, run_dir: Path) -> List[Tuple[str, int]]:
    _ = run_dir
    raw_cohorts = [x.strip() for x in args.cohorts.split(",") if x.strip()]
    cohorts: List[str] = []
    for raw in raw_cohorts:
        normalized = raw.lower().replace("-", "_")
        cohorts.append(sp.COHORT_ALIASES.get(normalized, normalized))

    prompt_ids = [int(x.strip()) for x in args.prompt_ids.split(",") if x.strip()]
    invalid_prompt_ids = [pid for pid in prompt_ids if pid not in COT_SIMPLE_PROMPTS]
    if invalid_prompt_ids:
        valid_ids = ",".join(str(x) for x in sorted(COT_SIMPLE_PROMPTS.keys()))
        invalid_ids = ",".join(str(x) for x in sorted(set(invalid_prompt_ids)))
        raise ValueError(f"Unknown CoT prompt id(s): {invalid_ids}. Valid ids: {valid_ids}")

    work: List[Tuple[str, int]] = []
    for cohort in cohorts:
        if cohort not in sp.COHORT_TO_CONDITION_NAME:
            print(f"Warning: unknown cohort {cohort}, skipping", file=sys.stderr)
            continue
        for prompt_id in prompt_ids:
            work.append((cohort, prompt_id))
    return work


def run_patching(args: argparse.Namespace, run_dir: Path) -> None:
    selected_score_keys: Tuple[str, ...] = args.selected_score_keys
    work_list = build_work_list(args, run_dir)
    progress = sp.load_progress(run_dir)
    completed_set = set(progress.get("completed", []))
    if args.resume:
        work_list = [w for w in work_list if sp.unit_key(w[0], w[1]) not in completed_set]
        print(f"Resume: {len(completed_set)} completed, {len(work_list)} remaining", flush=True)

    if not work_list:
        print("No work units to run.", flush=True)
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
        key = sp.unit_key(cohort, prompt_id)
        try:
            gender = sp.COHORT_TO_PATCH_GENDER.get(cohort, "Male")
            condition_name = sp.COHORT_TO_CONDITION_NAME.get(cohort, cohort)
            corrupt_template = resolve_corrupt_template(prompt_id, args.corrupt_mode)
            clean_text, _, patch_token_from = sp.build_clean_prompt(llm, gender)
            corrupted_text, corrupted_tokens = build_corrupt_prompt(
                llm, corrupt_template, condition_name, score_point=args.score_point
            )
            target_id = int(sp._validated_gender_token_ids(llm, gender)[-1].item())

            sweep = sp.run_patch_sweep(
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
                selected_score_keys=selected_score_keys,
                patch_target=args.patch_target,
                step=args.layer_step,
            )
            shape_key = next((k for k in selected_score_keys if k in sweep), None)
            if shape_key is None:
                raise RuntimeError("No selected score matrices found in sweep output.")
            score_ref = sweep[shape_key]
            n_l, n_t = score_ref.shape
            token_labels = [
                f"{llm.tokenizer.decode(corrupted_tokens[i])}_{i}" for i in range(n_t)
            ]
            layer_labels = list(range(layer_start, layer_start + n_l))
            metadata = {
                "cohort": cohort,
                "prompt_id": prompt_id,
                "prompt_set": "cot",
                "cot_prompt_label": cot_prompt_label(prompt_id),
                "cot_corrupt_mode": args.corrupt_mode,
                "score_point": args.score_point,
                "model_name": args.model_name,
                "patch_gender": gender,
                "patch_target": args.patch_target,
                "condition_name": condition_name,
                "corrupted_prob": sweep.get("corrupted_prob", 0.0),
                "corrupted_logprob": sweep.get("corrupted_logprob", float("-inf")),
                "num_layers": num_layers,
                "num_tokens": n_t,
                "score_keys": list(selected_score_keys),
            }
            sp.save_unit_artifact(
                run_dir,
                cohort,
                prompt_id,
                {k: sweep[k] for k in selected_score_keys if k in sweep},
                token_labels,
                layer_labels,
                metadata,
            )

            if args.save_heatmaps and "rewrite_scores" in sweep:
                plot_dir = run_dir / "heatmaps"
                plot_dir.mkdir(parents=True, exist_ok=True)
                try:
                    sp.plot_heatmap(
                        sweep["rewrite_scores"],
                        token_labels,
                        layer_labels,
                        f"{cohort} prompt{prompt_id} rewrite_scores",
                        str(plot_dir / f"{cohort}_prompt{prompt_id}_rewrite_scores"),
                        args.plot_format,
                        mode=args.heatmap_mode,
                        token_window=args.heatmap_token_window,
                        overview_bin_size=args.heatmap_overview_bin_size,
                    )
                except Exception as e:
                    print(f"Plot warning (heatmap) {key}: {e}", file=sys.stderr, flush=True)

            if args.save_layer_plots:
                plot_dir = run_dir / "layer_plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                for score_key in selected_score_keys:
                    if score_key not in sweep:
                        continue
                    stats = sp.layer_aggregates(
                        sweep[score_key], top_k=args.top_k, trim_frac=args.trim_frac
                    )
                    per_prompt_dir = plot_dir / "per_prompt" / score_key
                    per_prompt_dir.mkdir(parents=True, exist_ok=True)
                    base = per_prompt_dir / f"{cohort}_prompt{prompt_id}_{score_key}"
                    try:
                        sp.plot_layer_curves(
                            stats,
                            layer_labels,
                            f"{cohort} prompt{prompt_id} {score_key}",
                            str(base),
                            args.plot_format,
                            top_k=args.top_k,
                        )
                    except Exception as e:
                        print(f"Plot warning (layer curves) {key} {score_key}: {e}", file=sys.stderr, flush=True)

            sp.mark_completed(run_dir, key, progress)
            progress = sp.load_progress(run_dir)
            print(f"Done {key}", flush=True)
        except Exception as e:
            sp.mark_failed(run_dir, key, str(e), progress)
            progress = sp.load_progress(run_dir)
            print(f"Failed {key}: {e}", file=sys.stderr, flush=True)
            raise

    sp.build_aggregates(run_dir, args)


def main() -> None:
    args = parse_args()
    args.selected_score_keys = sp._resolve_score_keys(args.score_keys)
    run_dir = get_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_plots_only:
        sp.rebuild_plots_only(args, run_dir)
        return

    progress = sp.load_progress(run_dir)
    if not progress.get("config_hash"):
        progress["config_hash"] = sp._config_hash(args)
        progress["model_name"] = args.model_name
        sp.save_progress(run_dir, progress)

    if args.dry_run:
        work_list = build_work_list(args, run_dir)
        print(f"Dry run: run_dir={run_dir}, work_units={len(work_list)}", flush=True)
        print(f"Model: {args.model_name}", flush=True)
        print(f"CoT corrupt mode: {args.corrupt_mode}", flush=True)
        print(f"Score point: {args.score_point}", flush=True)
        print(f"Patch target: {args.patch_target}", flush=True)
        print(f"Selected score keys: {','.join(args.selected_score_keys)}", flush=True)
        print(f"Progress: {sp.load_progress(run_dir)}", flush=True)
        fake = np.random.randn(4, 10).astype(np.float32)
        stats = sp.layer_aggregates(fake, top_k=args.top_k, trim_frac=args.trim_frac)
        assert all(v.shape == (4,) for v in stats.values()), "layer_aggregates shape"
        print("Dry run OK.", flush=True)
        return

    run_patching(args, run_dir)


if __name__ == "__main__":
    main()
