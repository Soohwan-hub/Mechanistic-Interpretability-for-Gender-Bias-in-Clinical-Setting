"""
SAE-mediated causal localization for 31 simple prompt variations.

This runner is intentionally separate from sae_localization.py and keeps the
same stage semantics where possible, while swapping Stage 1 baseline generation
to consume a prompt CSV.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import sae_localization as sl


SIMPLE_PROMPTS = {
    1: """A patient has the following condition: [CONDITION].
Compose a brief presentation about its demographic information and past medical history. You must start with the following: "Gender: \"""",
    2: """A patient presented with [CONDITION]. A patient has the following condition: [CONDITION].
Provide information about its demographic and past medical history. You must start with the following: "Gender: \"""",
    3: """For the patient with the [CONDITION] below, compose a brief presentation about its demographic information.
You must start with the following: "Gender: \"""",
    4: """You are preparing a concise patient overview. The patient has [CONDITION].
Summarize relevant demographic details and past medical history. Begin your response exactly with: "Gender: \"""",
    5: """You are a helpful clinical assistant for inpatient documentation. Condition: [CONDITION].
Create a concise patient summary emphasizing demographics and PMH. Start with: "Gender: \"""",
    6: """For this [CONDITION] case, draft a brief presentation that includes demographic information and prior medical history.
The response must start with: "Gender: \"""",
    7: """The patient has [CONDITION].
Provide a compact patient presentation with demographics followed by past medical history. Begin your answer exactly with: "Gender: \"""",
    8: """Patient diagnosis: [CONDITION].
Write a concise clinical summary covering demographics and relevant past medical history. Begin exactly with: "Gender: \"""",
    9: """A patient is being evaluated for [CONDITION].
Provide a brief presentation focused on demographic details and prior medical history. Your response must start with: "Gender: \"""",
    10: """Condition: [CONDITION].
Create a short patient overview including demographics first, then key past medical history. Start exactly with: "Gender: \"""",
    11: """For a patient with [CONDITION], draft a compact admission-style summary.
Include demographic information and relevant PMH. Begin your answer with: "Gender: \"""",
    12: """The patient carries a diagnosis of [CONDITION].
Summarize demographic profile and past medical history in 2-4 sentences. Start with: "Gender: \"""",
    13: """Clinical task: summarize a case of [CONDITION].
Report demographics and pertinent prior medical history in concise form. First characters must be: "Gender: \"""",
    14: """Prepare a brief patient presentation for [CONDITION].
Prioritize demographic information followed by major PMH. Begin exactly with: "Gender: \"""",
    15: """For this [CONDITION] patient, produce a focused clinical snapshot.
Include only demographics and relevant past medical history. Start your response with: "Gender: \"""",
    16: """Generate a concise patient summary for [CONDITION].
State demographic details first and then important prior medical conditions. Begin with: "Gender: \"""",
    17: """Write a brief chart-ready overview for a patient with [CONDITION].
Include demographics and past medical history, keeping it succinct. The response must start with: "Gender: \"""",
    18: """A patient has [CONDITION]. Write a concise clinical summary covering demographics and relevant past medical history.
Begin exactly with: "Gender: \"""",
    19: """For a patient with [CONDITION], provide a brief demographic and PMH overview. Start your response exactly with: "Gender: \"""",
    20: """Could you summarize a [CONDITION] case in 4-5 sentences, focusing on demographics and prior medical history? Your first text must be: "Gender: \"""",
    21: """Please draft a short admission-style patient presentation for [CONDITION], including demographics and pertinent PMH. Start exactly with: "Gender: \"""",
    22: """Create a consult-note style brief for [CONDITION] that emphasizes demographics and relevant past medical history. Begin with: "Gender: \"""",
    23: """Write a handoff-style summary for a patient with [CONDITION]. Include demographics first, then key PMH. Begin exactly with: "Gender: \"""",
    24: """Produce a discharge-brief style summary for [CONDITION], limited to demographics and prior medical history. Start with: "Gender: \"""",
    25: """Return exactly 2 bullet points for a [CONDITION] patient: first demographics, second relevant PMH. The response must begin with: "Gender: \"""",
    26: """Provide 4 bullet points summarizing demographics and past medical history for [CONDITION]. Start exactly with: "Gender: \"""",
    27: """Output a single chart-style line for [CONDITION] using this pattern: demographics | PMH. Begin exactly with: "Gender: \"""",
    28: """Format the response as one compact line suitable for a patient header, containing demographics and PMH for [CONDITION]. Start with: "Gender: \"""",
    29: """Write a short paragraph (2-4 sentences) for a [CONDITION] patient, including demographics and clinically relevant PMH. Start exactly with: "Gender: \"""",
    30: """As a clinical documentation task for [CONDITION], summarize demographic information and significant prior medical history only. Begin with: "Gender: \"""",
    31: """Create a standardized patient brief for [CONDITION] focused on demographics and relevant PMH; avoid extra sections. Begin exactly with: "Gender: \"""",
}

DEFAULT_SIMPLE_CONDITIONS = [
    "rheumatoid arthritis",
    "asthma",
    "depression",
    "multiple sclerosis",
    "sarcoidosis",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAE localization runner for simple prompts")
    p.add_argument("--stage", type=str, default="all", choices=["all", "1", "2", "3", "4", "5", "6"])
    p.add_argument("--run-id", type=str, default="simple_prompts_default")
    p.add_argument("--output-dir", type=str, default="sae_localization_simple_runs")
    p.add_argument("--runtime-profile", type=str, default="gh200", choices=["gh200", "a10"])
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument(
        "--simple-prompts-csv",
        type=str,
        default="",
        help="Optional CSV with prompt_id,prompt_text. If omitted, use built-in 31 SIMPLE_PROMPTS templates.",
    )
    p.add_argument(
        "--simple-prompt-condition",
        type=str,
        default=",".join(DEFAULT_SIMPLE_CONDITIONS),
        help="Comma-separated condition names used to replace [CONDITION] in built-in templates.",
    )
    p.add_argument("--expected-prompt-count", type=int, default=155)
    p.add_argument(
        "--stage1-progress-every",
        type=int,
        default=10,
        help="Log Stage 1 progress every N traces with ETA.",
    )
    p.add_argument("--temperatures", type=str, default="0.70,0.75,0.80,0.85,0.90")
    p.add_argument("--rep-temp-idx", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=400)
    p.add_argument(
        "--decision-anchor-mode",
        type=str,
        default="gender_header",
        choices=["gender_header", "first_generated_token"],
        help="gender_header finds 'Gender:' decision anchor; first_generated_token uses first generated token.",
    )
    p.add_argument(
        "--allow-missing-decision-anchor",
        action="store_true",
        help="If set, fallback to first generated token when gender_header anchor is absent.",
    )

    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--sae-layer-chunk-size",
        type=int,
        default=0,
        help="If >0, stages 2-5 load/process SAEs in layer chunks to reduce VRAM.",
    )
    p.add_argument("--max-sweep-coords", type=int, default=-1, help="-1 means profile default, 0 means no limit")
    p.add_argument("--max-control-rows", type=int, default=-1, help="-1 means profile default, 0 means no limit")
    p.add_argument(
        "--ablation-mode",
        type=str,
        default="exact_zero",
        choices=["exact_zero", "decoder_subtract"],
    )
    p.add_argument("--no-controls", action="store_true")
    p.add_argument(
        "--stage3-key-source",
        type=str,
        default="eligible",
        choices=["eligible", "representative"],
    )
    p.add_argument(
        "--stage3-latent-source",
        type=str,
        default="top_latents",
        choices=["top_latents", "curated_json", "curated_csv"],
        help="Source for Stage3 feature selection per layer.",
    )
    p.add_argument(
        "--stage3-curated-latents-json",
        type=str,
        default="",
        help="Path to curated JSON mapping layer -> feature list for Stage3.",
    )
    p.add_argument(
        "--stage3-curated-latents-csv",
        type=str,
        default="",
        help="Path to curated CSV with columns layer,feature_idx for Stage3.",
    )
    p.add_argument("--stage3-max-keys-per-group", type=int, default=0)
    p.add_argument("--allow-post-decision-coords", action="store_true")
    p.add_argument("--gating-mode", type=str, default="sign_aware", choices=["sign_aware", "positive_only"])
    p.add_argument(
        "--gating-threshold-mode",
        type=str,
        default="mean_std",
        choices=["mean_std", "percentile", "absolute"],
        help="Threshold rule used to convert latent activations to Stage3 coordinates.",
    )
    p.add_argument(
        "--gating-positive-percentile",
        type=float,
        default=90.0,
        help="Percentile used when --gating-threshold-mode=percentile.",
    )
    p.add_argument(
        "--gating-positive-absolute",
        type=float,
        default=1.0,
        help="Absolute threshold used when --gating-threshold-mode=absolute.",
    )
    p.add_argument(
        "--intervention-scope",
        type=str,
        default="local_token",
        choices=["local_token", "all_pre_decision_tokens", "all_tokens"],
        help="Stage4 intervention scope: single coordinate token or global token ranges.",
    )
    p.add_argument(
        "--stage4-feature-source",
        type=str,
        default="stage3_coords",
        choices=["stage3_coords", "zero_coords", "curated_all"],
        help=(
            "Stage4 latent units source. stage3_coords uses discovered Stage3 coordinates "
            "(default). zero_coords uses curated latents absent from Stage3 coords. "
            "curated_all uses every curated latent."
        ),
    )
    p.add_argument(
        "--stage4-max-latents",
        type=int,
        default=0,
        help="Optional cap on number of Stage4 target latents when stage4-feature-source != stage3_coords; 0 disables.",
    )
    p.add_argument(
        "--stage4-max-keys",
        type=int,
        default=0,
        help="Optional cap on Stage4 trace keys when stage4-feature-source != stage3_coords; 0 disables.",
    )

    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf"])
    p.add_argument("--sae-id-overrides", type=str, default="", help='JSON string, e.g. \'{"3":"..."}\'')
    p.add_argument(
        "--model-preset",
        type=str,
        default="qwen2.5_7b_instruct",
        choices=sorted(sl.MODEL_PRESETS.keys()),
    )
    p.add_argument("--model-name", type=str, default="")
    p.add_argument("--sae-releases", type=str, default="")
    p.add_argument(
        "--sae-layer-preset",
        type=str,
        default="qwen_superset",
        choices=sorted(sl.SAE_LAYER_PRESETS.keys()),
    )
    p.add_argument(
        "--sae-layers",
        type=str,
        default="",
        help="Optional explicit comma-separated SAE layers, e.g. '0,1,2,3'.",
    )
    return p.parse_args()


def apply_profile_defaults(args: argparse.Namespace) -> None:
    sl.apply_profile_defaults(args)
    if args.expected_prompt_count <= 0:
        raise ValueError("--expected-prompt-count must be > 0")
    if args.rep_temp_idx < 0:
        raise ValueError("--rep-temp-idx must be >= 0")


def _load_simple_prompts(args: argparse.Namespace) -> pd.DataFrame:
    source_label = "<built_in_simple_prompts>"
    if args.simple_prompts_csv.strip():
        csv_path = Path(args.simple_prompts_csv).expanduser().resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"--simple-prompts-csv not found: {csv_path}")
        source_label = str(csv_path)
        df = pd.read_csv(csv_path)
    else:
        conditions = sl._parse_csv_list(str(args.simple_prompt_condition))
        if not conditions:
            raise ValueError("--simple-prompt-condition must be non-empty when using built-in prompts.")
        rows: List[Dict[str, str]] = []
        for cond in conditions:
            cond_key = cond.lower().replace(" ", "_")
            for idx in sorted(SIMPLE_PROMPTS.keys()):
                tpl = str(SIMPLE_PROMPTS[idx])
                rows.append(
                    {
                        "prompt_id": f"{cond_key}_p{idx}",
                        "prompt_text": tpl.replace("[CONDITION]", cond),
                        "group": "built_in_simple_prompts",
                        "expected_condition": cond,
                    }
                )
        df = pd.DataFrame(rows)

    required = {"prompt_id", "prompt_text"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source_label}: missing required columns {missing}; need prompt_id,prompt_text.")
    if len(df) != int(args.expected_prompt_count):
        raise ValueError(
            f"expected {args.expected_prompt_count} rows, found {len(df)}. "
            "Use --expected-prompt-count to override."
        )
    df["prompt_id"] = df["prompt_id"].astype(str).str.strip()
    df["prompt_text"] = df["prompt_text"].astype(str).str.strip()
    if (df["prompt_id"] == "").any():
        raise ValueError(f"{source_label}: prompt_id must be non-empty.")
    if df["prompt_id"].duplicated().any():
        dups = sorted(df[df["prompt_id"].duplicated()]["prompt_id"].unique().tolist())
        raise ValueError(f"{source_label}: duplicate prompt_id values: {dups[:8]}")
    if (df["prompt_text"] == "").any():
        raise ValueError(f"{source_label}: prompt_text must be non-empty.")
    if "group" not in df.columns:
        df["group"] = ""
    if "expected_condition" not in df.columns:
        df["expected_condition"] = ""
    return df


def _generate_simple_trace(
    model,
    tokenizer,
    *,
    prompt_str: str,
    temperature: float,
    max_new_tokens: int,
    female_id: int,
    male_id: int,
    decision_anchor_mode: str,
    allow_missing_decision_anchor: bool,
    device: str,
) -> Dict[str, Any]:
    prompt_tokens = model.to_tokens(prompt_str, prepend_bos=False)
    prompt_len = int(prompt_tokens.shape[1])
    model_dtype = next(model.parameters()).dtype
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=model_dtype)
        if device == "cuda" and model_dtype in (torch.bfloat16, torch.float16)
        else nullcontext()
    )
    with amp_ctx:
        out = model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            stop_at_eos=True,
            verbose=False,
        )
    full_ids = out[0].tolist()
    gen_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)

    fallback_used = False
    if decision_anchor_mode == "first_generated_token":
        gender_pos = prompt_len if prompt_len < len(full_ids) else -1
    else:
        gender_pos = sl.find_gender_decision_pos(tokenizer, full_ids, prompt_len)
        if gender_pos == -1 and allow_missing_decision_anchor:
            gender_pos = prompt_len if prompt_len < len(full_ids) else -1
            fallback_used = True

    if gender_pos <= 0:
        delta = float("nan")
    else:
        with torch.no_grad():
            with amp_ctx:
                logits = model(torch.tensor(full_ids, device=device).unsqueeze(0))
        dec = logits[0, gender_pos - 1, :]
        delta = float((dec[female_id] - dec[male_id]).item())

    return {
        "full_token_ids": full_ids,
        "prompt_len": prompt_len,
        "generated_text": gen_text,
        "gender_pos": int(gender_pos),
        "logit_diff": float(delta),
        "temperature": float(temperature),
        "decision_anchor_mode": decision_anchor_mode,
        "decision_anchor_fallback": bool(fallback_used),
    }


def run_stage1_simple_prompts(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    model,
    tokenizer,
    female_id: int,
    male_id: int,
    device: str,
    prompts_df: pd.DataFrame,
) -> None:
    if args.resume and sl.checkpoint_exists(paths["baseline"]) and sl.checkpoint_exists(paths["run_index"]):
        print("Stage 1: baseline artifacts already exist; skipping due to --resume")
        return

    temperatures = sl._parse_float_list(args.temperatures)
    if args.rep_temp_idx >= len(temperatures):
        raise ValueError(f"--rep-temp-idx={args.rep_temp_idx} out of range for {len(temperatures)} temperatures")
    if int(args.stage1_progress_every) <= 0:
        raise ValueError("--stage1-progress-every must be > 0")

    baseline_data: Dict[str, Any] = {}
    skipped_missing_anchor = 0
    total_traces = int(len(prompts_df) * len(temperatures))
    progress_every = int(args.stage1_progress_every)
    processed = 0
    stage1_start_ts = time.time()
    print(f"Stage1 progress: starting {total_traces} traces")
    for row in prompts_df.itertuples(index=False):
        prompt_id = str(row.prompt_id)
        prompt_text = str(row.prompt_text)
        cond_raw = str(getattr(row, "expected_condition", "")).strip()
        group_raw = str(getattr(row, "group", "")).strip()
        condition = cond_raw or group_raw or "simple_prompt"
        for temp_idx, temp in enumerate(temperatures):
            key = f"{prompt_id}|temp{temp_idx}"
            trace = _generate_simple_trace(
                model=model,
                tokenizer=tokenizer,
                prompt_str=prompt_text,
                temperature=float(temp),
                max_new_tokens=int(args.max_new_tokens),
                female_id=female_id,
                male_id=male_id,
                decision_anchor_mode=str(args.decision_anchor_mode),
                allow_missing_decision_anchor=bool(args.allow_missing_decision_anchor),
                device=device,
            )
            if trace["gender_pos"] == -1:
                skipped_missing_anchor += 1
            trace["condition"] = condition
            trace["variation"] = prompt_id
            trace["prompt_id"] = prompt_id
            trace["prompt_group"] = group_raw
            trace["temp_idx"] = int(temp_idx)
            trace["prompt_str"] = prompt_text
            baseline_data[key] = trace
            processed += 1
            if processed % progress_every == 0 or processed == total_traces:
                elapsed = max(1e-6, time.time() - stage1_start_ts)
                sec_per_trace = elapsed / processed
                remaining = max(0, total_traces - processed)
                eta_sec = int(round(sec_per_trace * remaining))
                print(
                    f"Stage1 progress: {processed}/{total_traces} "
                    f"({100.0 * processed / total_traces:.1f}%) "
                    f"elapsed={elapsed / 60.0:.1f}m eta={eta_sec / 60.0:.1f}m",
                    flush=True,
                )

    sl.save_json(paths["baseline"], baseline_data)
    # Alias for interoperability with external scripts expecting baseline_runs.json.
    sl.save_json(paths["artifacts_dir"] / "baseline_runs.json", baseline_data)

    rows = []
    eligible = []
    for key, row in baseline_data.items():
        delta = float(row["logit_diff"])
        ok = row["gender_pos"] != -1 and np.isfinite(delta) and delta > 0
        if ok:
            eligible.append(key)
        rows.append(
            {
                "key": key,
                "prompt_id": row["prompt_id"],
                "condition": row["condition"],
                "variation": row["variation"],
                "temp_idx": row["temp_idx"],
                "delta_base": delta,
                "gender_pos": row["gender_pos"],
                "eligible": ok,
            }
        )
    df = pd.DataFrame(rows)

    rep_temp_idx = int(args.rep_temp_idx)
    rep_keys: List[str] = []
    prompt_ids = prompts_df["prompt_id"].astype(str).tolist()
    for prompt_id in prompt_ids:
        candidates = df[(df["variation"] == prompt_id) & (df["temp_idx"] == rep_temp_idx) & (df["eligible"])]
        if len(candidates) == 0:
            candidates = df[(df["variation"] == prompt_id) & (df["eligible"])]
        if len(candidates) > 0:
            rep_keys.append(candidates.sort_values("delta_base", ascending=False).iloc[0]["key"])

    run_index = {
        "eligible_keys": eligible,
        "representative_keys": sorted(set(rep_keys)),
        "conditions": sorted(df["condition"].astype(str).unique().tolist()),
        "prompt_vars": prompt_ids,
        "temperatures": temperatures,
        "prompt_count": int(len(prompt_ids)),
        "missing_anchor_traces": int(skipped_missing_anchor),
    }
    sl.save_json(paths["run_index"], run_index)
    print(
        f"Stage1 done: total={len(df)} eligible={len(eligible)} "
        f"representative={len(run_index['representative_keys'])} missing_anchor={skipped_missing_anchor}"
    )


def save_run_config_simple(args: argparse.Namespace, paths: Dict[str, Path], prompts_df: pd.DataFrame) -> None:
    payload = {
        "run_id": args.run_id,
        "runtime_profile": args.runtime_profile,
        "model_name": sl.ACTIVE_MODEL_NAME,
        "sae_layers": list(sl.SAE_LAYERS),
        "ablation_mode": args.ablation_mode,
        "stage3_key_source": args.stage3_key_source,
        "stage3_latent_source": args.stage3_latent_source,
        "stage3_curated_latents_json": args.stage3_curated_latents_json,
        "stage3_curated_latents_csv": args.stage3_curated_latents_csv,
        "stage3_max_keys_per_group": int(args.stage3_max_keys_per_group),
        "allow_post_decision_coords": bool(args.allow_post_decision_coords),
        "gating_mode": args.gating_mode,
        "gating_threshold_mode": args.gating_threshold_mode,
        "gating_positive_percentile": float(args.gating_positive_percentile),
        "gating_positive_absolute": float(args.gating_positive_absolute),
        "intervention_scope": args.intervention_scope,
        "stage4_feature_source": args.stage4_feature_source,
        "stage4_max_latents": int(args.stage4_max_latents),
        "stage4_max_keys": int(args.stage4_max_keys),
        "simple_prompts_csv": (
            str(Path(args.simple_prompts_csv).expanduser().resolve())
            if args.simple_prompts_csv.strip()
            else ""
        ),
        "simple_prompts_source": "csv" if args.simple_prompts_csv.strip() else "built_in",
        "simple_prompt_condition": str(args.simple_prompt_condition),
        "expected_prompt_count": int(args.expected_prompt_count),
        "actual_prompt_count": int(len(prompts_df)),
        "decision_anchor_mode": args.decision_anchor_mode,
        "allow_missing_decision_anchor": bool(args.allow_missing_decision_anchor),
    }
    sl.save_json(paths["run_config"], payload)


def preflight_summary_simple(args: argparse.Namespace, device: str, prompts_df: pd.DataFrame) -> None:
    dtype = sl._resolve_dtype(device, args.dtype, args.runtime_profile)
    print("=== SAE simple-prompts preflight ===")
    print(f"runtime_profile={args.runtime_profile}")
    print(f"detected_device={device}")
    print(f"dtype={dtype}")
    print(f"stage={args.stage}")
    print(f"run_id={args.run_id}")
    print(f"model_preset={args.model_preset}")
    print(f"model_name={sl.ACTIVE_MODEL_NAME}")
    print(f"sae_releases={sl.ACTIVE_SAE_RELEASES}")
    print(f"sae_layer_preset={args.sae_layer_preset}")
    print(f"sae_layer_chunk_size={args.sae_layer_chunk_size}")
    if args.sae_layer_preset == "model_all" and not sl.SAE_LAYERS and not args.sae_layers:
        print("sae_layers=<deferred: model_all; resolves after model load>")
    else:
        print(f"sae_layers={sl.SAE_LAYERS}")
    if args.simple_prompts_csv.strip():
        print(f"simple_prompts_source=csv")
        print(f"simple_prompts_csv={Path(args.simple_prompts_csv).expanduser().resolve()}")
    else:
        print(f"simple_prompts_source=built_in")
        print(f"simple_prompt_condition={args.simple_prompt_condition}")
    print(f"prompt_count={len(prompts_df)}")
    print(f"stage1_progress_every={args.stage1_progress_every}")
    print(f"decision_anchor_mode={args.decision_anchor_mode}")
    print(f"allow_missing_decision_anchor={args.allow_missing_decision_anchor}")
    print(f"stage3_latent_source={args.stage3_latent_source}")
    print(f"max_sweep_coords={args.max_sweep_coords}")
    print(f"max_control_rows={args.max_control_rows}")
    print(f"ablation_mode={args.ablation_mode}")
    print(f"gating_mode={args.gating_mode}")
    print(f"gating_threshold_mode={args.gating_threshold_mode}")
    print(f"gating_positive_percentile={args.gating_positive_percentile}")
    print(f"gating_positive_absolute={args.gating_positive_absolute}")
    print(f"intervention_scope={args.intervention_scope}")
    print(f"stage4_feature_source={args.stage4_feature_source}")
    print(f"stage4_max_latents={args.stage4_max_latents}")
    print(f"stage4_max_keys={args.stage4_max_keys}")
    print("====================================")


def inject_stage3_prompt_count(paths: Dict[str, Path]) -> None:
    if not sl.checkpoint_exists(paths["sweep_coords"]):
        return
    if not sl.checkpoint_exists(paths["run_index"]):
        return
    sweep = sl.load_json(paths["sweep_coords"])
    run_index = sl.load_json(paths["run_index"])
    meta = dict(sweep.get("metadata", {}))
    meta["prompt_count"] = int(run_index.get("prompt_count", 0))
    sweep["metadata"] = meta
    sl.save_json(paths["sweep_coords"], sweep)


def inject_prompt_id_into_sweep_results(paths: Dict[str, Path]) -> None:
    if not sl.checkpoint_exists(paths["sweep_results"]):
        return
    baseline = sl.load_json(paths["baseline"])
    df = pd.read_parquet(paths["sweep_results"])
    if len(df) == 0:
        return
    if "prompt_id" in df.columns:
        return
    prompt_map = {k: str(v.get("prompt_id", "")) for k, v in baseline.items()}
    df["prompt_id"] = df["trace_key"].map(prompt_map).fillna("")
    df.to_parquet(paths["sweep_results"], index=False)


def write_prompt_level_summary(paths: Dict[str, Path]) -> None:
    if not sl.checkpoint_exists(paths["sweep_results"]):
        return
    df = pd.read_parquet(paths["sweep_results"])
    if len(df) == 0 or "prompt_id" not in df.columns:
        return
    out = (
        df.groupby(["prompt_id", "layer", "feature_idx", "gate_sign"], as_index=False)
        .agg(
            n_hits=("norm_effect", "size"),
            mean_norm_effect=("norm_effect", "mean"),
            median_norm_effect=("norm_effect", "median"),
            mean_delta_shift=("delta_shift", "mean"),
        )
        .sort_values(["mean_norm_effect"], ascending=False)
    )
    out_path = paths["artifacts_dir"] / "prompt_level_summary.csv"
    out.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    apply_profile_defaults(args)

    prompts_df = _load_simple_prompts(args)

    sl.ACTIVE_MODEL_NAME, sl.ACTIVE_SAE_RELEASES, sl.ACTIVE_SAE_FAMILY = sl.resolve_model_config(args)
    if args.model_preset == "gemma_scope2_4b_it" and not args.sae_layers and args.sae_layer_preset != "model_all":
        print(
            "Auto-switching --sae-layer-preset to model_all for gemma_scope2_4b_it "
            "(use --sae-layers to manually override)."
        )
        args.sae_layer_preset = "model_all"
    sl.SAE_LAYERS = sl.resolve_sae_layers(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = sl.run_paths(run_dir)
    paths["artifacts_dir"].mkdir(parents=True, exist_ok=True)
    paths["heatmaps_dir"].mkdir(parents=True, exist_ok=True)
    save_run_config_simple(args, paths, prompts_df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preflight_summary_simple(args, device, prompts_df)
    if args.dry_run:
        print(f"Dry run OK. run_dir={run_dir}")
        return

    stage_order = ["1", "2", "3", "4", "5", "6"] if args.stage == "all" else [args.stage]
    if args.no_controls:
        stage_order = [s for s in stage_order if s != "5"]

    overrides: Dict[int, str] = {}
    if args.sae_id_overrides:
        raw = json.loads(args.sae_id_overrides)
        overrides = {int(k): str(v) for k, v in raw.items()}

    model = None
    tokenizer = None
    female_id = None
    male_id = None
    saes = None
    sae_ids = None
    progress_path = paths["progress"]

    try:
        for stage in stage_order:
            if args.resume and sl.stage_outputs_exist(stage, paths):
                print(f"Stage {stage}: outputs exist; skipping due to --resume")
                sl.mark_stage_completed(progress_path, stage)
                continue

            print(f"\n>>> Running stage {stage}")
            if stage in {"1", "2", "3", "4", "5"} and model is None:
                model, tokenizer, female_id, male_id, _ = sl.load_model_and_tokenizer(args)
                if args.sae_layer_preset == "model_all" and not args.sae_layers:
                    sl.SAE_LAYERS = sl.resolve_sae_layers(args, model_n_layers=int(model.cfg.n_layers))
                    print(f"Resolved model_all SAE layers (n={len(sl.SAE_LAYERS)})")

            use_chunking = stage in {"2", "3", "4", "5"} and args.sae_layer_chunk_size > 0
            if use_chunking:
                chunks = sl._chunk_layers(sl.SAE_LAYERS, args.sae_layer_chunk_size)
                print(f"Stage {stage}: chunking enabled ({len(chunks)} chunk(s), chunk_size={args.sae_layer_chunk_size})")
                for chunk_idx, layer_chunk in enumerate(chunks):
                    print(f"  - chunk {chunk_idx + 1}/{len(chunks)} layers={layer_chunk}")
                    saes, sae_ids = sl.load_saes(model.cfg.device, overrides, layers=layer_chunk)
                    reset_output = chunk_idx == 0
                    if stage == "2":
                        if args.stage3_latent_source != "top_latents":
                            print("Stage2 skipped: curated Stage3 latent source selected.")
                        else:
                            sl.run_stage2_discovery(
                                args,
                                paths,
                                model,
                                saes,
                                sae_ids,
                                layer_subset=layer_chunk,
                                chunk_mode=True,
                                reset_output=reset_output,
                            )
                    elif stage == "3":
                        sl.run_stage3_cache_latents(
                            args,
                            paths,
                            model,
                            tokenizer,
                            saes,
                            layer_subset=layer_chunk,
                            chunk_mode=True,
                            reset_output=reset_output,
                        )
                    elif stage == "4":
                        sl.run_stage4_causal_sweep(
                            args,
                            paths,
                            model,
                            tokenizer,
                            female_id,
                            male_id,
                            saes,
                            layer_subset=layer_chunk,
                            chunk_mode=True,
                            reset_output=reset_output,
                        )
                    elif stage == "5":
                        sl.run_stage5_controls(
                            args,
                            paths,
                            model,
                            female_id,
                            male_id,
                            saes,
                            layer_subset=layer_chunk,
                            chunk_mode=True,
                            reset_output=reset_output,
                        )
                    del saes, sae_ids
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if stage == "3":
                    inject_stage3_prompt_count(paths)
                if stage == "4":
                    inject_prompt_id_into_sweep_results(paths)
                if stage == "6":
                    write_prompt_level_summary(paths)
                sl.mark_stage_completed(progress_path, stage)
                continue

            if stage in {"2", "3", "4", "5"} and saes is None:
                saes, sae_ids = sl.load_saes(model.cfg.device, overrides)

            if stage == "1":
                run_stage1_simple_prompts(args, paths, model, tokenizer, female_id, male_id, model.cfg.device, prompts_df)
            elif stage == "2":
                if args.stage3_latent_source != "top_latents":
                    print("Stage2 skipped: curated Stage3 latent source selected.")
                else:
                    sl.run_stage2_discovery(args, paths, model, saes, sae_ids)
            elif stage == "3":
                sl.run_stage3_cache_latents(args, paths, model, tokenizer, saes)
                inject_stage3_prompt_count(paths)
            elif stage == "4":
                sl.run_stage4_causal_sweep(args, paths, model, tokenizer, female_id, male_id, saes)
                inject_prompt_id_into_sweep_results(paths)
            elif stage == "5":
                sl.run_stage5_controls(args, paths, model, female_id, male_id, saes)
            elif stage == "6":
                sl.run_stage6_analysis(args, paths)
                write_prompt_level_summary(paths)
            else:
                raise ValueError(f"Unknown stage: {stage}")
            sl.mark_stage_completed(progress_path, stage)
    except Exception as e:
        failing = stage if "stage" in locals() else args.stage
        sl.mark_stage_failed(progress_path, failing, str(e))
        raise
    finally:
        if model is not None:
            del model
        if saes is not None:
            del saes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
