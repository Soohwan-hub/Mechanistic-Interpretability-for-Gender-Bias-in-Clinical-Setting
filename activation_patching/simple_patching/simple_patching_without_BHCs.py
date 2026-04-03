"""
Simple gender activation patching for Qwen 2.5 7B: gather activations from
"The patient is Male/Female" at MLP layers, patch into BHC vignette prompts,
compute rewrite score at forced "Gender:" position. Supports checkpoint/resume,
Lambda GPU-friendly defaults, and visualizations (heatmaps + per-layer plots).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig

# Optional plotting (kaleido for static export)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODEL_OPTIONS = {
    1: "Qwen/Qwen2.5-7B-Instruct",
    2: "allenai/OLMo-7B-0724-Instruct-hf",
    3: "meta-llama/Llama-3.1-8B-Instruct",
}

COHORT_TO_CONDITION_NAME = {
    "depression": "depression",
    "asthma": "asthma",
    "multiple_sclerosis": "multiple sclerosis",
    "rheumatoid_arthritis": "rheumatoid arthritis",
    "sarcoidosis": "sarcoidosis",
}

# Patch with Male activations for all selected female-biased cohorts
# so the intervention is in the opposite direction of the bias.
COHORT_TO_PATCH_GENDER = {
    "depression": "Male",
    "asthma": "Male",
    "multiple_sclerosis": "Male",
    "rheumatoid_arthritis": "Male",
    "sarcoidosis": "Male",
}

DEFAULT_COHORTS = (
    "asthma",
    "depression",
    "multiple_sclerosis",
    "rheumatoid_arthritis",
    "sarcoidosis",
)

# Accept common spacing/hyphen/typo variants for cohort names on CLI.
COHORT_ALIASES = {
    "asthma": "asthma",
    "depression": "depression",
    "multiple_sclerosis": "multiple_sclerosis",
    "multiple sclerosis": "multiple_sclerosis",
    "multiple-scierosis": "multiple_sclerosis",
    "multiple scierosis": "multiple_sclerosis",
    "rheumatoid_arthritis": "rheumatoid_arthritis",
    "rheumatoid arthritis": "rheumatoid_arthritis",
    "rheunmatoid arthirties": "rheumatoid_arthritis",
    "rheunmatoid_arthirties": "rheumatoid_arthritis",
    "sarcoidosis": "sarcoidosis",
}

# Simple prompt templates: [CONDITION] = condition name
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

SCORE_MATRIX_KEYS = ("rewrite_scores", "logprob_scores", "logprob_delta_scores")
STAT_KEYS = ("mean", "median", "trimmed_mean", "topk_mean")


# -----------------------------------------------------------------------------
# Config and helpers
# -----------------------------------------------------------------------------
def _resolve(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _atomic_write_pickle(path: str, obj: Any) -> None:
    import pickle
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _config_hash(args: argparse.Namespace) -> str:
    h = hashlib.sha256()
    for k in sorted(vars(args).keys()):
        v = getattr(args, k)
        h.update(f"{k}={v!r}".encode())
    return h.hexdigest()[:16]


def _resolve_score_keys(raw: str) -> Tuple[str, ...]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError("Empty --score-keys. Use one or more score keys, or 'all'.")
    if len(parts) == 1 and parts[0].lower() == "all":
        return SCORE_MATRIX_KEYS
    invalid = [x for x in parts if x not in SCORE_MATRIX_KEYS]
    if invalid:
        valid = ",".join(SCORE_MATRIX_KEYS)
        invalid_joined = ",".join(sorted(set(invalid)))
        raise ValueError(f"Unknown --score-keys value(s): {invalid_joined}. Valid values: {valid},all")
    deduped: List[str] = []
    seen = set()
    for key in parts:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return tuple(deduped)


def _resolve_model_name(model_id: int) -> str:
    if model_id not in MODEL_OPTIONS:
        valid = ",".join(str(k) for k in sorted(MODEL_OPTIONS.keys()))
        raise ValueError(f"Unknown --model-id {model_id}. Valid values: {valid}")
    return MODEL_OPTIONS[model_id]


def _parse_nonnegative_int_csv(raw: str, arg_name: str) -> Tuple[int, ...]:
    value = raw.strip()
    if not value:
        return tuple()
    parsed: List[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid {arg_name} value {token!r}; expected comma-separated integers.") from exc
        if idx < 0:
            raise ValueError(f"Invalid {arg_name} value {token!r}; layer indices must be >= 0.")
        parsed.append(idx)
    return tuple(sorted(set(parsed)))


# -----------------------------------------------------------------------------
# Prompt building
# -----------------------------------------------------------------------------
def _validated_gender_token_ids(llm: LanguageModel, gender: str) -> torch.Tensor:
    """Return token ids for ' {gender}' with strict validation for this analysis."""
    normalized = gender.strip().lower()
    assert normalized in {"male", "female"}, (
        f"Expected gender to be male/female, got {gender!r}"
    )
    token_ids = llm.tokenizer(
        " " + gender,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    assert int(token_ids.numel()) == 1, (
        f"Expected ' {gender}' to map to a single token, got {int(token_ids.numel())} "
        f"tokens: {token_ids.tolist()} (decoded={llm.tokenizer.decode(token_ids.tolist())!r})"
    )
    decoded = llm.tokenizer.decode(token_ids.tolist()).strip().lower()
    assert decoded in {"male", "female"}, (
        f"Token decode check failed for gender={gender!r}: decoded={decoded!r}, ids={token_ids.tolist()}"
    )
    return token_ids


def build_clean_prompt(llm: LanguageModel, gender: str) -> Tuple[str, torch.Tensor, int]:
    """Build 'The patient is {Male|Female}' prompt and return (chat_text, token_ids, patch_token_idx)."""
    msg = {"role": "user", "content": f"The patient is {gender}"}
    clean_text = llm.tokenizer.apply_chat_template(
        [msg], tokenize=False, add_generation_prompt=True
    )
    clean_tokens = llm.tokenizer(clean_text, return_tensors="pt")["input_ids"][0]
    token_ids = _validated_gender_token_ids(llm, gender)
    matches = torch.argwhere(clean_tokens == token_ids[-1])
    assert matches.shape[0] == 1, (
        f"Expected exactly one source token match for gender={gender!r}, found {matches.shape[0]} "
        f"for token_id={int(token_ids[-1].item())}"
    )
    patch_token_from = matches[0][0].item()
    return clean_text, clean_tokens, patch_token_from


def build_corrupt_prompt(
    llm: LanguageModel,
    template: str,
    condition_name: str,
) -> Tuple[str, torch.Tensor]:
    """Build corrupted prompt with [CONDITION] replaced; append 'Gender:' for forced next token."""
    body = template.replace("[CONDITION]", condition_name)
    messages = [
        {"role": "system", "content": "You are a helpful clinical assistant."},
        {"role": "user", "content": body},
    ]
    corrupted_text = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    corrupted_text += "Gender:"
    corrupted_tokens = llm.tokenizer(corrupted_text, return_tensors="pt")["input_ids"][0]
    return corrupted_text, corrupted_tokens


# -----------------------------------------------------------------------------
# Patching sweep (MLP only, probability-based rewrite score)
# -----------------------------------------------------------------------------
def run_patch_sweep(
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
    """
    Run layer x token sweep and return score matrices/baselines.
    Uses MLP down_proj output for patching. Exposes:
      - rewrite_scores = (p* - p) / (1 - p)
      - logprob_scores = log p*(target)
      - logprob_delta_scores = log p*(target) - log p(target)
    """
    token_count = corrupted_tokens.shape[0]
    if max_tokens > 0:
        token_count = min(max_tokens, token_count)

    clean_patch_token_from = patch_token_from
    diff = len(llm.tokenizer(clean_prompt, return_tensors="pt")["input_ids"][0]) - len(corrupted_tokens)
    offset = diff if diff > 0 else 0

    layers_swept = list(range(layer_start, min(layer_end, num_layers)))
    n_l = len(layers_swept)
    need_rewrite = "rewrite_scores" in selected_score_keys
    need_logprob = "logprob_scores" in selected_score_keys
    need_logprob_delta = "logprob_delta_scores" in selected_score_keys
    rewrite_list: List[float] = []
    logprob_list: List[float] = []
    logprob_delta_list: List[float] = []
    corrupted_prob_val: Optional[float] = None
    corrupted_logprob_val: Optional[float] = None

    for start in range(0, n_l, step):
        end = min(start + step, n_l)
        layer_indices = [layers_swept[i] for i in range(start, end)]

        # Cache clean activations for this layer chunk in an isolated trace scope.
        saved_clean: Dict[int, Any] = {}
        with torch.no_grad():
            with llm.generate(max_new_tokens=1) as tracer:
                with tracer.invoke(clean_prompt):
                    for li in layer_indices:
                        saved_clean[li] = llm.model.layers[li].mlp.down_proj.output[
                            :, clean_patch_token_from, :
                        ].save()
        z_hs: Dict[int, torch.Tensor] = {}
        for li in layer_indices:
            z_hs[li] = _resolve(saved_clean[li]).detach().clone()

        # Compute baseline corrupted probability/log-prob once using a short-lived trace.
        if (need_rewrite and corrupted_prob_val is None) or (
            (need_logprob or need_logprob_delta) and corrupted_logprob_val is None
        ):
            with torch.no_grad():
                with llm.generate(max_new_tokens=1) as tracer:
                    with tracer.invoke(corrupted_prompt):
                        logits = llm.lm_head.output
                        if need_logprob or need_logprob_delta:
                            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
                            corrupted_logprob_proxy = log_probs[target_gender_token_id].save()
                            if need_rewrite:
                                corrupted_prob_proxy = torch.exp(log_probs[target_gender_token_id]).save()
                        elif need_rewrite:
                            probs = torch.softmax(logits[0, -1, :], dim=-1)
                            corrupted_prob_proxy = probs[target_gender_token_id].save()
            if need_rewrite:
                corrupted_prob_val = float(_resolve(corrupted_prob_proxy).cpu().float().item())
            if need_logprob or need_logprob_delta:
                corrupted_logprob_val = float(_resolve(corrupted_logprob_proxy).cpu().float().item())

        corrupted_prob = corrupted_prob_val if corrupted_prob_val is not None else 0.0
        corrupted_logprob = corrupted_logprob_val if corrupted_logprob_val is not None else float("-inf")
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
                            if need_logprob or need_logprob_delta:
                                patched_logprob = torch.log_softmax(
                                    patched_logits[0, -1, :], dim=-1
                                )[target_gender_token_id]
                            if need_rewrite:
                                if need_logprob or need_logprob_delta:
                                    patched_prob = torch.exp(patched_logprob)
                                else:
                                    patched_prob = torch.softmax(
                                        patched_logits[0, -1, :], dim=-1
                                    )[target_gender_token_id]
                                rewrite_score = (patched_prob - corrupted_prob) / denom
                                rewrite_proxy = rewrite_score.save()
                            if need_logprob:
                                logprob_proxy = patched_logprob.save()
                            if need_logprob_delta:
                                logprob_delta = patched_logprob - corrupted_logprob
                                logprob_delta_proxy = logprob_delta.save()
                if need_rewrite:
                    rewrite_list.append(float(_resolve(rewrite_proxy).cpu().float().item()))
                if need_logprob:
                    logprob_list.append(float(_resolve(logprob_proxy).cpu().float().item()))
                if need_logprob_delta:
                    logprob_delta_list.append(float(_resolve(logprob_delta_proxy).cpu().float().item()))

    result: Dict[str, Any] = {
        "corrupted_prob": corrupted_prob_val or 0.0,
        "corrupted_logprob": corrupted_logprob_val
        if corrupted_logprob_val is not None
        else float("-inf"),
    }
    if need_rewrite:
        result["rewrite_scores"] = np.array(rewrite_list, dtype=float).reshape(n_l, token_count)
    if need_logprob:
        result["logprob_scores"] = np.array(logprob_list, dtype=float).reshape(n_l, token_count)
    if need_logprob_delta:
        result["logprob_delta_scores"] = np.array(logprob_delta_list, dtype=float).reshape(n_l, token_count)
    return result


# -----------------------------------------------------------------------------
# Aggregation from raw matrix
# -----------------------------------------------------------------------------
def _trimmed_mean_1d(values: np.ndarray, trim_frac: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if trim_frac <= 0:
        return float(np.mean(finite))
    sorted_vals = np.sort(finite)
    trim_n = int(np.floor(sorted_vals.size * trim_frac))
    if (2 * trim_n) >= sorted_vals.size:
        return float(np.mean(sorted_vals))
    return float(np.mean(sorted_vals[trim_n: sorted_vals.size - trim_n]))


def layer_aggregates(score_matrix: np.ndarray, top_k: int = 30, trim_frac: float = 0.10) -> Dict[str, np.ndarray]:
    """Per-layer aggregate stats for score_matrix shape (num_layers, num_tokens)."""
    n_l, _ = score_matrix.shape
    per_layer_mean = np.nanmean(score_matrix, axis=1)
    per_layer_median = np.nanmedian(score_matrix, axis=1)
    per_layer_trimmed_mean = np.zeros(n_l)
    per_layer_topk_mean = np.zeros(n_l)

    for i in range(n_l):
        row = score_matrix[i]
        per_layer_trimmed_mean[i] = _trimmed_mean_1d(row, trim_frac)
        sorted_desc = np.sort(row)[::-1]
        k = min(top_k, len(sorted_desc))
        per_layer_topk_mean[i] = float(np.mean(sorted_desc[:k])) if k else np.nan

    return {
        "mean": per_layer_mean,
        "median": per_layer_median,
        "trimmed_mean": per_layer_trimmed_mean,
        "topk_mean": per_layer_topk_mean,
    }


def _filter_plot_stats_by_layers(
    per_layer_stats: Dict[str, np.ndarray],
    layer_labels: List[int],
    excluded_layers: Tuple[int, ...],
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    if not excluded_layers:
        return per_layer_stats, layer_labels
    excluded = set(excluded_layers)
    keep_idx = [i for i, layer in enumerate(layer_labels) if layer not in excluded]
    filtered_layer_labels = [layer_labels[i] for i in keep_idx]
    filtered_stats: Dict[str, np.ndarray] = {
        key: values[keep_idx] for key, values in per_layer_stats.items()
    }
    return filtered_stats, filtered_layer_labels


# -----------------------------------------------------------------------------
# Progress / checkpoint
# -----------------------------------------------------------------------------
def unit_key(cohort: str, prompt_id: int) -> str:
    return f"{cohort}:prompt{prompt_id}"


def load_progress(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "progress.json"
    if not p.exists():
        return {"completed": [], "failed": {}, "config_hash": "", "model_name": "", "updated": ""}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_progress(run_dir: Path, data: Dict[str, Any]) -> None:
    data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_write_json(str(run_dir / "progress.json"), data)


def mark_completed(run_dir: Path, key: str, progress: Dict[str, Any]) -> None:
    if key not in progress["completed"]:
        progress["completed"].append(key)
    if key in progress.get("failed", {}):
        del progress["failed"][key]
    save_progress(run_dir, progress)


def mark_failed(run_dir: Path, key: str, err: str, progress: Dict[str, Any]) -> None:
    progress.setdefault("failed", {})[key] = err
    save_progress(run_dir, progress)


# -----------------------------------------------------------------------------
# Artifacts: save/load per-unit results
# -----------------------------------------------------------------------------
def artifact_path(run_dir: Path, cohort: str, prompt_id: int) -> Path:
    return run_dir / "artifacts" / f"{cohort}_prompt{prompt_id}.pkl"


def parse_artifact_stem(name: str) -> Optional[Tuple[str, int]]:
    """
    Parse artifact stem into (cohort, prompt_id).

    Supported:
      - cohort_promptN
      - cohort_caseXXXX_promptN (legacy)
    """
    if "_prompt" not in name:
        return None
    prefix, prompt_suffix = name.rsplit("_prompt", 1)
    try:
        prompt_id = int(prompt_suffix)
    except ValueError:
        return None

    cohort = prefix
    if "_case" in prefix:
        maybe_cohort, maybe_case = prefix.rsplit("_case", 1)
        if maybe_case.isdigit():
            cohort = maybe_cohort
    if not cohort:
        return None
    return cohort, prompt_id


def save_unit_artifact(
    run_dir: Path,
    cohort: str,
    prompt_id: int,
    score_matrices: Dict[str, np.ndarray],
    token_labels: List[str],
    layer_labels: List[int],
    metadata: Dict[str, Any],
) -> None:
    path = artifact_path(run_dir, cohort, prompt_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_labels": token_labels,
        "layer_labels": layer_labels,
        "metadata": metadata,
    }
    for key in SCORE_MATRIX_KEYS:
        if key in score_matrices:
            payload[key] = score_matrices[key]
    _atomic_write_pickle(str(path), payload)


def load_unit_artifact(run_dir: Path, cohort: str, prompt_id: int) -> Dict[str, Any]:
    path = artifact_path(run_dir, cohort, prompt_id)
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
def plot_heatmap(
    rewrite_scores: np.ndarray,
    token_labels: List[str],
    layer_labels: List[int],
    title: str,
    path: str,
    plot_format: str = "pdf",
    mode: str = "single",
    token_window: int = 180,
    overview_bin_size: int = 10,
) -> None:
    if not _HAS_PLOTLY:
        return
    # Limit labels if too many tokens.
    n_t = rewrite_scores.shape[1]
    if len(token_labels) > 80:
        x_labels = [str(i) for i in range(n_t)]
    else:
        x_labels = token_labels
    ext = plot_format if plot_format.startswith(".") else f".{plot_format}"
    out = path if path.endswith(ext) else path + ext

    if mode == "single":
        fig = px.imshow(
            rewrite_scores,
            color_continuous_midpoint=0.0,
            color_continuous_scale="RdBu",
            labels={"x": "Token", "y": "Layer", "color": "Rewrite score"},
            x=x_labels,
            y=layer_labels,
            title=title,
        )
        fig.update_yaxes(tickmode="linear", tick0=min(layer_labels), dtick=1, automargin=True)
        fig.update_layout(height=max(700, 24 * len(layer_labels) + 160), width=1800)
        pio.write_image(fig, out)
        return

    if mode != "full_suite":
        raise ValueError(f"Unknown heatmap mode: {mode}")

    # full_suite: split layers into two heatmaps for readability on deep models.
    n_layers = rewrite_scores.shape[0]
    split_idx = max(1, n_layers // 2)
    chunks = [
        (rewrite_scores[:split_idx], layer_labels[:split_idx], "_part1"),
        (rewrite_scores[split_idx:], layer_labels[split_idx:], "_part2"),
    ]

    for chunk_scores, chunk_layers, suffix in chunks:
        if chunk_scores.size == 0:
            continue
        # Split token axis into windows so each image remains readable.
        window = token_window if token_window > 0 else chunk_scores.shape[1]
        num_tokens = chunk_scores.shape[1]
        for tok_start in range(0, num_tokens, window):
            tok_end = min(tok_start + window, num_tokens)
            window_scores = chunk_scores[:, tok_start:tok_end]
            window_x = x_labels[tok_start:tok_end]

            fig = px.imshow(
                window_scores,
                color_continuous_midpoint=0.0,
                color_continuous_scale="RdBu",
                labels={"x": "Token", "y": "Layer", "color": "Rewrite score"},
                x=window_x,
                y=chunk_layers,
                title=f"{title} ({chunk_layers[0]}-{chunk_layers[-1]}, tok {tok_start}-{tok_end-1})",
            )
            # Prevent Plotly from collapsing y ticks to only a few labels.
            fig.update_yaxes(tickmode="linear", tick0=min(chunk_layers), dtick=1, automargin=True)
            fig.update_layout(height=max(560, 28 * len(chunk_layers) + 140), width=1800)
            out_chunk = out.replace(ext, f"{suffix}_tok{tok_start:04d}-{tok_end-1:04d}{ext}")
            pio.write_image(fig, out_chunk)

    # Also save a compact overview by token binning for quick inspection.
    if overview_bin_size > 1:
        n_bins = (rewrite_scores.shape[1] + overview_bin_size - 1) // overview_bin_size
        binned = np.zeros((rewrite_scores.shape[0], n_bins), dtype=float)
        bin_labels: List[str] = []
        for i in range(n_bins):
            s = i * overview_bin_size
            e = min((i + 1) * overview_bin_size, rewrite_scores.shape[1])
            binned[:, i] = np.nanmean(rewrite_scores[:, s:e], axis=1)
            bin_labels.append(f"{s}-{e-1}")

        fig = px.imshow(
            binned,
            color_continuous_midpoint=0.0,
            color_continuous_scale="RdBu",
            labels={"x": f"Token bins ({overview_bin_size})", "y": "Layer", "color": "Rewrite score"},
            x=bin_labels,
            y=layer_labels,
            title=f"{title} (overview, binned)",
        )
        fig.update_yaxes(tickmode="linear", tick0=min(layer_labels), dtick=1, automargin=True)
        fig.update_layout(height=max(700, 22 * len(layer_labels) + 160), width=1800)
        out_overview = out.replace(ext, f"_overview_bin{overview_bin_size}{ext}")
        pio.write_image(fig, out_overview)


def plot_layer_curves(
    per_layer_stats: Dict[str, np.ndarray],
    layer_labels: List[int],
    title: str,
    path: str,
    plot_format: str = "pdf",
    top_k: int = 30,
) -> None:
    if not _HAS_PLOTLY:
        return
    fig = go.Figure()
    curve_defs = [
        ("mean", "Mean score"),
        ("median", "Median score"),
        ("trimmed_mean", "Trimmed mean score"),
        ("topk_mean", f"Top-{top_k} mean score"),
    ]
    for key, label in curve_defs:
        if key not in per_layer_stats:
            continue
        fig.add_trace(
            go.Scatter(
                x=layer_labels,
                y=per_layer_stats[key],
                name=label,
                mode="lines+markers",
            )
        )
    fig.update_layout(title=title, xaxis_title="Layer", yaxis_title="Score")
    ext = plot_format if plot_format.startswith(".") else f".{plot_format}"
    out = path if path.endswith(ext) else path + ext
    pio.write_image(fig, out)


def plot_top_layers_bar(
    layer_scores: List[Tuple[int, float]],
    title: str,
    path: str,
    plot_format: str = "pdf",
    top_n: int = 15,
) -> None:
    if not _HAS_PLOTLY:
        return
    layer_scores = sorted(layer_scores, key=lambda x: -x[1])[:top_n]
    layers = [x[0] for x in layer_scores]
    scores = [x[1] for x in layer_scores]
    fig = go.Figure(go.Bar(x=[str(l) for l in layers], y=scores))
    fig.update_layout(title=title, xaxis_title="Layer", yaxis_title="Score")
    ext = plot_format if plot_format.startswith(".") else f".{plot_format}"
    out = path if path.endswith(ext) else path + ext
    pio.write_image(fig, out)


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple gender activation patching (Qwen 2.5 7B, simple prompts)")
    p.add_argument("--run-id", type=str, default="default", help="Stable run folder for resume")
    p.add_argument("--resume", action="store_true", help="Skip completed units")
    p.add_argument("--output-dir", type=str, default="patching_results", help="Base output directory")
    p.add_argument(
        "--cohorts",
        type=str,
        default=",".join(DEFAULT_COHORTS),
        help="Comma-separated cohorts (e.g. asthma,depression,multiple_sclerosis,rheumatoid_arthritis,sarcoidosis)",
    )
    p.add_argument("--prompt-ids", type=str, default="1,2,3,4", help="Comma-separated prompt template ids")
    p.add_argument(
        "--score-keys",
        type=str,
        default="rewrite_scores",
        help="Comma-separated score matrix keys to compute/use. Values: rewrite_scores,logprob_scores,logprob_delta_scores,all",
    )
    p.add_argument(
        "--model-id",
        type=int,
        default=1,
        choices=sorted(MODEL_OPTIONS.keys()),
        help="Model selector: 1=Qwen/Qwen2.5-7B-Instruct, 2=allenai/OLMo-7B-0724-Instruct-hf, 3=meta-llama/Llama-3.1-8B-Instruct",
    )
    p.add_argument("--max-tokens", type=int, default=0, help="0 = full token sweep")
    p.add_argument("--layer-start", type=int, default=0, help="First layer index (inclusive)")
    p.add_argument("--layer-end", type=int, default=9999, help="Last layer index (exclusive)")
    p.add_argument("--layer-step", type=int, default=1, help="Layer step in sweep (memory tuning)")
    p.add_argument("--top-k", type=int, default=30, help="Top-k used for top-k mean aggregation")
    p.add_argument("--trim-frac", type=float, default=0.10, help="Trim fraction for trimmed mean aggregation")
    p.add_argument("--save-heatmaps", action="store_true", help="Save token×layer heatmap per unit")
    p.add_argument(
        "--save-layer-plots",
        action="store_true",
        help="Save per-layer plots (all 4 stats) and top-layer summaries in structured subfolders",
    )
    p.add_argument("--plot-format", type=str, default="pdf", choices=["pdf", "png"])
    p.add_argument("--heatmap-mode", type=str, default="single", choices=["single", "full_suite"], help="single: one all-layer heatmap; full_suite: split layers + token windows + overview")
    p.add_argument("--heatmap-token-window", type=int, default=180, help="Token columns per heatmap tile (0 = no token split)")
    p.add_argument("--heatmap-overview-bin-size", type=int, default=10, help="Bin size for compact overview heatmap (1 disables binning)")
    p.add_argument(
        "--exclude-plot-layers",
        type=str,
        default="",
        help="Comma-separated zero-indexed layer ids to omit from plot generation only (e.g. 0,1).",
    )
    p.add_argument("--rebuild-plots-only", action="store_true", help="Regenerate plots from saved artifacts only")
    p.add_argument("--dry-run", action="store_true", help="Only validate config, work list, and progress (no model load)")
    return p.parse_args()


def get_run_dir(args: argparse.Namespace) -> Path:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / args.run_id


def build_work_list(
    args: argparse.Namespace,
    run_dir: Path,
) -> List[Tuple[str, int]]:
    """List of (cohort, prompt_id) units (simple prompts only, no BHC cases)."""
    _ = run_dir  # reserved for future run-specific filtering
    raw_cohorts = [x.strip() for x in args.cohorts.split(",") if x.strip()]
    cohorts: List[str] = []
    for raw in raw_cohorts:
        normalized = raw.lower().replace("-", "_")
        resolved = COHORT_ALIASES.get(normalized, normalized)
        cohorts.append(resolved)
    prompt_ids = [int(x.strip()) for x in args.prompt_ids.split(",") if x.strip()]
    invalid_prompt_ids = [pid for pid in prompt_ids if pid not in SIMPLE_PROMPTS]
    if invalid_prompt_ids:
        valid_ids = ",".join(str(x) for x in sorted(SIMPLE_PROMPTS.keys()))
        invalid_ids = ",".join(str(x) for x in sorted(set(invalid_prompt_ids)))
        raise ValueError(
            f"Unknown prompt id(s): {invalid_ids}. Valid prompt ids: {valid_ids}"
        )
    work: List[Tuple[str, int]] = []
    for cohort in cohorts:
        if cohort not in COHORT_TO_CONDITION_NAME:
            print(f"Warning: unknown cohort {cohort}, skipping", file=sys.stderr)
            continue
        # Prompt-major ordering: run one prompt at a time for each cohort.
        for prompt_id in prompt_ids:
            work.append((cohort, prompt_id))
    return work


def run_patching(args: argparse.Namespace, run_dir: Path) -> None:
    selected_score_keys: Tuple[str, ...] = args.selected_score_keys
    work_list = build_work_list(args, run_dir)
    progress = load_progress(run_dir)
    completed_set = set(progress.get("completed", []))
    if args.resume:
        work_list = [w for w in work_list if unit_key(w[0], w[1]) not in completed_set]
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

    cohort_to_gender = COHORT_TO_PATCH_GENDER
    condition_names = COHORT_TO_CONDITION_NAME

    for cohort, prompt_id in work_list:
        key = unit_key(cohort, prompt_id)
        try:
            gender = cohort_to_gender.get(cohort, "Male")
            condition_name = condition_names.get(cohort, cohort)
            template = SIMPLE_PROMPTS[prompt_id]
            clean_text, _, patch_token_from = build_clean_prompt(llm, gender)
            corrupted_text, corrupted_tokens = build_corrupt_prompt(
                llm, template, condition_name
            )
            target_id = int(_validated_gender_token_ids(llm, gender)[-1].item())

            sweep = run_patch_sweep(
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
                "patch_gender": gender,
                "condition_name": condition_name,
                "corrupted_prob": sweep.get("corrupted_prob", 0.0),
                "corrupted_logprob": sweep.get("corrupted_logprob", float("-inf")),
                "num_layers": num_layers,
                "num_tokens": n_t,
                "score_keys": list(selected_score_keys),
            }
            save_unit_artifact(
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
                    plot_heatmap(
                        sweep["rewrite_scores"], token_labels, layer_labels,
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
                    stats = layer_aggregates(
                        sweep[score_key], top_k=args.top_k, trim_frac=args.trim_frac
                    )
                    per_prompt_dir = plot_dir / "per_prompt" / score_key
                    per_prompt_dir.mkdir(parents=True, exist_ok=True)
                    base = per_prompt_dir / f"{cohort}_prompt{prompt_id}_{score_key}"
                    try:
                        plot_stats, plot_layer_labels = _filter_plot_stats_by_layers(
                            stats, layer_labels, args.excluded_plot_layers
                        )
                        if not plot_layer_labels:
                            print(
                                f"Plot warning (layer curves) {key} {score_key}: all layers excluded by --exclude-plot-layers",
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                        plot_layer_curves(
                            plot_stats,
                            plot_layer_labels,
                            f"{cohort} prompt{prompt_id} {score_key}",
                            str(base),
                            args.plot_format,
                            top_k=args.top_k,
                        )
                    except Exception as e:
                        print(f"Plot warning (layer curves) {key} {score_key}: {e}", file=sys.stderr, flush=True)
            mark_completed(run_dir, key, progress)
            progress = load_progress(run_dir)
            print(f"Done {key}", flush=True)
        except Exception as e:
            mark_failed(run_dir, key, str(e), progress)
            progress = load_progress(run_dir)
            print(f"Failed {key}: {e}", file=sys.stderr, flush=True)
            raise

    # Build aggregate from all artifacts
    build_aggregates(run_dir, args)


def build_aggregates(run_dir: Path, args: argparse.Namespace) -> None:
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.exists():
        return
    agg_store: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    agg_store_by_cohort: Dict[str, Dict[str, Dict[str, Dict[int, List[float]]]]] = {}
    all_raw: List[Dict[str, Any]] = []

    selected_score_keys: Tuple[str, ...] = args.selected_score_keys
    stat_keys = STAT_KEYS
    for score_key in selected_score_keys:
        agg_store[score_key] = {s: {} for s in stat_keys}

    for p in artifacts_dir.glob("*.pkl"):
        parsed = parse_artifact_stem(p.stem)
        if parsed is None:
            continue
        cohort, prompt_id = parsed

        with open(p, "rb") as f:
            import pickle
            data = pickle.load(f)

        score_matrices: Dict[str, np.ndarray] = {}
        for score_key in selected_score_keys:
            if score_key in data:
                score_matrices[score_key] = data[score_key]
        if "rewrite_scores" in selected_score_keys and "rewrite_scores" not in score_matrices and "rewrite_scores" in data:
            score_matrices["rewrite_scores"] = data["rewrite_scores"]
        if not score_matrices:
            continue

        layer_labels = data["layer_labels"]
        raw_row: Dict[str, Any] = {
            "cohort": cohort,
            "prompt_id": prompt_id,
            "metadata": data.get("metadata", {}),
            "score_stats": {},
        }
        if cohort not in agg_store_by_cohort:
            agg_store_by_cohort[cohort] = {
                score_key: {s: {} for s in stat_keys}
                for score_key in selected_score_keys
            }

        for score_key, matrix in score_matrices.items():
            stats = layer_aggregates(matrix, top_k=args.top_k, trim_frac=args.trim_frac)
            raw_row["score_stats"][score_key] = {k: v.tolist() for k, v in stats.items()}
            for i, layer in enumerate(layer_labels):
                for stat_key in stat_keys:
                    val = float(stats[stat_key][i])
                    agg_store[score_key][stat_key].setdefault(layer, []).append(val)
                    agg_store_by_cohort[cohort][score_key][stat_key].setdefault(layer, []).append(val)
        all_raw.append(raw_row)

    if not all_raw:
        return

    layer_set = set()
    for score_key in selected_score_keys:
        for stat_key in stat_keys:
            layer_set.update(agg_store[score_key][stat_key].keys())
    if not layer_set:
        return
    layers_sorted = sorted(layer_set)

    rows = []
    for layer in layers_sorted:
        row: Dict[str, Any] = {
            "layer": layer,
        }
        for score_key in selected_score_keys:
            num_units_for_score = 0
            for stat_key in stat_keys:
                vals = agg_store[score_key][stat_key].get(layer, [])
                if vals:
                    num_units_for_score = max(num_units_for_score, len(vals))
                col = f"{score_key}_{stat_key}"
                row[col] = float(np.mean(vals)) if vals else float("nan")
            row[f"{score_key}_num_units"] = num_units_for_score
        rows.append(row)

    out_csv = run_dir / "aggregate_per_layer.csv"
    with open(out_csv, "w", newline="") as f:
        import csv
        fieldnames = ["layer"]
        for score_key in selected_score_keys:
            for stat_key in stat_keys:
                fieldnames.append(f"{score_key}_{stat_key}")
            fieldnames.append(f"{score_key}_num_units")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    out_json = run_dir / "aggregate_per_layer.json"
    _atomic_write_json(str(out_json), {"per_layer": rows, "raw_units": all_raw})
    print(f"Wrote {out_csv} and {out_json}", flush=True)

    # Summary bar charts (cohort-specific), split by stat key subfolder.
    if _HAS_PLOTLY and (args.save_heatmaps or args.save_layer_plots):
        plot_dir = run_dir / "layer_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        excluded_plot_layers = set(args.excluded_plot_layers)
        stat_labels = {
            "mean": "mean",
            "median": "median",
            "trimmed_mean": "trimmed mean",
            "topk_mean": f"top-{args.top_k} mean",
        }
        for cohort, cohort_store in sorted(agg_store_by_cohort.items()):
            for score_key in selected_score_keys:
                for stat_key in stat_keys:
                    layer_scores_stat: List[Tuple[int, float]] = []
                    for layer in layers_sorted:
                        if layer in excluded_plot_layers:
                            continue
                        vals = cohort_store[score_key][stat_key].get(layer, [])
                        if vals:
                            layer_scores_stat.append((layer, float(np.mean(vals))))
                    if not layer_scores_stat:
                        continue
                    top_layers_dir = plot_dir / "top_layers" / stat_key
                    top_layers_dir.mkdir(parents=True, exist_ok=True)
                    plot_top_layers_bar(
                        layer_scores_stat,
                        f"Top layers ({cohort}) by {stat_labels.get(stat_key, stat_key)} {score_key}",
                        str(top_layers_dir / f"top_layers_{stat_key}_{cohort}_{score_key}"),
                        args.plot_format,
                    )


def rebuild_plots_only(args: argparse.Namespace, run_dir: Path) -> None:
    """Regenerate all plots from existing artifacts."""
    selected_score_keys: Tuple[str, ...] = args.selected_score_keys
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.exists():
        print("No artifacts dir found.", flush=True)
        return
    for p in sorted(artifacts_dir.glob("*.pkl")):
        name = p.stem
        parsed = parse_artifact_stem(name)
        if parsed is None:
            continue
        cohort, prompt_id = parsed
        with open(p, "rb") as f:
            import pickle
            data = pickle.load(f)
        score_matrices: Dict[str, np.ndarray] = {}
        for score_key in selected_score_keys:
            if score_key in data:
                score_matrices[score_key] = data[score_key]
        if "rewrite_scores" in selected_score_keys and "rewrite_scores" in data and "rewrite_scores" not in score_matrices:
            score_matrices["rewrite_scores"] = data["rewrite_scores"]
        if not score_matrices:
            continue
        token_labels = data["token_labels"]
        layer_labels = data["layer_labels"]
        ext = f".{args.plot_format}"
        if args.save_heatmaps and "rewrite_scores" in score_matrices:
            rs = score_matrices["rewrite_scores"]
            plot_dir = run_dir / "heatmaps"
            plot_dir.mkdir(parents=True, exist_ok=True)
            base_heatmap = plot_dir / (name + "_rewrite_scores" + ext)
            first_window = args.heatmap_token_window if args.heatmap_token_window > 0 else len(token_labels)
            first_window_end = max(0, min(len(token_labels), first_window) - 1)
            expected_first_tile = plot_dir / (name + "_rewrite_scores" + f"_part1_tok0000-{first_window_end:04d}" + ext)
            heatmap_overview = plot_dir / (name + "_rewrite_scores" + f"_overview_bin{args.heatmap_overview_bin_size}" + ext)

            needs_plot = False
            if args.heatmap_mode == "single":
                needs_plot = not base_heatmap.exists()
            else:
                needs_plot = (not expected_first_tile.exists()) or (not heatmap_overview.exists())

            if needs_plot:
                plot_heatmap(
                    rs,
                    token_labels,
                    layer_labels,
                    f"{name} rewrite_scores",
                    str(plot_dir / (name + "_rewrite_scores")),
                    args.plot_format,
                    mode=args.heatmap_mode,
                    token_window=args.heatmap_token_window,
                    overview_bin_size=args.heatmap_overview_bin_size,
                )
        if args.save_layer_plots:
            plot_dir = run_dir / "layer_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            for score_key, matrix in score_matrices.items():
                stats = layer_aggregates(matrix, top_k=args.top_k, trim_frac=args.trim_frac)
                per_prompt_dir = plot_dir / "per_prompt" / score_key
                per_prompt_dir.mkdir(parents=True, exist_ok=True)
                layer_plot_path = per_prompt_dir / (name + f"_{score_key}" + ext)
                if not layer_plot_path.exists():
                    plot_stats, plot_layer_labels = _filter_plot_stats_by_layers(
                        stats, layer_labels, args.excluded_plot_layers
                    )
                    if not plot_layer_labels:
                        print(
                            f"Plot warning (layer curves) {name} {score_key}: all layers excluded by --exclude-plot-layers",
                            file=sys.stderr,
                            flush=True,
                        )
                        continue
                    plot_layer_curves(
                        plot_stats,
                        plot_layer_labels,
                        f"{name} {score_key}",
                        str(per_prompt_dir / (name + f"_{score_key}")),
                        args.plot_format,
                        top_k=args.top_k,
                    )
    build_aggregates(run_dir, args)
    print("Rebuild plots done.", flush=True)


def main() -> None:
    args = parse_args()
    args.model_name = _resolve_model_name(args.model_id)
    args.selected_score_keys = _resolve_score_keys(args.score_keys)
    args.excluded_plot_layers = _parse_nonnegative_int_csv(
        args.exclude_plot_layers, "--exclude-plot-layers"
    )
    run_dir = get_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_plots_only:
        rebuild_plots_only(args, run_dir)
        return

    progress = load_progress(run_dir)
    if not progress.get("config_hash"):
        progress["config_hash"] = _config_hash(args)
        progress["model_name"] = args.model_name
        save_progress(run_dir, progress)

    if args.dry_run:
        work_list = build_work_list(args, run_dir)
        print(f"Dry run: run_dir={run_dir}, work_units={len(work_list)}", flush=True)
        print(f"Selected score keys: {','.join(args.selected_score_keys)}", flush=True)
        print(f"Progress: {load_progress(run_dir)}", flush=True)
        # Validate layer_aggregates
        fake = np.random.randn(4, 10).astype(np.float32)
        stats = layer_aggregates(fake, top_k=args.top_k, trim_frac=args.trim_frac)
        assert all(v.shape == (4,) for v in stats.values()), "layer_aggregates shape"
        print("Dry run OK (config, work list, progress, layer_aggregates).", flush=True)
        return

    run_patching(args, run_dir)


if __name__ == "__main__":
    main()
