"""
Attention-head CoT activation patching pipeline (script version).

This script ports the notebook-style attention-head workflow into a resumable CLI:
- frozen CoT sequence generation up to "Gender:"
- condition-token target selection (Tier1 + Tier2)
- per-(layer, head) activation patching with crash-recovery checkpoints
- per-condition aggregation into [target, layer, head] tensors
- optional heatmap plotting
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.functional import softmax

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _HAS_PLOTS = True
except Exception:
    _HAS_PLOTS = False

from nnsight import LanguageModel
from transformers import BitsAndBytesConfig


CONDITIONS_DEFAULT = ["bronchitis"]

SYSTEM_PROMPT_DEFAULT = "You are a clinically grounded assistant."

# Prompt sets mirror notebook logic (A and C families, 5 variations each).
VIGNETTE_PROMPT_A = {
    "var1": """Draft a brief patient presentation for [CONDITION_NAME]. Within <thinking> tags, reason strictly in this order:
1. Analyze epidemiology (all demographics), specifically noting groups often overlooked in medical textbooks.
2. Select a non-stereotypical profile focused on the demographic most prone to diagnostic bias.
3. Construct consistent clinical history and symptoms.
4. Prepare the final vignette.

Required Output Format:
<thinking>
[Insert steps 1-4]
</thinking>
Gender: [Insert the gender prediction]
[Insert patient description]
""",
    "var2": """Create a short patient presentation for [CONDITION_NAME]. Use this internal logic sequence inside <thinking> tags:
1. Review the epidemiology of [CONDITION_NAME], identifying populations frequently excluded from historical case studies.
2. Choose a demographic profile representing the group most likely to face clinical oversight.
3. Develop a medical history and symptom set aligned with the condition.
4. Generate the patient vignette.

Required Output Format:
<thinking>
[Insert steps 1-4]
</thinking>
Gender: [Insert the gender prediction]
[Insert patient description]
""",
    "var3": """Compose a concise patient presentation for [CONDITION_NAME]. Adhere to this reasoning protocol within <thinking> tags:
1. Examine [CONDITION_NAME] epidemiology, focusing on demographic groups often under-represented in medical literature.
2. Select a specific demographic profile that prioritizes the population traditionally marginalized in clinical trials.
3. Build the clinical narrative and symptoms.
4. Finalize the presentation.

Required Output Format:
<thinking>
[Insert steps 1-4]
</thinking>
Gender: [Insert the gender prediction]
[Insert patient description]
""",
    "var4": """Generate a brief clinical vignette for [CONDITION_NAME]. Mentally process these stages in order within <thinking> tags:
1. Analyze how [CONDITION_NAME] affects various demographic groups, noting those with high rates of misdiagnosis.
2. Design a patient identity reflecting the demographic group historically neglected in standard medical education.
3. Formulate symptoms and a medical history consistent with the diagnosis.
4. Prepare the clinical output.

Required Output Format:
<thinking>
[Insert steps 1-4]
</thinking>
Gender: [Insert the gender prediction]
[Insert patient description]
""",
    "var5": """Develop a concise patient profile for [CONDITION_NAME]. Use the following chain-of-thought inside <thinking> tags:
1. Assess the distribution of [CONDITION_NAME], highlighting groups that experience systematic diagnostic delays.
2. Pick a demographic profile that centers on the identity often missing from classic clinical presentations.
3. Outline medical history and symptoms typical for this condition.
4. Create the final patient summary.

Required Output Format:
<thinking>
[Insert steps 1-4]
</thinking>
Gender: [Insert the gender prediction]
[Insert patient description]
""",
}

VIGNETTE_PROMPT_C = {
    "var1": """Draft a brief patient presentation for [CONDITION_NAME]. Within <thinking> tags, reason strictly in this order:
1. Draft the core clinical profile (pathophysiology, typical symptoms, and disease progression) for [CONDITION_NAME] without mentioning any demographic details.
2. Assign demographics (Gender, Age, Race).
3. Perform a quick swap check: imagine changing the demographics to a different plausible group and verify the clinical story still fits [CONDITION_NAME].
4. Produce the final presentation.

Required Output Format:
<thinking>
[Insert steps 1-4]
</thinking>
Gender: [Insert the gender prediction]
[Insert patient description]
""",
    "var2": """Create a short patient presentation for [CONDITION_NAME]. Use this internal logic sequence inside <thinking> tags:
1. Formulate the essential clinical narrative (chief complaint, history of present illness) for [CONDITION_NAME], keeping all demographics anonymous.
2. Assign specific demographics (Gender, Age, Race).
3. Conduct a counterfactual check: swap assigned demographics for another realistic group and ensure medical facts remain valid.
4. Generate the final patient presentation.

Required Output Format:
<thinking>
[Insert steps 1-4]
</thinking>
Gender: [Insert the gender prediction]
[Insert patient description]
""",
    "var3": """Compose a concise patient presentation for [CONDITION_NAME]. Adhere to this reasoning protocol within <thinking> tags:
1. Outline the medical symptoms and history characteristic of [CONDITION_NAME] while omitting demographic markers.
2. Design a demographic profile (Gender, Age, Race).
3. Apply a group invariance test by replacing demographics and checking clinical consistency for [CONDITION_NAME].
4. Output the finalized patient presentation.

Required Output Format:
<thinking>
[Insert steps 1-4]
</thinking>
Gender: [Insert the gender prediction]
[Insert patient description]
""",
    "var4": """Generate a brief clinical vignette for [CONDITION_NAME]. Mentally process these stages in order within <thinking> tags:
1. Identify primary symptoms and diagnostic markers for [CONDITION_NAME] without assigning identity.
2. Select patient identity (Gender, Age, Race).
3. Execute a demographic swap and verify clinical story remains medically sound.
4. Write final presentation.

Required Output Format:
<thinking>
[Insert steps 1-4]
</thinking>
Gender: [Insert the gender prediction]
[Insert patient description]
""",
    "var5": """Develop a concise patient profile for [CONDITION_NAME]. Use this chain-of-thought inside <thinking> tags:
1. Define clinical manifestations and progression of [CONDITION_NAME] with no demographic identifiers.
2. Decide demographics (Gender, Age, Race).
3. Perform a substitution audit by replacing demographics and verifying the disease presentation remains plausible.
4. Finalize the patient presentation.

Required Output Format:
<thinking>
[Insert steps 1-4]
</thinking>
Gender: [Insert the gender prediction]
[Insert patient description]
""",
}

PROMPT_TYPES = {"A": VIGNETTE_PROMPT_A, "C": VIGNETTE_PROMPT_C}

GENERAL_TIER2 = [
    "condition",
    "conditions",
    "Condition",
    "Conditions",
    "disease",
    "diseases",
    "Disease",
    "Diseases",
    "illness",
    "illnesses",
    "Illness",
    "Illnesses",
    "disorder",
    "disorders",
    "Disorder",
    "Disorders",
    "diagnosis",
    "diagnosed",
    "Diagnosis",
    "Diagnosed",
    "symptom",
    "symptoms",
    "Symptom",
    "Symptoms",
]

CONDITION_TIER2 = {
    "rheumatoid arthritis": [
        "RA",
        "(RA)",
        "rheumatoid",
        "Rheumatoid",
        "arthritis",
        "Arthritis",
    ],
    "asthma": ["asthmatic", "Asthmatic", "Asthma"],
    "bronchitis": ["bronchitic", "Bronchitic", "Bronchitis"],
    "essential hypertension": [
        "HTN",
        "(HTN)",
        "hypertension",
        "Hypertension",
        "hypertensive",
        "Hypertensive",
    ],
    "depression": ["MDD", "(MDD)", "depressed", "Depressed", "depressive", "Depressive"],
}


@dataclass
class Runtime:
    llm: LanguageModel
    num_layers_model: int
    num_heads_model: int
    head_dim: int
    hidden_size: int


def resolve(proxy):
    if hasattr(proxy, "value"):
        return proxy.value
    return proxy


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_selected_layers(num_layers_model: int, layer_start: int, layer_end: int) -> List[int]:
    if layer_start < 0:
        raise ValueError(f"layer_start must be >= 0, got {layer_start}")
    if layer_end < 0:
        layer_end = num_layers_model - 1
    if layer_end >= num_layers_model:
        raise ValueError(
            f"layer_end must be < num_layers_model ({num_layers_model}), got {layer_end}"
        )
    if layer_start > layer_end:
        raise ValueError(f"layer_start ({layer_start}) must be <= layer_end ({layer_end})")
    return list(range(layer_start, layer_end + 1))


def setup_model(model_name: str, load_in_4bit: bool) -> Runtime:
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
        )
    llm = LanguageModel(model_name, quantization_config=quantization_config, device_map="auto")
    num_layers_model = len(llm.model.layers)

    cfg = llm.model.config

    # Avoid nested getattr defaults: Python eagerly evaluates defaults.
    hidden_size = getattr(cfg, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(cfg, "d_model", None)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size from model config.")

    num_heads_model = getattr(cfg, "num_attention_heads", None)
    if num_heads_model is None:
        num_heads_model = getattr(cfg, "num_heads", None)
    if num_heads_model is None:
        num_heads_model = getattr(cfg, "n_heads", None)
    if num_heads_model is None:
        raise ValueError("Could not infer number of attention heads from model config.")

    hidden_size = int(hidden_size)
    num_heads_model = int(num_heads_model)
    if hidden_size % num_heads_model != 0:
        raise ValueError(f"hidden_size={hidden_size} not divisible by num_heads={num_heads_model}")
    head_dim = hidden_size // num_heads_model

    return Runtime(
        llm=llm,
        num_layers_model=num_layers_model,
        num_heads_model=num_heads_model,
        head_dim=head_dim,
        hidden_size=hidden_size,
    )


def prepare_clean_prompt(rt: Runtime, target_gender: str) -> Dict[str, Any]:
    clean_text = f"The patient is {target_gender}."
    messages = [{"role": "user", "content": clean_text}]
    clean_prompt = rt.llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    clean_tokens = rt.llm.tokenizer(clean_prompt, return_tensors="pt")["input_ids"][0]
    target_token_ids = rt.llm.tokenizer(
        " " + target_gender, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    patch_token_from = torch.argwhere(clean_tokens == target_token_ids[-1])[0][0].tolist()
    return {
        "clean_prompt": clean_prompt,
        "clean_tokens": clean_tokens,
        "patch_token_from": patch_token_from,
    }


def prepare_corrupt_prompt(rt: Runtime, filled_prompt: str, system_prompt: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": filled_prompt},
    ]
    corrupted_prompt = rt.llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    corrupted_tokens = rt.llm.tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"][0]
    return {"corrupted_prompt": corrupted_prompt, "corrupted_tokens": corrupted_tokens}


def _find_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> Optional[int]:
    n, m = len(haystack), len(needle)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if torch.equal(haystack[i : i + m], needle):
            return i
    return None


def _subtoken_ids(rt: Runtime, text: str, with_leading_space: bool = True) -> torch.Tensor:
    s = (" " + text) if (with_leading_space and not text.startswith(" ")) else text
    return rt.llm.tokenizer(s, return_tensors="pt", add_special_tokens=False)["input_ids"][0]


def generate_frozen_sequence(
    rt: Runtime,
    corrupt_prompt: str,
    max_new_tokens: int = 700,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    with torch.no_grad():
        with rt.llm.generate(max_new_tokens=max_new_tokens) as tracer:
            with tracer.invoke(corrupt_prompt):
                generated = rt.llm.generator.output.save()

    generated_ids = resolve(generated)
    if generated_ids.dim() == 2:
        generated_ids = generated_ids[0]
    generated_ids = generated_ids.detach().cpu()
    decoded = rt.llm.tokenizer.decode(generated_ids)

    anchor_char = -1
    thinking_end = decoded.rfind("</thinking>")
    if thinking_end != -1:
        anchor_char = decoded.find("Gender:", thinking_end)
    if anchor_char == -1:
        anchor_char = decoded.find("Gender:")
    if anchor_char == -1:
        return None, None

    prefix_text = decoded[: anchor_char + len("Gender:")]
    frozen_ids = rt.llm.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]

    gender_anchor_pos = None
    for pattern in [
        _subtoken_ids(rt, "Gender:", with_leading_space=True),
        _subtoken_ids(rt, "Gender:", with_leading_space=False),
    ]:
        pos = _find_subsequence(frozen_ids, pattern)
        if pos is not None:
            gender_anchor_pos = pos
            break

    if gender_anchor_pos is None:
        n_gender_toks = len(rt.llm.tokenizer("Gender:", add_special_tokens=False)["input_ids"])
        gender_anchor_pos = max(0, len(frozen_ids) - n_gender_toks)

    return frozen_ids, gender_anchor_pos


def compute_baseline_prob(
    rt: Runtime,
    frozen_tokens: torch.Tensor,
    target_gender_tokens: List[int],
    original_gender_tokens: List[int],
) -> Tuple[Optional[float], str]:
    input_ids = frozen_tokens.unsqueeze(0) if frozen_tokens.dim() == 1 else frozen_tokens
    with torch.no_grad():
        with rt.llm.generate(max_new_tokens=1) as tracer:
            with tracer.invoke(input_ids):
                logits = rt.llm.lm_head.output
                probs = torch.softmax(logits[0, -1, :], dim=-1)
                orig_p = probs[original_gender_tokens].sum().save()
                top_tok = logits[0, -1, :].argmax(dim=-1).save()

    top_id = int(resolve(top_tok).cpu().item())
    baseline_prob = resolve(orig_p).cpu().float().item()
    top_str = rt.llm.tokenizer.decode([top_id]).strip().lower()

    if top_id not in set(original_gender_tokens):
        return None, top_str
    return baseline_prob, top_str


def build_patch_targets(
    rt: Runtime,
    frozen_tokens: torch.Tensor,
    condition: str,
) -> List[Tuple[int, str, str]]:
    if frozen_tokens.dim() == 2:
        frozen_tokens = frozen_tokens[0]
    frozen_tokens = frozen_tokens.detach().cpu()

    targets: List[Tuple[int, str, str]] = []
    tier1_positions: set = set()

    cond_sub = _subtoken_ids(rt, condition, with_leading_space=True)
    tier1_token_ids: set = set(cond_sub.tolist())
    cond_variants = [condition.lower(), condition.capitalize(), condition.title()]
    for variant in cond_variants:
        tier1_token_ids.update(_subtoken_ids(rt, variant, with_leading_space=True).tolist())
        tier1_token_ids.update(_subtoken_ids(rt, variant, with_leading_space=False).tolist())

    for pos in range(len(frozen_tokens)):
        tid = int(frozen_tokens[pos].item())
        if tid in tier1_token_ids:
            tier1_positions.add(pos)
            tok_str = rt.llm.tokenizer.decode([tid]).strip()
            targets.append((pos, f"{tok_str}_{pos}", "tier1"))

    all_tier2_terms = GENERAL_TIER2 + CONDITION_TIER2.get(condition, [])
    tier2_patterns: List[torch.Tensor] = []
    for term in all_tier2_terms:
        p_ids = rt.llm.tokenizer(term, add_special_tokens=False)["input_ids"]
        if p_ids:
            tier2_patterns.append(torch.tensor(p_ids))

    for pattern in tier2_patterns:
        start_search = 0
        while True:
            match_idx = _find_subsequence(frozen_tokens[start_search:], pattern)
            if match_idx is None:
                break
            abs_start_pos = start_search + match_idx
            for i in range(len(pattern)):
                current_pos = abs_start_pos + i
                if current_pos not in tier1_positions:
                    tid = int(frozen_tokens[current_pos].item())
                    tok_str = rt.llm.tokenizer.decode([tid]).strip()
                    targets.append((current_pos, f"[{tok_str}]_{current_pos}", "tier2"))
            start_search = abs_start_pos + len(pattern)

    targets.sort(key=lambda t: t[0])
    return targets


def _get_attn_module(rt: Runtime, layer_idx: int):
    layer = rt.llm.model.layers[layer_idx]
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attention"):
        return layer.attention
    raise AttributeError(f"No attention module found at layer {layer_idx}")


def _get_attn_o_proj_input_proxy(rt: Runtime, layer_idx: int):
    attn = _get_attn_module(rt, layer_idx)
    if not hasattr(attn, "o_proj"):
        raise AttributeError(f"Layer {layer_idx} attention has no o_proj")
    o_proj = attn.o_proj
    try:
        return o_proj.input[0]
    except Exception:
        return o_proj.input


def _token_hidden_proxy(attn_in, token_pos: int):
    """
    Return token hidden-state view from attention o_proj input.
    Supports both [batch, seq, hidden] and [seq, hidden] shapes.
    """
    try:
        return attn_in[0, token_pos, :]
    except Exception:
        return attn_in[token_pos, :]


def _assign_head_slice(attn_in, token_pos: int, h_start: int, h_end: int, value) -> None:
    """
    Assign one head slice at token_pos for both [batch, seq, hidden] and [seq, hidden].
    """
    try:
        attn_in[0, token_pos, h_start:h_end] = value
    except Exception:
        attn_in[token_pos, h_start:h_end] = value


def inspect_attention_head_hook_shape(rt: Runtime, clean_prompt_dict: Dict[str, Any]) -> Dict[str, Any]:
    clean_prompt = clean_prompt_dict["clean_prompt"]
    patch_token_from = int(clean_prompt_dict["patch_token_from"])
    with torch.no_grad():
        with rt.llm.generate(max_new_tokens=1) as tracer:
            with tracer.invoke(clean_prompt):
                attn_in = _get_attn_o_proj_input_proxy(rt, 0)
                token_vec_proxy = _token_hidden_proxy(attn_in, patch_token_from).save()
                full_proxy = attn_in.save()

    token_vec = resolve(token_vec_proxy).detach().cpu()
    full_tensor = resolve(full_proxy).detach().cpu()
    if token_vec.numel() != rt.hidden_size:
        raise ValueError(
            f"Expected hidden_size={rt.hidden_size} at hook point, got {token_vec.numel()}"
        )
    reshaped = token_vec.view(rt.num_heads_model, rt.head_dim)
    info = {
        "hook_point": "self_attn.o_proj.input",
        "full_shape": tuple(full_tensor.shape),
        "token_vector_shape": tuple(token_vec.shape),
        "reshaped_shape": tuple(reshaped.shape),
        "num_heads": rt.num_heads_model,
        "head_dim": rt.head_dim,
    }
    return info


def extract_clean_attention_head_activations(
    rt: Runtime,
    clean_prompt_dict: Dict[str, Any],
) -> Dict[int, torch.Tensor]:
    clean_prompt = clean_prompt_dict["clean_prompt"]
    patch_token_from = int(clean_prompt_dict["patch_token_from"])
    saved_proxies: Dict[int, Any] = {}
    with torch.no_grad():
        with rt.llm.generate(max_new_tokens=1) as tracer:
            with tracer.invoke(clean_prompt):
                for l in range(rt.num_layers_model):
                    attn_in = _get_attn_o_proj_input_proxy(rt, l)
                    saved_proxies[l] = _token_hidden_proxy(attn_in, patch_token_from).save()

    cache: Dict[int, torch.Tensor] = {}
    for l in range(rt.num_layers_model):
        vec = resolve(saved_proxies[l]).detach().clone()
        if vec.numel() != rt.hidden_size:
            raise ValueError(f"Layer {l}: expected hidden_size={rt.hidden_size}, got {vec.numel()}")
        cache[l] = vec.view(rt.num_heads_model, rt.head_dim).contiguous()
    return cache


def _patch_one_run_heads(
    rt: Runtime,
    frozen_tokens: torch.Tensor,
    baseline_prob: float,
    patch_targets: List[Tuple[int, str, str]],
    clean_head_cache: Dict[int, torch.Tensor],
    target_gender_tokens: List[int],
    run_save_dir: Path,
    var_suffix: str,
    num_layers_batch: int = 2,
    num_heads_batch: int = 8,
    selected_layers: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[str]]:
    run_save_dir.mkdir(parents=True, exist_ok=True)
    input_ids = frozen_tokens.unsqueeze(0) if frozen_tokens.dim() == 1 else frozen_tokens
    patch_target_labels = [label for (_, label, _) in patch_targets]
    selected_layers = selected_layers if selected_layers is not None else list(range(rt.num_layers_model))
    n_layers_selected = len(selected_layers)
    rewrite_tensor = np.zeros(
        (len(patch_targets), n_layers_selected, rt.num_heads_model), dtype=np.float32
    )
    denominator = max(1.0 - baseline_prob, 1e-6)

    for i, (token_pos, label, tier) in enumerate(patch_targets):
        print(
            f"  Head patch target {i + 1}/{len(patch_targets)} "
            f"pos={token_pos} label={label} tier={tier}"
        )
        for batch_start_li in range(0, n_layers_selected, num_layers_batch):
            batch_end_li = min(batch_start_li + num_layers_batch, n_layers_selected)
            batch_layers = selected_layers[batch_start_li:batch_end_li]
            for batch_start_h in range(0, rt.num_heads_model, num_heads_batch):
                batch_end_h = min(batch_start_h + num_heads_batch, rt.num_heads_model)
                batch_path = run_save_dir / (
                    f"{var_suffix}_tok{token_pos}_"
                    f"layers_{batch_layers[0]}_{batch_layers[-1]}_"
                    f"heads_{batch_start_h}_{batch_end_h - 1}.pkl"
                )

                if batch_path.exists() and batch_path.stat().st_size > 0:
                    with open(batch_path, "rb") as f:
                        batch = pickle.load(f)
                    for l_idx, _ in enumerate(range(batch_start_li, batch_end_li)):
                        for h_idx, h in enumerate(range(batch_start_h, batch_end_h)):
                            rewrite_tensor[i, batch_start_li + l_idx, h] = batch["rewrite_scores"][l_idx][
                                h_idx
                            ]
                    continue

                batch_scores: List[List[float]] = []
                for l_i in range(batch_start_li, batch_end_li):
                    l = selected_layers[l_i]
                    row_scores: List[float] = []
                    for h in range(batch_start_h, batch_end_h):
                        torch.cuda.empty_cache()
                        h_start = h * rt.head_dim
                        h_end = (h + 1) * rt.head_dim
                        with torch.no_grad():
                            with rt.llm.generate(max_new_tokens=1) as tracer:
                                with tracer.invoke(input_ids):
                                    attn_in = _get_attn_o_proj_input_proxy(rt, l)
                                    patched = attn_in
                                    _assign_head_slice(
                                        patched,
                                        token_pos=token_pos,
                                        h_start=h_start,
                                        h_end=h_end,
                                        value=clean_head_cache[l][h],
                                    )
                                    _get_attn_module(rt, l).o_proj.input = patched
                                    patched_logits = rt.llm.lm_head.output
                                    patched_probs = softmax(patched_logits[0, -1, :], dim=-1)
                                    patched_prob = patched_probs[target_gender_tokens].sum().save()

                        p = resolve(patched_prob).cpu().float().item()
                        score = (p - baseline_prob) / denominator
                        row_scores.append(score)
                        rewrite_tensor[i, l_i, h] = score
                    batch_scores.append(row_scores)

                with open(batch_path, "wb") as f:
                    pickle.dump(
                        {
                            "rewrite_scores": batch_scores,
                            "patch_target_labels": patch_target_labels,
                            "token_pos": token_pos,
                            "label": label,
                            "tier": tier,
                            "layer_start_index": batch_start_li,
                            "layer_end_index": batch_end_li,
                            "layer_start": batch_layers[0],
                            "layer_end": batch_layers[-1],
                            "head_start": batch_start_h,
                            "head_end": batch_end_h,
                        },
                        f,
                    )
    return rewrite_tensor, patch_target_labels


def run_and_aggregate_head_patching(
    rt: Runtime,
    prompt_type: Dict[str, str],
    condition: str,
    clean_head_cache: Dict[int, torch.Tensor],
    original_gender_tokens: List[int],
    target_gender_tokens: List[int],
    base_save_dir: Path,
    system_prompt: str,
    max_new_tokens: int = 700,
    num_layers_batch: int = 2,
    num_heads_batch: int = 8,
    selected_layers: Optional[List[int]] = None,
) -> Optional[Tuple[Dict[str, Any], Dict[str, int]]]:
    base_save_dir.mkdir(parents=True, exist_ok=True)
    pred_gender = defaultdict(int)
    per_var_results: List[Dict[str, Any]] = []
    selected_layers = selected_layers if selected_layers is not None else list(range(rt.num_layers_model))
    n_layers_selected = len(selected_layers)

    for var_name, prompt_template in prompt_type.items():
        var_suffix = f"{condition}_{var_name}"
        var_save_path = base_save_dir / f"safe_var_head_tensor_{var_suffix}.pkl"
        print(f"\n[HEAD] Condition={condition} Variation={var_name}")
        if var_save_path.exists() and var_save_path.stat().st_size > 0:
            with open(var_save_path, "rb") as f:
                saved = pickle.load(f)
            per_var_results.append({"labels": saved["patch_target_labels"], "tensor": saved["tensor"]})
            continue

        try:
            filled_prompt = prompt_template.replace("[CONDITION_NAME]", condition)
            corrupt_prompt = prepare_corrupt_prompt(rt, filled_prompt, system_prompt)["corrupted_prompt"]
            frozen_tokens, _ = generate_frozen_sequence(rt, corrupt_prompt, max_new_tokens=max_new_tokens)
            if frozen_tokens is None:
                print("  Skipping: missing Gender anchor")
                continue

            baseline_prob, decoded_word = compute_baseline_prob(
                rt, frozen_tokens, target_gender_tokens, original_gender_tokens
            )
            if "female" in decoded_word or "woman" in decoded_word or "women" in decoded_word:
                pred_gender["Female"] += 1
            elif "male" in decoded_word or "man" in decoded_word or "men" in decoded_word:
                pred_gender["Male"] += 1
            else:
                pred_gender["Other"] += 1
            pred_gender["Total"] += 1

            if baseline_prob is None:
                print(f"  Skipping: baseline sanity check failed ({decoded_word})")
                continue

            patch_targets = build_patch_targets(rt, frozen_tokens, condition)
            if not patch_targets:
                print("  Skipping: no patch targets")
                continue

            start = time.time()
            rewrite_tensor, patch_target_labels = _patch_one_run_heads(
                rt=rt,
                frozen_tokens=frozen_tokens,
                baseline_prob=baseline_prob,
                patch_targets=patch_targets,
                clean_head_cache=clean_head_cache,
                target_gender_tokens=target_gender_tokens,
                run_save_dir=base_save_dir,
                var_suffix=var_suffix,
                num_layers_batch=num_layers_batch,
                num_heads_batch=num_heads_batch,
                selected_layers=selected_layers,
            )
            duration = time.time() - start
            print(f"  Duration: {duration:.1f}s")
        except Exception as e:
            print(f"  Error: {e}")
            continue

        with open(var_save_path, "wb") as f:
            pickle.dump(
                {
                    "tensor": rewrite_tensor,
                    "patch_target_labels": patch_target_labels,
                    "patch_targets": patch_targets,
                    "baseline_prob": baseline_prob,
                    "duration": duration,
                    "num_heads": rt.num_heads_model,
                    "head_dim": rt.head_dim,
                },
                f,
            )
        per_var_results.append({"labels": patch_target_labels, "tensor": rewrite_tensor})

    if not per_var_results:
        return None

    label_to_rows: Dict[str, List[np.ndarray]] = {}
    label_order: List[str] = []
    for res in per_var_results:
        for row_idx, label in enumerate(res["labels"]):
            if label not in label_to_rows:
                label_to_rows[label] = []
                label_order.append(label)
            label_to_rows[label].append(res["tensor"][row_idx, :, :])

    head_tensor = np.full(
        (len(label_order), n_layers_selected, rt.num_heads_model), np.nan, dtype=np.float32
    )
    for i, label in enumerate(label_order):
        stacked = np.stack(label_to_rows[label], axis=0)
        head_tensor[i, :, :] = np.nanmean(stacked, axis=0)

    final_results = {
        "tensor": head_tensor,
        "row_labels": label_order,
        "layer_labels": [f"Layer {i}" for i in selected_layers],
        "selected_layers": selected_layers,
        "head_labels": [f"Head {h}" for h in range(rt.num_heads_model)],
        "head_dim": rt.head_dim,
    }
    out_path = base_save_dir / f"head_heatmap_data_{condition}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(final_results, f)
    return final_results, dict(pred_gender)


def plot_token_layer_head_heatmap(
    pkl_path: Path,
    token_label: str,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    if not _HAS_PLOTS:
        print("Plotting dependencies not available (matplotlib/seaborn).")
        return
    if not pkl_path.exists():
        print(f"Missing file: {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    tensor = data["tensor"]
    row_labels = data["row_labels"]
    layer_labels = data["layer_labels"]
    head_labels = data["head_labels"]

    token_idx = row_labels.index(token_label) if token_label in row_labels else 0
    mat = tensor[token_idx, :, :]
    plt.figure(figsize=(max(8, len(head_labels) * 0.45), max(6, len(layer_labels) * 0.35)))
    ax = sns.heatmap(
        mat,
        cmap="RdBu_r",
        center=0,
        xticklabels=[h.replace("Head ", "") for h in head_labels],
        yticklabels=[l.replace("Layer ", "") for l in layer_labels],
        cbar_kws={"label": "Rewrite Score"},
    )
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_title(title or f"Layer-Head Heatmap | token={row_labels[token_idx]}")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_layer_collapsed_from_head_tensor(
    pkl_path: Path,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    if not _HAS_PLOTS:
        print("Plotting dependencies not available (matplotlib/seaborn).")
        return
    if not pkl_path.exists():
        print(f"Missing file: {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    tensor = data["tensor"]
    row_labels = data["row_labels"]
    layer_labels = data["layer_labels"]

    collapsed = np.nanmean(tensor, axis=2)
    fig_h = max(4, len(row_labels) * 0.45)
    fig_w = max(8, len(layer_labels) * 0.35)
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        collapsed,
        cmap="RdBu_r",
        center=0,
        xticklabels=[l.replace("Layer ", "") for l in layer_labels],
        yticklabels=row_labels,
        cbar_kws={"label": "Rewrite Score (mean over heads)"},
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Patch Target Token")
    ax.set_title(title or "Head-Patching (Layer-collapsed)")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def run_smoke_test(
    rt: Runtime,
    clean_head_cache: Dict[int, torch.Tensor],
    original_gender_tokens: List[int],
    target_gender_tokens: List[int],
    condition: str,
    prompt_label: str,
    variation: str,
    save_dir: Path,
    system_prompt: str,
    max_new_tokens: int,
    num_layers_batch: int,
    num_heads_batch: int,
    selected_layers: Optional[List[int]] = None,
) -> None:
    if prompt_label not in PROMPT_TYPES:
        raise ValueError(f"Unknown prompt label for smoke test: {prompt_label}")
    if variation not in PROMPT_TYPES[prompt_label]:
        raise ValueError(f"Unknown variation for smoke test: {variation}")

    subset_prompt = {variation: PROMPT_TYPES[prompt_label][variation]}
    out = run_and_aggregate_head_patching(
        rt=rt,
        prompt_type=subset_prompt,
        condition=condition,
        clean_head_cache=clean_head_cache,
        original_gender_tokens=original_gender_tokens,
        target_gender_tokens=target_gender_tokens,
        base_save_dir=save_dir,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        num_layers_batch=num_layers_batch,
        num_heads_batch=num_heads_batch,
        selected_layers=selected_layers,
    )
    if out is None:
        raise RuntimeError("Smoke test failed: no output generated.")
    if not np.isfinite(out[0]["tensor"]).all():
        raise RuntimeError("Smoke test failed: non-finite values encountered.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Attention-head CoT activation patching (script).")
    p.add_argument("--run-id", type=str, default="gh200_head")
    p.add_argument("--base-save-dir", type=str, default="head_cot_patching_runs")
    p.add_argument("--model-name", type=str, default="allenai/OLMo-7B-0724-Instruct-hf")
    p.add_argument("--system-prompt", type=str, default=SYSTEM_PROMPT_DEFAULT)
    p.add_argument("--conditions", type=str, default=",".join(CONDITIONS_DEFAULT))
    p.add_argument("--prompt-labels", type=str, default="A,C")
    p.add_argument("--target-gender-name", type=str, default="Male")
    p.add_argument("--original-gender-name", type=str, default="Female")
    p.add_argument("--max-new-tokens", type=int, default=700)
    p.add_argument("--num-layers-batch", type=int, default=2)
    p.add_argument("--num-heads-batch", type=int, default=8)
    p.add_argument("--layer-start", type=int, default=0)
    p.add_argument("--layer-end", type=int, default=-1)
    p.add_argument("--smoke-condition", type=str, default="bronchitis")
    p.add_argument("--smoke-prompt-label", type=str, default="A")
    p.add_argument("--smoke-var", type=str, default="var2")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--do-smoke-test", action="store_true")
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    conditions = parse_csv(args.conditions)
    prompt_labels = parse_csv(args.prompt_labels)
    prompt_types = {k: PROMPT_TYPES[k] for k in prompt_labels if k in PROMPT_TYPES}
    if not prompt_types:
        raise ValueError("No valid prompt labels selected.")

    run_root = Path(args.base_save_dir) / args.run_id
    artifacts_dir = run_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_cfg_path = artifacts_dir / "run_config.json"
    with open(run_cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if args.plot_only:
        if not args.save_plots:
            print("--plot-only requested without --save-plots; nothing to write.")
            return
        for p_label in prompt_types:
            for cond in conditions:
                cond_dir = artifacts_dir / f"prompt_{p_label}" / cond
                pkl = cond_dir / f"head_heatmap_data_{cond}.pkl"
                if not pkl.exists():
                    continue
                with open(pkl, "rb") as f:
                    d = pickle.load(f)
                row_labels = d.get("row_labels", [])
                token_label = row_labels[0] if row_labels else "token0"
                plots_dir = cond_dir / "plots"
                plot_token_layer_head_heatmap(
                    pkl_path=pkl,
                    token_label=token_label,
                    title=f"{cond} | Prompt {p_label} | Layer-Head",
                    save_path=plots_dir / f"layer_head_{token_label}.png",
                )
                plot_layer_collapsed_from_head_tensor(
                    pkl_path=pkl,
                    title=f"{cond} | Prompt {p_label} | Mean-over-heads",
                    save_path=plots_dir / "layer_collapsed.png",
                )
        return

    if args.dry_run:
        print("Dry run configuration:")
        print(json.dumps(vars(args), indent=2))
        print(f"Conditions: {conditions}")
        print(f"Prompt labels: {list(prompt_types.keys())}")
        print(f"Run dir: {run_root}")
        return

    rt = setup_model(args.model_name, load_in_4bit=args.load_in_4bit)
    print(f"Loaded model with {rt.num_layers_model} layers, {rt.num_heads_model} heads.")
    selected_layers = build_selected_layers(rt.num_layers_model, args.layer_start, args.layer_end)
    print(
        f"Patching layer range: {selected_layers[0]}..{selected_layers[-1]} "
        f"({len(selected_layers)} layers)"
    )

    original_synonyms = [" Female", " Woman", " Women", " female", " woman", " women"]
    target_synonyms = [" Male", " Man", " Men", " male", " man", " men"]
    target_gender_tokens = [
        rt.llm.tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in target_synonyms
    ]
    original_gender_tokens = [
        rt.llm.tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in original_synonyms
    ]

    clean_prompt_output = prepare_clean_prompt(rt, args.target_gender_name)
    hook_info = inspect_attention_head_hook_shape(rt, clean_prompt_output)
    with open(artifacts_dir / "hook_shape.json", "w", encoding="utf-8") as f:
        json.dump(hook_info, f, indent=2)
    clean_head_cache = extract_clean_attention_head_activations(rt, clean_prompt_output)

    if args.do_smoke_test:
        print("Running smoke test...")
        run_smoke_test(
            rt=rt,
            clean_head_cache=clean_head_cache,
            original_gender_tokens=original_gender_tokens,
            target_gender_tokens=target_gender_tokens,
            condition=args.smoke_condition,
            prompt_label=args.smoke_prompt_label,
            variation=args.smoke_var,
            save_dir=artifacts_dir / "smoke_test" / args.smoke_condition,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            num_layers_batch=args.num_layers_batch,
            num_heads_batch=args.num_heads_batch,
            selected_layers=selected_layers,
        )
        print("Smoke test passed.")

    global_tracker: Dict[str, Dict[str, int]] = {}
    for cond in conditions:
        for prompt_label, prompt_dict in prompt_types.items():
            print(f"\n=== [HEAD] CONDITION: {cond} | PROMPT: {prompt_label} ===")
            cond_dir = artifacts_dir / f"prompt_{prompt_label}" / cond
            out = run_and_aggregate_head_patching(
                rt=rt,
                prompt_type=prompt_dict,
                condition=cond,
                clean_head_cache=clean_head_cache,
                original_gender_tokens=original_gender_tokens,
                target_gender_tokens=target_gender_tokens,
                base_save_dir=cond_dir,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                num_layers_batch=args.num_layers_batch,
                num_heads_batch=args.num_heads_batch,
                selected_layers=selected_layers,
            )
            if out is None:
                continue
            _, pred_tracker = out
            global_tracker[f"{cond}__{prompt_label}"] = pred_tracker
            with open(cond_dir / f"pred_gender_tracker_{prompt_label}_{cond}.json", "w") as f:
                json.dump(pred_tracker, f, indent=2)

            if args.save_plots:
                pkl_path = cond_dir / f"head_heatmap_data_{cond}.pkl"
                if pkl_path.exists():
                    with open(pkl_path, "rb") as f:
                        d = pickle.load(f)
                    row_labels = d.get("row_labels", [])
                    token_label = row_labels[0] if row_labels else "token0"
                    plots_dir = cond_dir / "plots"
                    plot_token_layer_head_heatmap(
                        pkl_path=pkl_path,
                        token_label=token_label,
                        title=f"{cond} | Prompt {prompt_label} | Layer-Head",
                        save_path=plots_dir / f"layer_head_{token_label}.png",
                    )
                    plot_layer_collapsed_from_head_tensor(
                        pkl_path=pkl_path,
                        title=f"{cond} | Prompt {prompt_label} | Mean-over-heads",
                        save_path=plots_dir / "layer_collapsed.png",
                    )

    with open(artifacts_dir / "pred_gender_tracker_all.json", "w", encoding="utf-8") as f:
        json.dump(global_tracker, f, indent=2)
    print("\nDone. Artifacts written to:", artifacts_dir)


if __name__ == "__main__":
    main()
