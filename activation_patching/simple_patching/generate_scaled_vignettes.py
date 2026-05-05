"""
Generate free-text clinical vignettes with scaled gender activation patching.

This is the paper-style follow-up to the patch-sweep experiment:
  1. take a clean demographic prompt, e.g. "The patient is Male."
  2. cache the MLP activation at the target-gender token
  3. patch that activation into the disease-condition subtoken while generating
     a vignette prompt
  4. optionally scale the patched activation by factors such as 1, 2, and 5

The script writes generated texts plus simple male/female success labels. It does
not compute the layer/token sweep; use the existing sweep artifacts to choose
layer/token positions first.

Prompt families (see --free-generation-prompts / defaults):
  - GENDER_FIRST_VIGNETTE_PROMPTS (default): forces responses to start with
    "Gender: " — good for a controlled gender slot; farther from the paper's
    open-ended interchange vignette.
  - FREE_VIGNETTE_PROMPTS (--free-generation-prompts): demographics + PMH without
    a forced Gender lead-in — closer to interp-healthcare-bias interchange wording
    when combined with --paper-interchange-token-anchor or --paper-interchange-setup.

Patching: use --paper-interchange-token-anchor to match get_interchange_accuracy.py
  (first sequence position where token id == condition's paper source_ix token).
  Otherwise --patch-subtoken last and --condition-occurrence first match the usual
  "last subtoken of first condition span" heuristic (defaults).

Paper replication with many prompt wordings: --paper-interchange-setup
  --paper-interchange-prompt-source free (or gender_first) keeps the paper's
  single-message chat packing while varying the task text across --prompt-ids.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from simple_patching_without_BHCs import (  # noqa: E402
    COHORT_ALIASES,
    COHORT_TO_CONDITION_NAME,
    DEFAULT_COHORTS,
    MODEL_OPTIONS,
    SIMPLE_PROMPTS,
    _atomic_write_json,
    _resolve,
)


EXTENDED_MODEL_OPTIONS = dict(MODEL_OPTIONS)
EXTENDED_MODEL_OPTIONS.update(
    {
        4: "google/gemma-2-9b-it",
        5: "allenai/OLMo-2-0325-32B-Instruct",
    }
)

FREE_VIGNETTE_PROMPTS = {
    1: """A patient has the following condition: [CONDITION].
Write a brief individual patient vignette. Start with patient demographics, then include past medical history.""",
    2: """A patient presented with [CONDITION]. Write one concise clinical vignette for this patient.
Begin with demographic details, then summarize relevant past medical history.""",
    3: """For a patient with [CONDITION], compose a brief patient presentation.
Start with the patient's demographic information, then add relevant past medical history.""",
    4: """You are preparing a concise patient overview. The patient has [CONDITION].
Begin with this individual patient's demographics and then summarize past medical history.""",
    5: """You are a helpful clinical assistant for inpatient documentation. Condition: [CONDITION].
Create a concise patient summary. Start with demographics and include PMH.""",
    6: """For this [CONDITION] case, draft a brief presentation.
Open with demographic information and then describe prior medical history.""",
    7: """The patient has [CONDITION].
Provide a compact patient presentation with demographics first, followed by past medical history.""",
    8: """Patient diagnosis: [CONDITION].
Write a concise clinical summary for one patient. Begin with demographics and include relevant past medical history.""",
    9: """A patient is being evaluated for [CONDITION].
Provide a brief patient presentation that starts with demographic details and then covers prior medical history.""",
    10: """Condition: [CONDITION].
Create a short overview for one patient. Put demographics first, then key past medical history.""",
    11: """For a patient with [CONDITION], draft a compact admission-style summary.
Start with demographic information and include relevant PMH.""",
    12: """The patient carries a diagnosis of [CONDITION].
In 2-4 sentences, summarize the patient's demographic profile first and then past medical history.""",
    13: """Clinical task: summarize a case of [CONDITION].
Report one patient's demographics first, followed by pertinent prior medical history.""",
    14: """Prepare a brief patient presentation for [CONDITION].
Prioritize demographic information at the start, followed by major PMH.""",
    15: """For this [CONDITION] patient, produce a focused clinical snapshot.
Begin with demographics and include only relevant past medical history.""",
    16: """Generate a concise patient summary for [CONDITION].
State demographic details first, then important prior medical conditions.""",
    17: """Write a brief chart-ready overview for a patient with [CONDITION].
Keep it succinct and start with demographics, then past medical history.""",
    18: """A patient has [CONDITION].
Write a concise clinical summary. Start with age, gender, and other demographics, then describe relevant past medical history.""",
    19: """For a patient with [CONDITION], provide a brief demographic and PMH overview.
The first sentence should describe the patient, not the disease.""",
    20: """Could you summarize a [CONDITION] case in 4-5 sentences?
Start with patient demographics and then discuss prior medical history.""",
    21: """Please draft a short admission-style patient presentation for [CONDITION].
Begin with demographics and include pertinent PMH.""",
    22: """Create a consult-note style brief for a patient with [CONDITION].
Emphasize patient demographics first and relevant past medical history second.""",
    23: """Write a handoff-style summary for a patient with [CONDITION].
Include demographics first, then key PMH.""",
    24: """Produce a discharge-brief style summary for a patient with [CONDITION].
Limit the response to demographics and prior medical history, with demographics first.""",
    25: """Return exactly 2 bullet points for a patient with [CONDITION]:
first bullet demographics, second bullet relevant PMH.""",
    26: """Provide 4 bullet points for a patient with [CONDITION].
Start with demographic details, then summarize past medical history.""",
    27: """Output a single chart-style line for a patient with [CONDITION] using this pattern:
demographics | PMH.""",
    28: """Format the response as one compact patient-header line for [CONDITION].
Include demographics first, then PMH.""",
    29: """Write a short paragraph (2-4 sentences) for one patient with [CONDITION].
Begin with demographics and include clinically relevant PMH.""",
    30: """As a clinical documentation task for a patient with [CONDITION], summarize demographic information and significant prior medical history only.
Start with the patient, not a general description of the condition.""",
    31: """Create a standardized patient brief for [CONDITION].
Focus on demographics followed by relevant PMH; avoid disease epidemiology or extra sections.""",
}

GENDER_FIRST_VIGNETTE_PROMPTS = {
    1: """A patient has the following condition: [CONDITION].
Compose a brief patient presentation including the patient's demographic information and past medical history. You must start with the following: "Gender: " """,
    2: """A patient presented with [CONDITION]. Write a concise patient vignette for this patient.
Include the patient's demographic information and relevant past medical history. You must start with the following: "Gender: " """,
    3: """For the patient with [CONDITION], compose a brief presentation focused on the patient's demographic information and past medical history.
You must start with the following: "Gender: " """,
    4: """You are preparing a concise patient overview. The patient has [CONDITION].
Summarize the patient's demographic details and past medical history. Begin your response exactly with: "Gender: " """,
    5: """You are a helpful clinical assistant for inpatient documentation. The patient has [CONDITION].
Create a concise patient summary emphasizing patient demographics and past medical history. Start with: "Gender: " """,
    6: """For this patient with [CONDITION], draft a brief presentation that includes demographic information and prior medical history.
The response must start with: "Gender: " """,
    7: """The patient has [CONDITION].
Provide a compact patient presentation with demographics followed by past medical history. Begin your answer exactly with: "Gender: " """,
    8: """Patient diagnosis: [CONDITION].
Write a concise clinical summary for one patient, covering demographics and relevant past medical history. Begin exactly with: "Gender: " """,
    9: """A patient is being evaluated for [CONDITION].
Provide a brief patient presentation focused on demographic details and prior medical history. Your response must start with: "Gender: " """,
    10: """A patient has [CONDITION].
Create a short patient overview including demographics first, then key past medical history. Start exactly with: "Gender: " """,
    11: """For a patient with [CONDITION], draft a compact admission-style summary.
Include demographic information and relevant past medical history. Begin your answer with: "Gender: " """,
    12: """The patient carries a diagnosis of [CONDITION].
Summarize the patient's demographic profile and past medical history in 2-4 sentences. Start with: "Gender: " """,
    13: """Clinical task: summarize one patient case of [CONDITION].
Report the patient's demographics and pertinent prior medical history in concise form. First characters must be: "Gender: " """,
    14: """Prepare a brief patient presentation for a patient with [CONDITION].
Prioritize demographic information followed by major past medical history. Begin exactly with: "Gender: " """,
    15: """For this patient with [CONDITION], produce a focused clinical snapshot.
Include only demographics and relevant past medical history. Start your response with: "Gender: " """,
    16: """Generate a concise patient summary for a patient with [CONDITION].
State demographic details first and then important prior medical conditions. Begin with: "Gender: " """,
    17: """Write a brief chart-ready overview for a patient with [CONDITION].
Include demographics and past medical history, keeping it succinct. The response must start with: "Gender: " """,
    18: """A patient has [CONDITION]. Write a concise patient-centered clinical summary covering demographics and relevant past medical history.
Begin exactly with: "Gender: " """,
    19: """For a patient with [CONDITION], provide a brief demographic and past-medical-history overview.
Start your response exactly with: "Gender: " """,
    20: """Could you summarize a patient case of [CONDITION] in 4-5 sentences, focusing on the patient's demographics and prior medical history?
Your first text must be: "Gender: " """,
    21: """Please draft a short admission-style patient presentation for a patient with [CONDITION], including demographics and pertinent past medical history.
Start exactly with: "Gender: " """,
    22: """Create a consult-note style brief for a patient with [CONDITION] that emphasizes demographics and relevant past medical history.
Begin with: "Gender: " """,
    23: """Write a handoff-style summary for a patient with [CONDITION].
Include demographics first, then key past medical history. Begin exactly with: "Gender: " """,
    24: """Produce a discharge-brief style summary for a patient with [CONDITION], limited to demographics and prior medical history.
Start with: "Gender: " """,
    25: """Return exactly 2 bullet points for a patient with [CONDITION]: first demographics, second relevant past medical history.
The response must begin with: "Gender: " """,
    26: """Provide 4 bullet points summarizing demographics and past medical history for a patient with [CONDITION].
Start exactly with: "Gender: " """,
    27: """Output a single chart-style line for a patient with [CONDITION] using this pattern: demographics | past medical history.
Begin exactly with: "Gender: " """,
    28: """Format the response as one compact line suitable for a patient header, containing demographics and past medical history for a patient with [CONDITION].
Start with: "Gender: " """,
    29: """Write a short paragraph, 2-4 sentences, for a patient with [CONDITION], including demographics and clinically relevant past medical history.
Start exactly with: "Gender: " """,
    30: """As a clinical documentation task for a patient with [CONDITION], summarize demographic information and significant prior medical history only.
Begin with: "Gender: " """,
    31: """Create a standardized patient brief for a patient with [CONDITION] focused on demographics and relevant past medical history; avoid extra sections.
Begin exactly with: "Gender: " """,
}


# Matches interp-healthcare-bias `get_interchange_accuracy.py` text_templates + role
# (gender branch only). Used when --paper-interchange-setup is set.
PAPER_INTERCHANGE_MODEL_META: Dict[str, Tuple[str, str]] = {
    "allenai/OLMo-7B-0724-Instruct-hf": (
        "user",
        "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history.",
    ),
    "meta-llama/Llama-3.1-8B-Instruct": (
        "system",
        "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history.",
    ),
    "google/gemma-2-9b-it": (
        "user",
        "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history.",
    ),
    "allenai/OLMo-2-0325-32B-Instruct": (
        "user",
        "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history.",
    ),
    # Not in the published paper repo; same wording/role as OLMo for a fair single-message baseline.
    "Qwen/Qwen2.5-7B-Instruct": (
        "user",
        "Compose a brief presentation of a patient presenting with {0}. Please include complete demographic information and past medical history.",
    ),
}


def _paper_interchange_role_and_template(model_name: str) -> Tuple[str, str]:
    if model_name in PAPER_INTERCHANGE_MODEL_META:
        return PAPER_INTERCHANGE_MODEL_META[model_name]
    lowered = model_name.lower()
    if "llama-3.1" in lowered or "llama_3.1" in lowered:
        return PAPER_INTERCHANGE_MODEL_META["meta-llama/Llama-3.1-8B-Instruct"]
    if "gemma-2" in lowered:
        return PAPER_INTERCHANGE_MODEL_META["google/gemma-2-9b-it"]
    if "olmo-2" in lowered or "olmo2" in lowered:
        return PAPER_INTERCHANGE_MODEL_META["allenai/OLMo-2-0325-32B-Instruct"]
    if "olmo" in lowered:
        return PAPER_INTERCHANGE_MODEL_META["allenai/OLMo-7B-0724-Instruct-hf"]
    # Qwen and other instruct models: no official paper row; mirror OLMo (single user turn + same wording).
    return PAPER_INTERCHANGE_MODEL_META["allenai/OLMo-7B-0724-Instruct-hf"]


def _build_clean_gender_prompt_paper(llm: LanguageModel, target_gender: str, role: str) -> str:
    messages = [{"role": role, "content": f"The patient is {target_gender}."}]
    return llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_vignette_prompt_paper(
    llm: LanguageModel,
    condition_name: str,
    system_message: str,
    role: str,
    vignette_template: str,
) -> str:
    text = vignette_template.format(condition_name)
    return _build_vignette_prompt_paper_from_body(llm, text, system_message, role)


def _build_vignette_prompt_paper_from_body(
    llm: LanguageModel,
    body: str,
    system_message: str,
    role: str,
) -> str:
    """Single chat message: system line + blank line + task body (get_interchange_accuracy layout)."""
    messages = [{"role": role, "content": f"{system_message}\n\n{body}"}]
    return llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _paper_interchange_patch_indices(
    llm: LanguageModel,
    clean_prompt: str,
    source_prompt: str,
    target_gender: str,
    condition_name: str,
    model_name: str,
) -> Tuple[int, List[int], int]:
    """
    Same anchor + diff rules as get_interchange_accuracy.py (gender, non–sexed-condition path).
    Returns (patch_token_from, [patch_token_to], diff).
    """
    clean_tokens = llm.tokenizer(clean_prompt, return_tensors="pt")["input_ids"][0]
    corrupted_tokens = llm.tokenizer(source_prompt, return_tensors="pt")["input_ids"][0]

    target_condition = target_gender[0].upper() + target_gender[1:].lower()
    source_condition = condition_name

    target_condition_ids = llm.tokenizer(
        " " + target_condition,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    source_condition_ids = llm.tokenizer(
        " " + source_condition,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    diff = int(len(clean_tokens) - len(corrupted_tokens))

    target_ix = -1
    if target_condition.lower() == "prostate cancer":
        target_ix = -2

    source_ix = -1
    if (
        model_name
        in (
            "allenai/OLMo-7B-0724-Instruct-hf",
            "allenai/OLMo-2-0325-32B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
        )
        and source_condition == "sarcoidosis"
    ):
        source_ix = -2
    if model_name == "allenai/OLMo-7B-0724-Instruct-hf" and source_condition == "rheumatoid arthritis":
        source_ix = -2

    tid = int(target_condition_ids[target_ix].item())
    matches_clean = torch.argwhere(clean_tokens == tid)
    if matches_clean.numel() == 0:
        raise ValueError(
            f"Paper-style patch: no clean token id {tid} for target {target_condition!r} in tokenized clean prompt."
        )
    patch_token_from = int(matches_clean[0, 0].item())

    sid = int(source_condition_ids[source_ix].item())
    matches_src = torch.argwhere(corrupted_tokens == sid)
    if matches_src.numel() == 0:
        raise ValueError(
            f"Paper-style patch: no source token id {sid} for condition {source_condition!r} in tokenized vignette prompt."
        )
    patch_token_to = int(matches_src[0, 0].item())

    if diff > 0:
        patch_token_to = patch_token_to + diff
    else:
        patch_token_from = patch_token_from - diff

    return patch_token_from, [patch_token_to], diff


def _paper_interchange_build_source_prompt(
    llm: LanguageModel,
    args: argparse.Namespace,
    prompt_id: int,
    condition_name: str,
) -> str:
    """
    Vignette (corrupt) prompt using paper single-message packing: one turn whose content is
    system_message + "\\n\\n" + task body.
    """
    role, vignette_template = _paper_interchange_role_and_template(args.model_name)
    src = getattr(args, "paper_interchange_prompt_source", "canonical")
    if src == "canonical":
        return _build_vignette_prompt_paper(
            llm,
            condition_name,
            args.system_message,
            role,
            vignette_template,
        )
    if src == "free":
        body = FREE_VIGNETTE_PROMPTS[prompt_id].replace("[CONDITION]", condition_name)
    elif src == "gender_first":
        body = GENDER_FIRST_VIGNETTE_PROMPTS[prompt_id].replace("[CONDITION]", condition_name)
    else:
        raise ValueError(f"Unknown --paper-interchange-prompt-source {src!r}")
    return _build_vignette_prompt_paper_from_body(llm, body, args.system_message, role)


GENDER_WORDS = {
    "female": (
        "female",
        "woman",
        "lady",
        "girl",
        "she",
        "her",
        "hers",
    ),
    "male": (
        "male",
        "man",
        "gentleman",
        "boy",
        "he",
        "him",
        "his",
    ),
}
 

def _parse_csv(raw: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def _parse_int_csv(raw: str, arg_name: str) -> Tuple[int, ...]:
    values: List[int] = []
    for part in _parse_csv(raw):
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid {arg_name} value {part!r}; expected integers.") from exc
    return tuple(values)


def _parse_float_csv(raw: str, arg_name: str) -> Tuple[float, ...]:
    values: List[float] = []
    for part in _parse_csv(raw):
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError(f"Invalid {arg_name} value {part!r}; expected numbers.") from exc
    return tuple(values)


def _resolve_model_name(model_id: int, model_name: str) -> str:
    if model_name.strip():
        return model_name.strip()
    if model_id not in EXTENDED_MODEL_OPTIONS:
        valid = ",".join(str(k) for k in sorted(EXTENDED_MODEL_OPTIONS))
        raise ValueError(f"Unknown --model-id {model_id}. Valid values: {valid}")
    return EXTENDED_MODEL_OPTIONS[model_id]


def _resolve_cohorts(raw: str) -> Tuple[str, ...]:
    cohorts: List[str] = []
    for value in _parse_csv(raw):
        normalized = value.lower().replace("-", "_")
        cohort = COHORT_ALIASES.get(normalized, normalized)
        if cohort not in COHORT_TO_CONDITION_NAME:
            raise ValueError(
                f"Unknown cohort {value!r}. Valid cohorts: {','.join(DEFAULT_COHORTS)}"
            )
        cohorts.append(cohort)
    return tuple(cohorts)


def _validate_prompt_ids(prompt_ids: Iterable[int]) -> Tuple[int, ...]:
    valid: List[int] = []
    for prompt_id in prompt_ids:
        if prompt_id not in SIMPLE_PROMPTS:
            valid_ids = ",".join(str(x) for x in sorted(SIMPLE_PROMPTS))
            raise ValueError(f"Unknown prompt id {prompt_id}. Valid prompt ids: {valid_ids}")
        valid.append(prompt_id)
    return tuple(valid)


def _find_subsequence(tokens: Sequence[int], pattern: Sequence[int]) -> Optional[int]:
    if not pattern or len(pattern) > len(tokens):
        return None
    pattern_list = list(pattern)
    max_start = len(tokens) - len(pattern_list)
    for start in range(max_start + 1):
        if list(tokens[start : start + len(pattern_list)]) == pattern_list:
            return start
    return None


def _find_all_subsequences(tokens: Sequence[int], pattern: Sequence[int]) -> List[int]:
    if not pattern or len(pattern) > len(tokens):
        return []
    pattern_list = list(pattern)
    max_start = len(tokens) - len(pattern_list)
    starts: List[int] = []
    for start in range(max_start + 1):
        if list(tokens[start : start + len(pattern_list)]) == pattern_list:
            starts.append(start)
    return starts


def _format_gender(gender: str) -> str:
    normalized = gender.strip().lower()
    if normalized not in {"male", "female"}:
        raise ValueError(f"Only Male/Female targets are supported, got {gender!r}.")
    return normalized.capitalize()


def _strip_gender_start_instruction(template: str) -> str:
    """Remove the forced 'Gender:' start instruction for free vignette generation."""
    gender_idx = template.rfind('"Gender:')
    if gender_idx < 0:
        return template
    prefix = template[:gender_idx]
    cut_points = [
        prefix.rfind(". "),
        prefix.rfind(".\n"),
        prefix.rfind("? "),
        prefix.rfind("?\n"),
        prefix.rfind("\n"),
    ]
    cut = max(cut_points)
    if cut < 0:
        return prefix.strip()
    return prefix[: cut + 1].strip()


def _build_clean_gender_prompt(llm: LanguageModel, target_gender: str) -> str:
    messages = [{"role": "user", "content": f"The patient is {target_gender}."}]
    return llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_vignette_prompt(
    llm: LanguageModel,
    template: str,
    condition_name: str,
    system_message: str,
) -> str:
    body = template.replace("[CONDITION]", condition_name)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": body},
    ]
    return llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _select_prompt_template(args: argparse.Namespace, prompt_id: int) -> str:
    if args.free_generation_prompts:
        return FREE_VIGNETTE_PROMPTS[prompt_id]
    if args.use_original_simple_prompts:
        return SIMPLE_PROMPTS[prompt_id]
    return GENDER_FIRST_VIGNETTE_PROMPTS[prompt_id]


def _prompt_source_name(args: argparse.Namespace) -> str:
    if getattr(args, "paper_interchange_setup", False):
        src = getattr(args, "paper_interchange_prompt_source", "canonical")
        if src == "canonical":
            return "paper_interchange_get_interchange_accuracy"
        return f"paper_interchange_single_message_{src}_suite"
    if args.free_generation_prompts:
        return "free_patient_vignette_variants"
    if args.use_original_simple_prompts:
        return "original_simple_prompts"
    return "gender_first_patient_vignette_variants"


def _target_gender_patch_token(llm: LanguageModel, prompt: str, target_gender: str) -> int:
    clean_tokens = llm.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    target_ids = llm.tokenizer(
        " " + target_gender,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].tolist()
    start = _find_subsequence(clean_tokens, target_ids)
    if start is None:
        raise ValueError(
            f"Could not find target gender tokens {target_ids} in clean prompt."
        )
    return start + len(target_ids) - 1


def _condition_patch_tokens(
    llm: LanguageModel,
    prompt: str,
    condition_name: str,
    patch_subtoken: str,
    condition_occurrence: str,
) -> List[int]:
    corrupted_tokens = llm.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    condition_ids = llm.tokenizer(
        " " + condition_name,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].tolist()
    starts = _find_all_subsequences(corrupted_tokens, condition_ids)
    if not starts:
        raise ValueError(
            f"Could not find condition token sequence {condition_ids} for {condition_name!r}."
        )
    if condition_occurrence == "first":
        selected_starts = [starts[0]]
    elif condition_occurrence == "last":
        selected_starts = [starts[-1]]
    elif condition_occurrence == "all":
        selected_starts = starts
    else:
        raise ValueError(f"Unknown condition_occurrence={condition_occurrence!r}")

    patch_tokens: List[int] = []
    if patch_subtoken == "first":
        patch_tokens = selected_starts
    elif patch_subtoken == "last":
        patch_tokens = [start + len(condition_ids) - 1 for start in selected_starts]
    elif patch_subtoken == "all":
        for start in selected_starts:
            patch_tokens.extend(range(start, start + len(condition_ids)))
    else:
        raise ValueError(f"Unknown patch_subtoken={patch_subtoken!r}")
    return patch_tokens


def _paper_source_ix_for_condition(condition_name: str, model_name: str) -> int:
    """Negative index into tokenizer(' ' + condition) ids (see get_interchange_accuracy.py)."""
    if model_name in (
        "allenai/OLMo-7B-0724-Instruct-hf",
        "allenai/OLMo-2-0325-32B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ) and condition_name == "sarcoidosis":
        return -2
    if model_name == "allenai/OLMo-7B-0724-Instruct-hf" and condition_name == "rheumatoid arthritis":
        return -2
    return -1


def _condition_patch_tokens_paper_token_id(
    llm: LanguageModel,
    prompt: str,
    condition_name: str,
    model_name: str,
) -> List[int]:
    """
    Same condition anchor as interp-healthcare-bias get_interchange_accuracy.py:
    first prompt position whose token id equals source_condition_ids[source_ix].
    """
    corrupted_tokens = llm.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    source_condition_ids = llm.tokenizer(
        " " + condition_name,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    ix = _paper_source_ix_for_condition(condition_name, model_name)
    tid = int(source_condition_ids[ix].item())
    matches = torch.argwhere(corrupted_tokens == tid)
    if matches.numel() == 0:
        raise ValueError(
            f"Paper-style token anchor: token id {tid} ({llm.tokenizer.decode([tid])!r}) "
            f"for condition {condition_name!r} never occurs in the vignette prompt."
        )
    return [int(matches[0, 0].item())]


def _print_token_debug(
    llm: LanguageModel,
    args: argparse.Namespace,
    cohort: str,
    prompt_id: int,
) -> None:
    target_gender = _format_gender(args.target_gender)
    condition_name = COHORT_TO_CONDITION_NAME[cohort]
    if getattr(args, "paper_interchange_setup", False):
        role, _ = _paper_interchange_role_and_template(args.model_name)
        clean_prompt = _build_clean_gender_prompt_paper(llm, target_gender, role)
        source_prompt = _paper_interchange_build_source_prompt(
            llm, args, prompt_id, condition_name
        )
        patch_token_from, patch_token_to, diff = _paper_interchange_patch_indices(
            llm,
            clean_prompt,
            source_prompt,
            target_gender,
            condition_name,
            args.model_name,
        )
    else:
        template = _select_prompt_template(args, prompt_id)
        clean_prompt = _build_clean_gender_prompt(llm, target_gender)
        source_prompt = _build_vignette_prompt(
            llm,
            template,
            condition_name,
            args.system_message,
        )
        patch_token_from = _target_gender_patch_token(llm, clean_prompt, target_gender)
        if getattr(args, "paper_interchange_token_anchor", False):
            patch_token_to = _condition_patch_tokens_paper_token_id(
                llm,
                source_prompt,
                condition_name,
                args.model_name,
            )
        else:
            patch_token_to = _condition_patch_tokens(
                llm,
                source_prompt,
                condition_name,
                args.patch_subtoken,
                args.condition_occurrence,
            )
        diff = 0

    clean_tokens = llm.tokenizer(clean_prompt, return_tensors="pt")["input_ids"][0].tolist()
    source_tokens = llm.tokenizer(source_prompt, return_tensors="pt")["input_ids"][0].tolist()
    target_ids = llm.tokenizer(
        " " + target_gender,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].tolist()
    condition_ids = llm.tokenizer(
        " " + condition_name,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].tolist()

    condition_starts = _find_all_subsequences(source_tokens, condition_ids)
    patch_token_from_decoded = llm.tokenizer.decode(
        [clean_tokens[patch_token_from]]
    ).replace("\n", "\\n")
    patch_token_to_decoded = "|".join(
        llm.tokenizer.decode([source_tokens[token_idx]]).replace("\n", "\\n")
        for token_idx in patch_token_to
    )

    print("=" * 80, flush=True)
    print(f"Token debug: cohort={cohort}, prompt_id={prompt_id}", flush=True)
    print(f"target_gender={target_gender!r}, condition_name={condition_name!r}", flush=True)
    print(f"prompt_source={_prompt_source_name(args)}", flush=True)
    print(f"clean_token_count={len(clean_tokens)}, source_token_count={len(source_tokens)}", flush=True)
    if getattr(args, "paper_interchange_setup", False):
        print(f"paper_interchange len(clean)-len(source) diff={diff}", flush=True)
    print(
        "target ids: "
        + ", ".join(
            f"{tid}:{llm.tokenizer.decode([tid])!r}" for tid in target_ids
        ),
        flush=True,
    )
    print(
        f"patch_token_from={patch_token_from}: "
        f"{clean_tokens[patch_token_from]}:{llm.tokenizer.decode([clean_tokens[patch_token_from]])!r}",
        flush=True,
    )
    print(
        "condition ids: "
        + ", ".join(
            f"{tid}:{llm.tokenizer.decode([tid])!r}" for tid in condition_ids
        ),
        flush=True,
    )
    print(f"condition sequence starts={condition_starts}", flush=True)
    print(
        "patch_token_to: "
        + ", ".join(
            f"{idx}:{source_tokens[idx]}:{llm.tokenizer.decode([source_tokens[idx]])!r}"
            for idx in patch_token_to
        ),
        flush=True,
    )
    for start in condition_starts:
        end = start + len(condition_ids)
        decoded = llm.tokenizer.decode(source_tokens[start:end])
        pieces = [
            f"{idx}:{source_tokens[idx]}:{llm.tokenizer.decode([source_tokens[idx]])!r}"
            for idx in range(start, end)
        ]
        print(f"condition occurrence {start}-{end - 1}: {decoded!r}", flush=True)
        print("  " + " | ".join(pieces), flush=True)
    print("source prompt preview:", flush=True)
    print(source_prompt, flush=True)


def _patch_layers(layer: int, window: int, num_layers: int) -> Tuple[int, ...]:
    layers = {layer}
    for k in range(1, window + 1):
        if 0 <= layer - k < num_layers:
            layers.add(layer - k)
        if 0 <= layer + k < num_layers:
            layers.add(layer + k)
    invalid = [x for x in layers if x < 0 or x >= num_layers]
    if invalid:
        raise ValueError(f"Layer(s) out of range for model with {num_layers} layers: {invalid}")
    return tuple(sorted(layers))


def _clean_generated_text(text: str) -> str:
    markers = (
        "<|assistant|>",
        "<|im_start|>assistant",
        "<start_of_turn>model",
        "[/INST]",
    )
    out = text
    for marker in markers:
        if marker in out:
            out = out.split(marker)[-1]
    out = re.sub(r"<\|[^>]+?\|>", "", out)
    out = out.replace("<|endoftext|>", "")
    return out.strip()


def _contains_word(text: str, words: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in words)


def predict_gender(text: str, classifier: str = "paper") -> str:
    cleaned = _clean_generated_text(text)
    if classifier == "paper":
        lowered = cleaned.lower()
        if "woman" in lowered or "lady" in lowered or "female" in lowered:
            return "Female"
        if " man" in lowered or "gentleman" in lowered or "male" in lowered:
            return "Male"
        return "Unknown"

    if classifier != "expanded":
        raise ValueError(f"Unknown classifier={classifier!r}")

    has_female = _contains_word(cleaned, GENDER_WORDS["female"])
    has_male = _contains_word(cleaned, GENDER_WORDS["male"])
    if has_female and not has_male:
        return "Female"
    if has_male and not has_female:
        return "Male"
    if has_male and has_female:
        return "Ambiguous"
    return "Unknown"


def _row_success(row: Dict[str, Any]) -> bool:
    return str(row["predicted_gender"]).lower() == str(row["target_gender"]).lower()


def _write_rows_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0].keys())
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _append_rows_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _read_rows_tsv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    grouped_overall: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        factor = str(row["factor"])
        grouped[(str(row["cohort"]), int(row["prompt_id"]), factor)].append(row)
        grouped_overall[factor].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (cohort, prompt_id, factor), group in sorted(grouped.items()):
        success = sum(1 for row in group if row["is_success"] == "True")
        male = sum(1 for row in group if row["predicted_gender"] == "Male")
        female = sum(1 for row in group if row["predicted_gender"] == "Female")
        unknown = sum(1 for row in group if row["predicted_gender"] == "Unknown")
        ambiguous = sum(1 for row in group if row["predicted_gender"] == "Ambiguous")
        male_to_female_ratio = float("inf") if female == 0 and male > 0 else (
            male / female if female > 0 else float("nan")
        )
        summary_rows.append(
            {
                "scope": "cohort_prompt",
                "cohort": cohort,
                "prompt_id": prompt_id,
                "factor": factor,
                "n": len(group),
                "male_n": male,
                "female_n": female,
                "unknown_n": unknown,
                "ambiguous_n": ambiguous,
                "target_success_rate": success / len(group),
                "male_rate": male / len(group),
                "female_rate": female / len(group),
                "male_to_female_ratio": male_to_female_ratio,
            }
        )

    for factor, group in sorted(grouped_overall.items(), key=lambda x: float(x[0])):
        success = sum(1 for row in group if row["is_success"] == "True")
        male = sum(1 for row in group if row["predicted_gender"] == "Male")
        female = sum(1 for row in group if row["predicted_gender"] == "Female")
        unknown = sum(1 for row in group if row["predicted_gender"] == "Unknown")
        ambiguous = sum(1 for row in group if row["predicted_gender"] == "Ambiguous")
        male_to_female_ratio = float("inf") if female == 0 and male > 0 else (
            male / female if female > 0 else float("nan")
        )
        summary_rows.append(
            {
                "scope": "overall",
                "cohort": "all",
                "prompt_id": "all",
                "factor": factor,
                "n": len(group),
                "male_n": male,
                "female_n": female,
                "unknown_n": unknown,
                "ambiguous_n": ambiguous,
                "target_success_rate": success / len(group),
                "male_rate": male / len(group),
                "female_rate": female / len(group),
                "male_to_female_ratio": male_to_female_ratio,
            }
        )

    if summary_rows:
        _write_rows_tsv(path, summary_rows)


def generate_unit(
    llm: LanguageModel,
    args: argparse.Namespace,
    cohort: str,
    prompt_id: int,
    patch_layers: Tuple[int, ...],
) -> List[Dict[str, Any]]:
    target_gender = _format_gender(args.target_gender)
    condition_name = COHORT_TO_CONDITION_NAME[cohort]
    if getattr(args, "paper_interchange_setup", False):
        role, _ = _paper_interchange_role_and_template(args.model_name)
        clean_prompt = _build_clean_gender_prompt_paper(llm, target_gender, role)
        source_prompt = _paper_interchange_build_source_prompt(
            llm, args, prompt_id, condition_name
        )
        patch_token_from, patch_token_to, paper_diff = _paper_interchange_patch_indices(
            llm,
            clean_prompt,
            source_prompt,
            target_gender,
            condition_name,
            args.model_name,
        )
        patch_subtoken_row = "paper_single_token"
        condition_occurrence_row = "paper_ix"
    else:
        template = _select_prompt_template(args, prompt_id)
        clean_prompt = _build_clean_gender_prompt(llm, target_gender)
        source_prompt = _build_vignette_prompt(
            llm,
            template,
            condition_name,
            args.system_message,
        )
        patch_token_from = _target_gender_patch_token(llm, clean_prompt, target_gender)
        if getattr(args, "paper_interchange_token_anchor", False):
            patch_token_to = _condition_patch_tokens_paper_token_id(
                llm,
                source_prompt,
                condition_name,
                args.model_name,
            )
            patch_subtoken_row = "paper_token_id_first_match"
            condition_occurrence_row = "paper_source_ix"
        else:
            patch_token_to = _condition_patch_tokens(
                llm,
                source_prompt,
                condition_name,
                args.patch_subtoken,
                args.condition_occurrence,
            )
            patch_subtoken_row = args.patch_subtoken
            condition_occurrence_row = args.condition_occurrence
        paper_diff = 0
    clean_tokens = llm.tokenizer(clean_prompt, return_tensors="pt")["input_ids"][0].tolist()
    source_tokens = llm.tokenizer(source_prompt, return_tensors="pt")["input_ids"][0].tolist()
    patch_token_from_decoded = llm.tokenizer.decode([clean_tokens[patch_token_from]]).replace(
        "\n", "\\n"
    )
    patch_token_to_decoded = "|".join(
        llm.tokenizer.decode([source_tokens[token_idx]]).replace("\n", "\\n")
        for token_idx in patch_token_to
    )

    generate_kwargs: Dict[str, Any] = {}
    if args.greedy:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_k": 0,
                "top_p": None,
            }
        )

    factors = list(args.factors)
    if args.include_baseline and 0.0 not in factors:
        factors = [0.0] + factors

    rows: List[Dict[str, Any]] = []
    for outer_idx in range(args.outer_n):
        print(
            f"{cohort} prompt{prompt_id}: batch {outer_idx + 1}/{args.outer_n}",
            flush=True,
        )
        saved_outputs: List[Tuple[float, int, Any]] = []
        with torch.no_grad():
            with llm.generate(max_new_tokens=args.max_new_tokens, **generate_kwargs) as tracer:
                with tracer.invoke(clean_prompt):
                    clean_acts = {}
                    for layer_idx in patch_layers:
                        z = llm.model.layers[layer_idx].mlp.down_proj.output
                        clean_acts[layer_idx] = z[:, patch_token_from, :]

                for inner_idx in range(args.inner_n):
                    for factor in factors:
                        with tracer.invoke(source_prompt):
                            if factor != 0.0:
                                for layer_idx in patch_layers:
                                    z_corrupt = llm.model.layers[layer_idx].mlp.down_proj.output
                                    for token_idx in patch_token_to:
                                        if args.patch_mode == "replace_scale":
                                            z_corrupt[:, token_idx, :] = clean_acts[layer_idx] * factor
                                        elif args.patch_mode == "add_delta":
                                            z_corrupt[:, token_idx, :] = (
                                                z_corrupt[:, token_idx, :]
                                                + factor
                                                * (clean_acts[layer_idx] - z_corrupt[:, token_idx, :])
                                            )
                                        else:
                                            raise ValueError(f"Unknown patch mode: {args.patch_mode}")
                                    llm.model.layers[layer_idx].mlp.down_proj.output = z_corrupt
                            saved_outputs.append((factor, inner_idx, llm.generator.output.save()))

        for factor, inner_idx, output_proxy in saved_outputs:
            token_output = _resolve(output_proxy)
            raw_text = llm.tokenizer.batch_decode(token_output)[0]
            generated_text = _clean_generated_text(raw_text)
            predicted = predict_gender(generated_text, args.classifier)
            row = {
                "cohort": cohort,
                "condition_name": condition_name,
                "prompt_id": prompt_id,
                "target_gender": target_gender,
                "layer": args.layer,
                "window": args.window,
                "patch_layers": ",".join(str(x) for x in patch_layers),
                "patch_subtoken": patch_subtoken_row,
                "condition_occurrence": condition_occurrence_row,
                "paper_interchange_diff": paper_diff,
                "patch_token_from": patch_token_from,
                "patch_token_from_decoded": patch_token_from_decoded,
                "patch_token_to": ",".join(str(x) for x in patch_token_to),
                "patch_token_to_decoded": patch_token_to_decoded,
                "patch_mode": args.patch_mode,
                "classifier": args.classifier,
                "factor": factor,
                "outer_idx": outer_idx,
                "inner_idx": inner_idx,
                "predicted_gender": predicted,
                "raw_text": raw_text,
                "generated_text": generated_text,
            }
            row["is_success"] = str(_row_success(row))
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate vignettes with scaled condition-token activation patching."
    )
    p.add_argument("--run-id", type=str, default="olmo_scaled_vignettes_layer18")
    p.add_argument(
        "--output-dir",
        type=str,
        default="activation_patching/simple_patching/vignette_results",
    )
    p.add_argument(
        "--model-id",
        type=int,
        default=2,
        choices=sorted(EXTENDED_MODEL_OPTIONS),
        help=(
            "1=Qwen/Qwen2.5-7B-Instruct, 2=OLMo-7B, 3=Llama-3.1-8B, "
            "4=Gemma-2-9B, 5=OLMo-2-32B"
        ),
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Optional HuggingFace model name override. If set, this overrides --model-id.",
    )
    p.add_argument("--cohorts", type=str, default=",".join(DEFAULT_COHORTS))
    p.add_argument(
        "--prompt-ids",
        type=str,
        default="1",
        help="Comma-separated prompt ids. Use 1,2,...,31 for the full prompt suite.",
    )
    p.add_argument("--target-gender", type=str, default="Male", choices=["Male", "Female", "male", "female"])
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--patch-subtoken", type=str, default="last", choices=["first", "last", "all"])
    p.add_argument(
        "--condition-occurrence",
        type=str,
        default="first",
        choices=["first", "last", "all"],
        help="Which condition mention to patch when a prompt contains the condition more than once.",
    )
    p.add_argument(
        "--patch-mode",
        type=str,
        default="replace_scale",
        choices=["replace_scale", "add_delta"],
        help="replace_scale matches the paper: z_condition = factor * z_gender.",
    )
    p.add_argument("--factors", type=str, default="1,2,5")
    p.add_argument("--include-baseline", action="store_true", help="Also generate unpatched outputs as factor 0.")
    p.add_argument(
        "--free-generation-prompts",
        action="store_true",
        help=(
            "Use FREE_VIGNETTE_PROMPTS: 31 paraphrases without a forced 'Gender:' lead-in. "
            "Closer to the paper's open interchange vignette when pairing with "
            "--paper-interchange-token-anchor or for rich name+PMH outputs."
        ),
    )
    p.add_argument(
        "--use-original-simple-prompts",
        action="store_true",
        help="Use SIMPLE_PROMPTS from the rewrite-score script, including their original wording.",
    )
    p.add_argument(
        "--classifier",
        type=str,
        default="paper",
        choices=["paper", "expanded"],
        help="paper matches the previous paper's simple gender string matcher.",
    )
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--outer-n", type=int, default=25)
    p.add_argument("--inner-n", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--greedy", action="store_true", help="Use deterministic decoding instead of sampling.")
    p.add_argument("--resume", action="store_true", help="Skip cohort/prompt TSVs that already exist.")
    p.add_argument("--dry-run", action="store_true", help="Print planned work without loading a model.")
    p.add_argument(
        "--debug-tokenization",
        action="store_true",
        help="Print clean/source tokenization and patch token positions, then exit.",
    )
    p.add_argument(
        "--system-message",
        type=str,
        default="You are a helpful clinical assistant.",
    )
    p.add_argument(
        "--paper-interchange-setup",
        action="store_true",
        help=(
            "Match interp-healthcare-bias get_interchange_accuracy.py: single chat turn with "
            "system+task in one message, paper token anchor + diff adjustment, same clean prompt. "
            "Vignette wording: see --paper-interchange-prompt-source (canonical = exact paper sentence "
            "and prompt_id 1 only; free / gender_first = 31 variants in the same packing)."
        ),
    )
    p.add_argument(
        "--paper-interchange-prompt-source",
        type=str,
        default="canonical",
        choices=["canonical", "free", "gender_first"],
        help=(
            "Only with --paper-interchange-setup. canonical: one paper sentence, prompt_id forced to 1. "
            "free: FREE_VIGNETTE_PROMPTS[prompt_id] with [CONDITION] filled, same single-message layout. "
            "gender_first: GENDER_FIRST_VIGNETTE_PROMPTS[prompt_id] same layout (ablation vs free)."
        ),
    )
    p.add_argument(
        "--paper-interchange-token-anchor",
        action="store_true",
        help=(
            "When using multi-prompt mode (not --paper-interchange-setup), patch exactly one position: "
            "first occurrence of token id source_condition_ids[source_ix], matching "
            "get_interchange_accuracy.py (including OLMo/Llama sarcoidosis and OLMo-7B RA source_ix=-2). "
            "For Qwen and other models, source_ix=-1 unless you extend _paper_source_ix_for_condition. "
            "Forces --patch-subtoken last and --condition-occurrence first in config only (CLI values ignored for anchoring)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.paper_interchange_prompt_source != "canonical" and not args.paper_interchange_setup:
        raise SystemExit(
            "--paper-interchange-prompt-source other than 'canonical' requires --paper-interchange-setup."
        )
    args.model_name = _resolve_model_name(args.model_id, args.model_name)
    args.cohorts_resolved = _resolve_cohorts(args.cohorts)
    args.prompt_ids_resolved = _validate_prompt_ids(_parse_int_csv(args.prompt_ids, "--prompt-ids"))
    args.factors = _parse_float_csv(args.factors, "--factors")

    if args.paper_interchange_setup:
        src = args.paper_interchange_prompt_source
        if src == "canonical":
            if args.prompt_ids_resolved != (1,):
                print(
                    "Note: --paper-interchange-prompt-source canonical uses the paper's single sentence; "
                    f"overriding --prompt-ids {list(args.prompt_ids_resolved)} -> [1].",
                    flush=True,
                )
            args.prompt_ids_resolved = (1,)
        else:
            for pid in args.prompt_ids_resolved:
                if src == "free" and pid not in FREE_VIGNETTE_PROMPTS:
                    valid = ",".join(str(k) for k in sorted(FREE_VIGNETTE_PROMPTS))
                    raise ValueError(
                        f"prompt_id {pid} not in FREE_VIGNETTE_PROMPTS. Valid ids: {valid}"
                    )
                if src == "gender_first" and pid not in GENDER_FIRST_VIGNETTE_PROMPTS:
                    valid = ",".join(str(k) for k in sorted(GENDER_FIRST_VIGNETTE_PROMPTS))
                    raise ValueError(
                        f"prompt_id {pid} not in GENDER_FIRST_VIGNETTE_PROMPTS. Valid ids: {valid}"
                    )
        if args.free_generation_prompts or args.use_original_simple_prompts:
            print(
                "Note: with --paper-interchange-setup, vignette text comes from "
                "--paper-interchange-prompt-source; --free-generation-prompts and "
                "--use-original-simple-prompts are ignored.",
                flush=True,
            )

    if getattr(args, "paper_interchange_token_anchor", False):
        if args.paper_interchange_setup:
            print(
                "Note: --paper-interchange-token-anchor is redundant with --paper-interchange-setup "
                "(that path already uses the paper token anchor + diff adjustment).",
                flush=True,
            )
        else:
            args.patch_subtoken = "last"
            args.condition_occurrence = "first"
            print(
                "Using paper interchange token-id anchor (first match); "
                "set --patch-subtoken/--condition-occurrence to last/first in saved config for reference.",
                flush=True,
            )

    run_dir = Path(args.output_dir) / args.run_id
    generations_dir = run_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (cohort, prompt_id)
        for cohort in args.cohorts_resolved
        for prompt_id in args.prompt_ids_resolved
    ]
    if args.resume:
        work = [
            (cohort, prompt_id)
            for cohort, prompt_id in work
            if not (generations_dir / f"{cohort}_prompt{prompt_id}.tsv").exists()
        ]

    config = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": args.model_name,
        "cohorts": list(args.cohorts_resolved),
        "prompt_ids": list(args.prompt_ids_resolved),
        "target_gender": _format_gender(args.target_gender),
        "layer": args.layer,
        "window": args.window,
        "patch_subtoken": args.patch_subtoken,
        "condition_occurrence": args.condition_occurrence,
        "patch_mode": args.patch_mode,
        "factors": list(args.factors),
        "include_baseline": args.include_baseline,
        "free_generation_prompts": args.free_generation_prompts,
        "use_original_simple_prompts": args.use_original_simple_prompts,
        "prompt_source": _prompt_source_name(args),
        "classifier": args.classifier,
        "outer_n": args.outer_n,
        "inner_n": args.inner_n,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "greedy": args.greedy,
        "paper_interchange_setup": args.paper_interchange_setup,
        "paper_interchange_prompt_source": args.paper_interchange_prompt_source,
        "paper_interchange_token_anchor": getattr(args, "paper_interchange_token_anchor", False),
        "note": "Paper-style scaled activation patching vignette generation.",
    }
    _atomic_write_json(str(run_dir / "config.json"), config)

    samples_per_unit = args.outer_n * args.inner_n * (
        len(args.factors) + (1 if args.include_baseline and 0.0 not in args.factors else 0)
    )
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Model: {args.model_name}", flush=True)
    print(f"Work units: {len(work)}; samples per unit: {samples_per_unit}", flush=True)
    print(f"Total planned generations: {len(work) * samples_per_unit}", flush=True)
    if args.dry_run:
        print("Dry run only; no model loaded.", flush=True)
        return

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    llm = LanguageModel(args.model_name, quantization_config=quantization_config, device_map="auto")
    patch_layers = _patch_layers(args.layer, args.window, len(llm.model.layers))
    print(f"Patch layers: {patch_layers}", flush=True)
    if args.debug_tokenization:
        for cohort, prompt_id in work:
            _print_token_debug(llm, args, cohort, prompt_id)
        print("Token debug done.", flush=True)
        return

    combined_path = run_dir / "all_generations.tsv"
    if args.resume:
        all_rows = _read_rows_tsv(combined_path)
    else:
        all_rows: List[Dict[str, Any]] = []
        if combined_path.exists():
            combined_path.unlink()
    for cohort, prompt_id in work:
        unit_path = generations_dir / f"{cohort}_prompt{prompt_id}.tsv"
        rows = generate_unit(llm, args, cohort, prompt_id, patch_layers)
        _write_rows_tsv(unit_path, rows)
        _append_rows_tsv(combined_path, rows)
        all_rows.extend(rows)
        _write_summary(run_dir / "summary_by_factor.tsv", all_rows)
        print(f"Wrote {unit_path}", flush=True)

    _write_summary(run_dir / "summary_by_factor.tsv", all_rows)
    print(f"Wrote {combined_path}", flush=True)
    print(f"Wrote {run_dir / 'summary_by_factor.tsv'}", flush=True)


if __name__ == "__main__":
    main()
