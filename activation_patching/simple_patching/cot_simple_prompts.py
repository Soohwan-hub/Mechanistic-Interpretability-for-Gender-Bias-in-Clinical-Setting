"""CoT prompt registry for simple no-BHC rewrite sweeps (<thinking> tag variants)."""
from __future__ import annotations

from typing import Dict, Tuple

from cot_thinking_prompts import FROZEN_PROMPT, VIGNETTE_PROMPT_A, VIGNETTE_PROMPT_C

COT_PROMPT_GROUPS: Tuple[Tuple[str, Dict[str, str]], ...] = (
    ("A", VIGNETTE_PROMPT_A),
    ("C", VIGNETTE_PROMPT_C),
)

COT_CORRUPT_MODE_CHOICES = ("full", "frozen")


def build_cot_simple_prompts() -> Dict[int, str]:
    """Map prompt ids 1–20 to CoT templates (10 Type A + 10 Type C)."""
    prompts: Dict[int, str] = {}
    prompt_id = 1
    for _group_name, group in COT_PROMPT_GROUPS:
        for var_name in sorted(group.keys(), key=lambda x: int(x.replace("var", ""))):
            prompts[prompt_id] = group[var_name]
            prompt_id += 1
    return prompts


def cot_prompt_label(prompt_id: int) -> str:
    """Human-readable label, e.g. A_var3 or C_var7."""
    if prompt_id < 1 or prompt_id > 20:
        raise ValueError(f"CoT prompt_id must be 1–20, got {prompt_id}")
    if prompt_id <= 10:
        return f"A_var{prompt_id}"
    return f"C_var{prompt_id - 10}"


COT_SIMPLE_PROMPTS = build_cot_simple_prompts()
