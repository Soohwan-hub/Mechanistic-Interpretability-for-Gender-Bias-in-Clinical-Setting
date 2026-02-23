from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# ============================================================
# 1. EXPERIMENT CONFIG
# ============================================================

@dataclass(frozen=True)
class ExperimentConfig:
    # Input note file to build prompt variants from.
    input_file: Path = Path("train.json")
    output_dir: Path = Path("data")
    triplets_file: Path = Path("data/case_triplets.jsonl")

    # Keep this capped for now (per project plan).
    max_cases: int = 100
    seed: int = 42

    # Prompt variants to run for vignette generation.
    active_prompt_ids: Tuple[str, ...] = (
        "vignette_simple_4",
        "vignette_cot_demographics_first",
        "vignette_cot_clinical_first",
    )

    # Pull condition from these fields in order.
    condition_fields: Tuple[str, ...] = ("condition", "cohort")

    # Case statements used to build neutral/male/female case variants.
    neutral_case_prefix: str = "This patient's gender is not specified."
    male_case_prefix: str = "This patient is male."
    female_case_prefix: str = "This patient is female."

    # Baseline model settings (can change later).
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "auto"
    torch_dtype: str = "auto"
    hf_token_env_var: str = "HF_TOKEN"

    # Keep generation defaults together for later runs.
    max_new_tokens: int = 128
    temperature: float = 0.0


# Single shared config object across scripts.
CONFIG = ExperimentConfig()
