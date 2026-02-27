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
        #TODO: Do we need the cot_swap prompt? created by Rundraash?
    )

    # Pull condition from these fields in order.
    condition_fields: Tuple[str, ...] = ("condition", "cohort")

    # Gender control strings used for neutral/male/female variants.
    neutral_gender_phrase: str = "Use Gender: Unknown for this vignette."
    male_gender_phrase: str = "Use Gender: Male for this vignette."
    female_gender_phrase: str = "Use Gender: Female for this vignette."

    # Baseline model settings (can change later).
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "auto"
    torch_dtype: str = "auto"


# Single shared config object across scripts.
CONFIG = ExperimentConfig()
