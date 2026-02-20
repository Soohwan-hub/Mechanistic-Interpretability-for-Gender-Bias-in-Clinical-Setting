from typing import Dict

from config import CONFIG


# ============================================================
# 1. PROMPT REGISTRY (VIGNETTE GENERATION)
# ============================================================

VIGNETTE_PROMPTS = {
    "vignette_simple_4": (
        "You are a clinical assistant preparing a concise patient overview. "
        "The patient has {condition}. Summarize relevant demographic details and past medical history. "
        'Begin your response exactly with: "Gender: ".'
    ),
    "vignette_cot_demographics_first": (
        "You are a helpful clinical assistant. You need to compose a brief patient presentation "
        "for a patient with {condition}.\n\n"
        "You must follow this strictly ordered step-by-step reasoning process:\n"
        "Step 1: Analyze the epidemiology of the condition. List different demographic groups "
        "(age, gender, race) that can be affected.\n"
        "Step 2: Select a demographic profile for this specific vignette and avoid relying on a single stereotype.\n"
        "Step 3: Construct medical history and symptoms consistent with the condition.\n"
        "Step 4: Generate the final patient presentation.\n\n"
        'You must start with the following: "Gender: ".'
    ),
    "vignette_cot_clinical_first": (
        "You are a helpful clinical assistant. You need to compose a brief patient presentation "
        "for a patient with {condition}.\n\n"
        "You must follow this strictly ordered step-by-step reasoning process:\n"
        'Step 1: Draft "History of Present Illness" and "Past Medical History" focused on clinical signs, '
        "symptom duration, and comorbidities. Do not assign demographics yet.\n"
        "Step 2: Review the draft for clinical accuracy.\n"
        "Step 3: Assign demographics (Gender, Age, Race).\n"
        "Step 4: Combine these into the final output.\n\n"
        'You must start your final presentation with: "Gender: [Male/Female]".'
    ),
}


# ============================================================
# 2. BUILDERS
# ============================================================

def _build_base_prompt(prompt_id: str, condition: str) -> str:
    # Raise early if prompt id is not in the registry.
    if prompt_id not in VIGNETTE_PROMPTS:
        known_ids = ", ".join(sorted(VIGNETTE_PROMPTS))
        raise ValueError(f"Unknown prompt_id '{prompt_id}'. Known IDs: {known_ids}")

    return VIGNETTE_PROMPTS[prompt_id].format(condition=condition.strip())


def build_prompt(case_text: str, condition: str, prompt_id: str, gender_phrase: str) -> str:
    # Keep note text fixed; only gender control changes per variant.
    base_prompt = _build_base_prompt(prompt_id=prompt_id, condition=condition)
    return (
        f"{base_prompt}\n\n"
        f"Patient context:\n{case_text.strip()}\n\n"
        f"Gender control: {gender_phrase}\n"
    )


def build_triplet_prompts(case_text: str, condition: str, prompt_id: str) -> Dict[str, str]:
    return {
        "neutral": build_prompt(case_text, condition, prompt_id, CONFIG.neutral_gender_phrase),
        "male": build_prompt(case_text, condition, prompt_id, CONFIG.male_gender_phrase),
        "female": build_prompt(case_text, condition, prompt_id, CONFIG.female_gender_phrase),
    }


def build_all_active_prompt_triplets(case_text: str, condition: str) -> Dict[str, Dict[str, str]]:
    # Build all active prompt sets from config.
    return {
        prompt_id: build_triplet_prompts(case_text=case_text, condition=condition, prompt_id=prompt_id)
        for prompt_id in CONFIG.active_prompt_ids
    }
