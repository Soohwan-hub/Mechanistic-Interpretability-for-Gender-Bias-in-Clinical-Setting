import time

from config import CONFIG
from model_runner import dependencies_available, forward_with_hidden_states, load_model
from prompts import build_all_active_prompt_triplets


# ============================================================
# 1. SMALL GPU SMOKE TEST
# ============================================================

# Five quick examples to verify model load + prompt path + GPU forward.
SMOKE_CASES = [
    {
        "condition": "Heart Failure",
        "text": "Patient reports dyspnea on exertion, orthopnea, and leg swelling for two weeks.",
    },
    {
        "condition": "Depression",
        "text": "Patient reports low mood, poor sleep, loss of interest, and anxiety symptoms.",
    },
    {
        "condition": "Asthma",
        "text": "Patient has wheezing, chest tightness, and intermittent shortness of breath.",
    },
    {
        "condition": "Pneumonia",
        "text": "Patient has fever, productive cough, pleuritic chest pain, and fatigue.",
    },
    {
        "condition": "Diabetes",
        "text": "Patient has polyuria, polydipsia, fatigue, and elevated glucose on recent labs.",
    },
]


def _build_case_variants(case_text: str):
    text = case_text.strip()
    return {
        "neutral": f"{CONFIG.neutral_case_prefix} {text}",
        "male": f"{CONFIG.male_case_prefix} {text}",
        "female": f"{CONFIG.female_case_prefix} {text}",
    }


def _generate_short_text(bundle, prompt: str) -> str:
    model_inputs = bundle.tokenizer(prompt, return_tensors="pt")
    model_inputs = {k: v.to(bundle.model.device) for k, v in model_inputs.items()}

    pad_token_id = bundle.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = bundle.tokenizer.eos_token_id

    output_ids = bundle.model.generate(
        **model_inputs,
        max_new_tokens=24,
        do_sample=False,
        temperature=0.0,
        pad_token_id=pad_token_id,
    )

    new_tokens = output_ids[0, model_inputs["input_ids"].shape[1]:]
    return bundle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    print("Starting Qwen smoke test...")

    if not dependencies_available():
        raise RuntimeError("Missing torch/transformers. Install dependencies first.")

    bundle = load_model()
    print(f"Model loaded: {CONFIG.model_name}")
    print(f"Resolved device: {bundle.device}")

    prompt_id = CONFIG.active_prompt_ids[0]
    print(f"Using prompt template: {prompt_id}")

    total_start = time.time()
    for idx, item in enumerate(SMOKE_CASES, start=1):
        case_variants = _build_case_variants(item["text"])
        prompt_triplets = build_all_active_prompt_triplets(
            case_variants=case_variants,
            condition=item["condition"],
            active_prompt_ids=(prompt_id,),
        )
        prompt = prompt_triplets[prompt_id]["neutral"]

        start = time.time()
        outputs = forward_with_hidden_states(bundle, prompt)
        text = _generate_short_text(bundle, prompt)
        elapsed = time.time() - start

        num_hidden = len(outputs.hidden_states) if outputs.hidden_states is not None else 0
        print(f"\nRun {idx}/5")
        print(f"Condition: {item['condition']}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Hidden states returned: {num_hidden}")
        print(f"Sample output: {text[:200]}")

    total_elapsed = time.time() - total_start
    print(f"\nSmoke test complete. Total time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
