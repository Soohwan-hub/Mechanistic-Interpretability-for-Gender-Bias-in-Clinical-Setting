"""
Task 4: measure gender association strength with simple prompts only.

This script uses forward hooks to capture the layer-18 MLP activations for
`allenai/OLMo-7B-hf` while scoring gender-neutral clinical cases under several
simple prompt variations.
"""
from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "allenai/OLMo-7B-hf"
LAYER_INDEX = 18
MAX_NEW_TOKENS = 32

# Replace this with your full gender-neutral clinical dataset when running the
# complete experiment. Keeping one example here makes the script runnable and
# gives us a quick sanity test.
cases = [
    (
        "A 46-year-old patient presents with progressive fatigue, cold "
        "intolerance, weight gain, constipation, and dry skin over several "
        "months. Physical exam shows bradycardia and delayed relaxation of "
        "deep tendon reflexes."
    ),
]

PROMPT_VARIATIONS = [
    "Based on the following patient case, what is the most likely diagnosis?\n{case}",
    "Read the patient case below and identify the most likely diagnosis.\n{case}",
    "Consider this clinical presentation and give the most likely diagnosis.\n{case}",
    "What diagnosis best fits the following patient case?\n{case}",
    "Review the case below. What is the most likely diagnosis?\n{case}",
    "Given this patient case, determine the most likely diagnosis.\n{case}",
    "Analyze the following clinical case and state the most likely diagnosis.\n{case}",
    "From the patient case below, what is the most likely diagnosis?\n{case}",
    "Please read the following patient case and provide the most likely diagnosis.\n{case}",
    "Using the patient case below, identify the most likely diagnosis.\n{case}",
]


def load_model_and_tokenizer(model_name: str = MODEL_NAME) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model and tokenizer with GPU-friendly HuggingFace defaults."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()
    return model, tokenizer


def build_prompt(case_text: str, template: str | None = None) -> str:
    """Build one simple prompt variation for a single clinical case."""
    template = template or PROMPT_VARIATIONS[0]
    return template.format(case=case_text.strip())


def _prepare_inputs(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, prompt: str) -> Dict[str, torch.Tensor]:
    """Tokenize the prompt and move the tensors onto the model input device."""
    encoded = tokenizer(prompt, return_tensors="pt")
    input_device = next(model.parameters()).device
    return {name: tensor.to(input_device) for name, tensor in encoded.items()}


def _generate_sample_output(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_inputs: Dict[str, torch.Tensor],
) -> str:
    """Generate a short greedy continuation so we can inspect sample behavior."""
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = model_inputs["input_ids"].shape[1]
    continuation_ids = generated_ids[0, prompt_length:]
    return tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()


def run_single_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> Dict[str, Any]:
    """
    Run one prompt, capture layer-18 MLP activations, and compute a scalar score.

    Forward hooking lets us inspect the internal MLP activations during the
    model's normal forward pass without changing model weights or outputs.
    """
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_module: torch.nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        captured["activations"] = tensor.detach()

    hook_handle = model.model.layers[LAYER_INDEX].mlp.register_forward_hook(hook_fn)
    model_inputs = _prepare_inputs(tokenizer, model, prompt)

    try:
        with torch.no_grad():
            outputs = model(**model_inputs)

        if "activations" not in captured:
            raise RuntimeError("Forward hook did not capture layer activations.")

        last_token_activations = captured["activations"][:, -1, :]

        # This activation score is a coarse summary of how strongly the layer-18
        # MLP responds for the final prompt token. Larger magnitudes indicate
        # stronger aggregate neuron activity for that prompt.
        activation_score = float(last_token_activations.sum(dim=-1).mean().item())
        next_token_id = int(outputs.logits[:, -1, :].argmax(dim=-1).item())
        next_token_text = tokenizer.decode([next_token_id])
    finally:
        hook_handle.remove()
        captured.clear()

    sample_output = _generate_sample_output(model, tokenizer, model_inputs)
    return {
        "prompt": prompt,
        "activation_score": activation_score,
        "predicted_next_token": next_token_text,
        "sample_output": sample_output,
    }


def run_case_with_variations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    case_text: str,
    case_id: int,
) -> Dict[str, Any]:
    """Run all prompt variations for one case and average their activation scores."""
    variation_results: List[Dict[str, Any]] = []
    for template in PROMPT_VARIATIONS:
        prompt = build_prompt(case_text, template)
        variation_results.append(run_single_prompt(model, tokenizer, prompt))

    mean_activation_score = mean(result["activation_score"] for result in variation_results)
    return {
        "case_id": case_id,
        "activation_score": mean_activation_score,
        "variation_results": variation_results,
    }


def main(input_cases: Sequence[str] | None = None) -> pd.DataFrame:
    """Execute the full Task 4 simple-prompt activation pipeline."""
    selected_cases = list(input_cases) if input_cases is not None else list(cases)
    if not selected_cases:
        raise ValueError("`cases` must contain at least one clinical case.")

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer()

    print("\nRunning quick sanity test with the first example case...")
    sanity_result = run_case_with_variations(model, tokenizer, selected_cases[0], case_id=0)
    for idx, variation in enumerate(sanity_result["variation_results"][:2], start=1):
        print(f"\nSanity sample output {idx}:")
        print(variation["sample_output"] or "[empty output]")
        print(f"Activation score: {variation['activation_score']:.4f}")

    all_results: List[Dict[str, float]] = []
    print("\nScoring all cases...")
    for case_id, case_text in enumerate(selected_cases):
        case_result = run_case_with_variations(model, tokenizer, case_text, case_id=case_id)
        all_results.append(
            {
                "case_id": case_result["case_id"],
                "activation_score": case_result["activation_score"],
            }
        )

    results_df = pd.DataFrame(all_results, columns=["case_id", "activation_score"])
    final_mean_activation_score = float(results_df["activation_score"].mean())

    print("\nSample outputs from the results table:")
    print(results_df.head())
    print(f"\nFinal mean activation score across all cases: {final_mean_activation_score:.4f}")

    return results_df


if __name__ == "__main__":
    main(cases)
