import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import plotly.express as px
import plotly.io as pio
import torch
from nnsight import LanguageModel

from prompts_config import CORRUPT_PROMPTS


# --- Model and tokenizer ---
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
llm = LanguageModel(MODEL_NAME, dtype=torch.bfloat16, device_map="auto", load_in_4bit=True)


# --- Load depression cases from JSONL ---
def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


depression_cases = _load_jsonl(
    os.path.join(os.path.dirname(__file__), "data", "depression_cases.jsonl")
)


def prepare_clean_prompt(target_gender: str) -> Dict[str, Any]:
    """Prepare the clean prompt and locate the gender token position."""

    message = {
        "role": "user",
        "content": f"The patient is {target_gender}",
    }

    clean_prompt = llm.tokenizer.apply_chat_template(
        [message],
        tokenize=False,
        add_generation_prompt=True,
    )

    # `pt` returns PyTorch tensors.
    # `input_ids` is a sequence of integer token IDs of shape (batch, seq_len).
    # [0] selects the first batch -> (seq_len,).
    clean_tokens = llm.tokenizer(clean_prompt, return_tensors="pt")["input_ids"][0]

    print("\nClean tokens:    ", [llm.tokenizer.decode(t) for t in clean_tokens])

    # Get the token ids of the target gender only.
    target_token_ids = llm.tokenizer(
        " " + target_gender,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    # The last token of `target_gender` is the gender token.
    # We get the position where the clean tokens match the last target_gender token.
    # [0] gets the first occurrence of match, [0] unwraps the single-element tensor
    # to get the actual integer value; [0][0] leaves the tensor 0-dim. Therefore,
    # `.tolist()` returns a scalar, not a list. `patch_token_from` should be a
    # scalar value to use it for indexing later.
    patch_token_from = torch.argwhere(clean_tokens == target_token_ids[-1])[0][0].tolist()

    return {
        "clean_prompt": clean_prompt,
        "clean_tokens": clean_tokens,
        "patch_token_from": patch_token_from,
    }



def prepare_corrupt_prompt(prompt_name: str, condition: str) -> Dict[str, Any]:
    """Prepare the prompt for the corrupt run (the run with replaced activations)."""

    # Replace [CONDITION] placeholder with the actual condition name.
    # The templates use [CONDITION], not {}, so .replace() is required instead of .format().
    user_text = CORRUPT_PROMPTS[prompt_name].replace("[CONDITION]", condition)

    # Qwen2.5's chat template injects a default system message when none is provided.
    # An explicit empty system message suppresses it, since the user_text already
    # contains its own clinical instruction.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": user_text,
        }
    ]

    corrupted_prompt = llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("Corrupted prompt:", corrupted_prompt)

    corrupted_tokens = llm.tokenizer(
        corrupted_prompt,
        return_tensors="pt",
    )["input_ids"][0]
    print("Corrupted tokens:", [llm.tokenizer.decode(t) for t in corrupted_tokens])

    return {
        "corrupted_tokens": corrupted_tokens,
        "corrupted_prompt": corrupted_prompt,
    }


def plot_patching_results(
    rewrite_scores: np.ndarray,
    token_labels: List[str],
    layer_labels: List[int],
    title: str = "",
    path: str = "",
) -> None:
    """Render a heatmap of rewrite scores (layers x tokens) and optionally save as PDF."""

    fig = px.imshow(
        rewrite_scores,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Token", "y": "Layer", "color": " "},
        x=token_labels,
        y=[str(l) for l in layer_labels],
        title=title,
    )

    if path:
        pio.write_image(fig, path)


def run_patching_residual_stream(
    corrupt_prompt_name: str,
    original_gender: str,
    target_condition: str,
) -> Dict[str, Any]:
    """Perform activation patching on all layers and return rewrite scores.

    Works for one prompt variation and one clinical case.
    """

    # 1. ADJUST PATCH TOKEN FROM
 
    if original_gender == "Female":
        target_gender = "Male"
    else:
        target_gender = "Female"
    clean_prompt_output = prepare_clean_prompt(target_gender)
    clean_prompt = clean_prompt_output["clean_prompt"]
    clean_tokens = clean_prompt_output["clean_tokens"]
    patch_token_from = clean_prompt_output["patch_token_from"]

    corrupt_prompt_output = prepare_corrupt_prompt(corrupt_prompt_name, target_condition)
    corrupted_tokens = corrupt_prompt_output["corrupted_tokens"]
    corrupted_prompt = corrupt_prompt_output["corrupted_prompt"]

    diff = len(clean_tokens) - len(corrupted_tokens)
    offset = 0

    # Corrupted prompt is shorter -> the library will add left-pads to the corrupted
    # prompt so it becomes the same length as `clean_prompt`. The real token index is
    # then `token_idx + diff`.
    if diff > 0:
        offset = diff
        print(f"Corrupted prompt is shorter by {diff} tokens → offset = {offset}")
    else:
        patch_token_from = patch_token_from - diff
        print(f"Clean prompt is shorter by {-diff} tokens → patch_token_from adjusted to {patch_token_from}")

    # 2. PREPARE TOKEN ID OF MALE/FEMALE ANSWER:
    # so we know where to find Female/Male in the model's output distribution.
    original_gender_token = llm.tokenizer(f"  {original_gender}", add_special_tokens=False)["input_ids"][0]
    target_gender_token = llm.tokenizer(f"  {target_gender}", add_special_tokens=False)["input_ids"][0]

    print(f"\nOriginal gender token: {original_gender_token}-{original_gender}, Target gender token: {target_gender_token}-{target_gender}")

    # 3. PATCHING LOOP
    num_layers_batch = 5
    num_layers_model = len(llm.model.layers)
    softmax = torch.nn.Softmax(dim=-1)  # dim = -1 indicates over the vocabulary.

    # STEP 1: Cache clean activations once (a single forward pass, no generation needed).
    # Previously this was re-run inside every batch loop iteration, wasting ceil(N/batch)-1
    # redundant forward passes. By saving real tensors here, the proxies are resolved
    # before the patching loop, so each batch no longer needs its own clean run.
    print("\nCaching clean activations (one-time forward pass)...")
    _saved_clean: Dict[int, Any] = {}
    with torch.no_grad():
        with llm.generate(max_new_tokens=1) as tracer:
            with tracer.invoke(clean_prompt):
                for layer_idx in range(num_layers_model):
                    activations_clean = llm.model.layers[layer_idx].output[0]
                    _saved_clean[layer_idx] = activations_clean[:, patch_token_from, :].save()
    # Resolve proxies → concrete tensors that are valid across tracer boundaries.
    clean_activations_cache: Dict[int, torch.Tensor] = {
        i: proxy.value.detach().clone() for i, proxy in _saved_clean.items()
    }

    # STEP 2: Baseline corrupted run — also run once and cache the scalar probability.
    # first lm_head output is "Gender", .next() is ":", .next() is the gender token.
    print("Running corrupted baseline (one-time forward pass)...")
    with torch.no_grad():
        with llm.generate(max_new_tokens=3) as tracer:
            with tracer.invoke(corrupted_prompt):
                corrupted_logits = llm.lm_head.next().next().output
                _corrupted_highest_pred = corrupted_logits[0].argmax(dim=-1).save()
                _corrupted_prob = softmax(corrupted_logits[0][0])[original_gender_token].save()
    corrupted_prob_value: float = _corrupted_prob.value.item()
    total_corrupted_highest_predictions: List[Any] = [_corrupted_highest_pred.value]

    rewrite_scores: List[Any] = []
    patched_predictions: List[Any] = []

    for start in range(0, num_layers_model, num_layers_batch):
        end = min(start + num_layers_batch, num_layers_model)
        print(f"\nProcessing layers {start}–{end - 1} of {num_layers_model - 1} ...")

        with torch.no_grad():
            # Only patching runs remain here; clean and baseline are pre-cached above.
            # max_new_tokens=3 is sufficient: prefill + "Gender" + ":" + gender token.
            with llm.generate(max_new_tokens=3) as tracer:

                # STEP 3: PATCH each (LAYER, TOKEN)
                for layer_idx in range(start, end):
                    for token_idx in range(len(corrupted_tokens)):
                        with tracer.invoke(corrupted_prompt):
                            activations_corrupt = llm.model.layers[layer_idx].output[0]
                            # Inject the cached clean residual-stream vector at this position.
                            activations_corrupt[:, token_idx + offset, :] = clean_activations_cache[layer_idx]
                            # Put the patched activation back into the residual stream.
                            llm.model.layers[layer_idx].output[0] = activations_corrupt

                            patched_logits = llm.lm_head.next().next().output
                            patched_prediction = patched_logits[0].argmax(dim=-1).save()
                            patched_prob = softmax(patched_logits[0][0])[target_gender_token]

                            rewrite_score = (patched_prob - corrupted_prob_value) / (1 - patched_prob)
                            rewrite_scores.append(rewrite_score.save())
                            patched_predictions.append(patched_prediction)

    output = plot_results(rewrite_scores, corrupted_tokens, patched_predictions)
    return output


def plot_results(
    rewrite_scores: List[Any],
    corrupted_tokens: Any,
    patched_predictions: List[Any],
) -> Dict[str, Any]:
    """Aggregate rewrite scores, save them, and plot results."""

    # Move each score to CPU and extract scalar.
    rewrite_scores = [score.cpu().float().item() for score in rewrite_scores]
    rewrite_scores_array = np.array(rewrite_scores)

    num_layers_model = int(len(rewrite_scores_array) / len(corrupted_tokens)) if len(corrupted_tokens) > 0 else 0
    if num_layers_model > 0:
        rewrite_scores_array = np.reshape(rewrite_scores_array, (num_layers_model, -1))
    else:
        rewrite_scores_array = np.array([])

    # Build human-readable token labels from the corrupted prompt for the heatmap x-axis.
    corrupted_decoded = [llm.tokenizer.decode(t) for t in corrupted_tokens]
    token_labels = [f"{tok}_{i}" for i, tok in enumerate(corrupted_decoded)]
    layer_labels = list(range(num_layers_model))

    results = {
        "token_labels": token_labels,       # x-axis: token string + position index
        "layer_labels": layer_labels,       # y-axis: integer layer indices 0 … N_LAYERS-1
        "rewrite_scores": rewrite_scores_array,  # 2-D float array, shape (N_LAYERS, N_TOKENS)
        "patched_preds": [p.cpu().float().item() for p in patched_predictions],
    }

    scores_path = "rewrite_scores.pkl"
    with open(scores_path, "wb") as f:
        pickle.dump(results, f)  # serialise results dict; load later with pickle.load()

    print(f"\nResults saved → {scores_path}")

    # Layer 0 is often excluded from plots because its activations mostly reflect embeddings.
    if num_layers_model > 1 and rewrite_scores_array.size > 0:
        plot_path = "rewrite_scores_plot.pdf"
        plot_patching_results(
            rewrite_scores_array[1:, :],  # all layers except layer 0
            token_labels,
            layer_labels[1:],
            title="Residual Stream Activation Patching",
            path=plot_path,
        )
        print(f"Plot saved      → {plot_path}")

    return results


def run_all_prompt_variations() -> None:
    """Run activation patching on all target conditions and prompt variations."""

    # {"condition": "original_gender"}
    target_conditions = {
        "heart failure": "Male",
        "depression": "Female",
    }

    for condition, gender in target_conditions.items():
        for prompt_name in CORRUPT_PROMPTS:
            run_patching_residual_stream(
                corrupt_prompt_name=prompt_name,
                original_gender=gender,
                target_condition=condition,
            )


if __name__ == "__main__":
    # Run a single test: first depression case, cot_vignette_A prompt.
    # case["cohort"] == "Depression" → fills [CONDITION] in the prompt template.
    case = depression_cases[0]
    print("case: ", case)
    run_patching_residual_stream(
        corrupt_prompt_name="cot_vignette_A",
        original_gender="Female",
        target_condition=case["cohort"].lower(),  # "depression"
    )
