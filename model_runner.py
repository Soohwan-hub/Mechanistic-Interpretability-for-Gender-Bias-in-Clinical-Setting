from dataclasses import dataclass
from typing import Dict, Optional

from config import CONFIG


# ============================================================
# 1. MODEL SETUP
# ============================================================

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


@dataclass
class ModelBundle:
    tokenizer: object
    model: object
    device: str


def dependencies_available() -> bool:
    # Quick guard before trying to load model.
    return bool(torch and AutoModelForCausalLM and AutoTokenizer)


def load_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: Optional[str] = None,
) -> ModelBundle:
    if not dependencies_available():
        raise ImportError(
            "Missing dependencies for model loading. "
            "Install at least torch and transformers."
        )

    model_name = model_name or CONFIG.model_name
    device = device or CONFIG.device
    torch_dtype = torch_dtype or CONFIG.torch_dtype

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = {"device_map": device}
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    return ModelBundle(tokenizer=tokenizer, model=model, device=resolved_device)


# ============================================================
# 2. BASELINE SCORING HELPERS (PLACEHOLDER)
# ============================================================

def score_prompt_logprob(bundle: ModelBundle, prompt: str) -> float:
    """
    Lightweight placeholder score.
    Computes average next-token logprob over the prompt tokens.
    """
    with torch.no_grad():
        enc = bundle.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(bundle.model.device)

        if input_ids.size(1) < 2:
            return 0.0

        outputs = bundle.model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        return float(token_log_probs.mean().item())


def score_triplet_prompts(bundle: ModelBundle, prompts: Dict[str, str]) -> Dict[str, float]:
    required_keys = {"neutral", "male", "female"}
    missing = required_keys - prompts.keys()
    if missing:
        raise ValueError(f"Triplet prompts missing keys: {sorted(missing)}")

    return {
        "neutral": score_prompt_logprob(bundle, prompts["neutral"]),
        "male": score_prompt_logprob(bundle, prompts["male"]),
        "female": score_prompt_logprob(bundle, prompts["female"]),
    }


def score_prompt_triplet_collection(
    bundle: ModelBundle,
    prompt_triplets: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, float]]:
    # Returns scores keyed by prompt ID, then by neutral/male/female.
    scores: Dict[str, Dict[str, float]] = {}
    for prompt_id, triplet in prompt_triplets.items():
        scores[prompt_id] = score_triplet_prompts(bundle, triplet)
    return scores
