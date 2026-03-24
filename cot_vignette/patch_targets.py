from typing import Any


# ============================================================
# 1. PATCH TARGET HELPERS
# ============================================================
#
# These helpers are designed for the existing nnsight notebook flow.
# Use patch_target to switch where activation patching is applied:
#   - "residual"   : decoder layer residual stream (existing behavior)
#   - "mlp"        : MLP block output
#   - "attn"       : attention block output (o_proj output)
#   - "attn_head"  : attention head slice (o_proj input, flat head layout)
#
# NOTE: "attn_head" patches one head slice inside the concatenated
# [num_heads * head_dim] vector.
#


def get_patch_tensor(layer: Any, patch_target: str):
    if patch_target == "residual":
        return layer.output[0]
    if patch_target == "mlp":
        return layer.mlp.output[0]
    if patch_target == "attn":
        return layer.self_attn.o_proj.output[0]
    if patch_target == "attn_head":
        return layer.self_attn.o_proj.input[0]
    raise ValueError(f"Unknown patch_target: {patch_target}")


def set_patch_tensor(layer: Any, patch_target: str, patched_tensor: Any) -> None:
    if patch_target == "residual":
        layer.output[0] = patched_tensor
        return
    if patch_target == "mlp":
        layer.mlp.output[0] = patched_tensor
        return
    if patch_target == "attn":
        layer.self_attn.o_proj.output[0] = patched_tensor
        return
    if patch_target == "attn_head":
        layer.self_attn.o_proj.input[0] = patched_tensor
        return
    raise ValueError(f"Unknown patch_target: {patch_target}")


def patch_token_vector(
    layer: Any,
    patch_target: str,
    token_idx: int,
    clean_vector: Any,
) -> None:
    act = get_patch_tensor(layer, patch_target)
    act[token_idx, :] = clean_vector
    set_patch_tensor(layer, patch_target, act)


# ============================================================
# 2. ATTENTION HEAD HELPERS
# ============================================================

def _head_slice_bounds(hidden_size: int, num_heads: int, head_idx: int):
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) not divisible by num_heads ({num_heads})"
        )
    if head_idx < 0 or head_idx >= num_heads:
        raise ValueError(f"head_idx {head_idx} out of range [0, {num_heads - 1}]")

    head_dim = hidden_size // num_heads
    start = head_idx * head_dim
    end = start + head_dim
    return start, end


def extract_head_slice(flat_vector: Any, num_heads: int, head_idx: int):
    hidden_size = flat_vector.shape[-1]
    start, end = _head_slice_bounds(hidden_size, num_heads, head_idx)
    return flat_vector[start:end]


def patch_token_head_slice(
    layer: Any,
    token_idx: int,
    clean_head_slice: Any,
    num_heads: int,
    head_idx: int,
) -> None:
    # For head-level patching use the flattened attention output before o_proj.
    act = get_patch_tensor(layer, "attn_head")
    hidden_size = act.shape[-1]
    start, end = _head_slice_bounds(hidden_size, num_heads, head_idx)
    act[token_idx, start:end] = clean_head_slice
    set_patch_tensor(layer, "attn_head", act)
