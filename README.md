# Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting

# Files Structure

localize_bias/
    cot_vignette/
    simple/

build_dataset/
    dataset/
    - BHCs of depression and heart failure in json

# Research Stages

## 1. Localize Bias

### Number of Runs
#### CoT:
(3 Prompt Types x 5 Prompt Variations x 30 BHC cases) x 2 conditions = 900 runs
#### Simple:

### Metric: Average Rewrite Score Per Layer

## CoT patch target switch

For `cot_vignette/activation_patching.ipynb`, import:

`from cot_vignette.patch_targets import get_patch_tensor, set_patch_tensor, patch_token_vector, extract_head_slice, patch_token_head_slice`

and replace hardcoded residual-stream lines (`llm.model.layers[layer_idx].output[0]`) with patch-target-aware calls.

Supported patch targets:
- `residual`
- `mlp`
- `attn`
- `attn_head`
