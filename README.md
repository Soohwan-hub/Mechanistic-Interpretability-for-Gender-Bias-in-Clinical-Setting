# Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting


Create and use ur virtual env using pip
Make sure when installing using pip, you update the requirments.txt file (pip freeze > requirements.txt)
can install updated requirements.txt with (pip install -r requirements.txt). Nothing is in there rn

## CoT patch target switch

For `cot_vignette/activation_patching.ipynb`, you can now import:

`from cot_vignette.patch_targets import get_patch_tensor, set_patch_tensor, patch_token_vector, extract_head_slice, patch_token_head_slice`

and replace the hardcoded residual-stream lines:

`act = llm.model.layers[layer_idx].output[0]`

with patch-target-aware calls.

Supported patch targets:
- `residual`
- `mlp`
- `attn`
- `attn_head`
