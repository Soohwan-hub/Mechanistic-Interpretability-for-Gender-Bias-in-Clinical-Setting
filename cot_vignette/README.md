# cot_vignette/

Chain-of-thought vignette prompt construction and patch-target utilities
(main-branch code).

| File | What it is |
|---|---|
| `prompts_config.py` | Vignette prompt templates/config |
| `patch_targets.py` | `get_patch_tensor`, `set_patch_tensor`, `patch_token_vector`, `extract_head_slice`, `patch_token_head_slice` — patch-target-aware helpers, used from `localize_bias/cot_vignette/` notebooks |
| `run_bhc_activation_patching.py` | Runs activation patching over BHC (de-gendered medical note) cases |
| `patching_snippets.md` | Usage snippets |

Supported patch targets: `residual`, `mlp`, `attn`, `attn_head`.
