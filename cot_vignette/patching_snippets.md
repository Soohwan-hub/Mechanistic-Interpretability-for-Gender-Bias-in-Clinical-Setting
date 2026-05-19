# CoT Activation Patching: Target Switch Snippets

Use these snippets in `cot_vignette/activation_patching.ipynb` to test patching on
residual stream, MLP block, attention block, or attention head slices.

## 1) Add imports + config

```python
from cot_vignette.patch_targets import (
    get_patch_tensor,
    patch_token_vector,
    extract_head_slice,
    patch_token_head_slice,
)

# one of: "residual", "mlp", "attn", "attn_head"
patch_target = "mlp"

# only used for patch_target == "attn_head"
head_idx = 0
num_heads = llm.model.config.num_attention_heads
```

## 2) Replace clean-cache block

```python
_saved_clean = {}
with torch.no_grad():
    with llm.generate(max_new_tokens=1) as tracer:
        with tracer.invoke(clean_prompt):
            for layer_idx in range(num_layers_model):
                layer = llm.model.layers[layer_idx]
                act = get_patch_tensor(layer, patch_target)
                vec = act[patch_token_from, :]

                if patch_target == "attn_head":
                    vec = extract_head_slice(vec, num_heads=num_heads, head_idx=head_idx)

                _saved_clean[layer_idx] = vec.save()

clean_activations_cache = {
    i: resolve(proxy).detach().clone() for i, proxy in _saved_clean.items()
}
```

## 3) Replace patching lines in loop

Replace this:

```python
act = llm.model.layers[layer_idx].output[0]
act[token_idx + offset, :] = clean_activations_cache[layer_idx]
llm.model.layers[layer_idx].output[0] = act
```

With this:

```python
layer = llm.model.layers[layer_idx]
if patch_target == "attn_head":
    patch_token_head_slice(
        layer=layer,
        token_idx=token_idx + offset,
        clean_head_slice=clean_activations_cache[layer_idx],
        num_heads=num_heads,
        head_idx=head_idx,
    )
else:
    patch_token_vector(
        layer=layer,
        patch_target=patch_target,
        token_idx=token_idx + offset,
        clean_vector=clean_activations_cache[layer_idx],
    )
```

## 4) Optional output naming

```python
batch_path = os.path.join(
    save_dir,
    f"{patch_target}_layers_{start}_{end-1}.pkl"
)
```

For head-level runs:

```python
batch_path = os.path.join(
    save_dir,
    f"{patch_target}_h{head_idx}_layers_{start}_{end-1}.pkl"
)
```
