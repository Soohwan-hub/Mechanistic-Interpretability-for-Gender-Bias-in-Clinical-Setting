# Token Indexing Validation Changes
3/13 7pm
This document tracks the validation changes added to `simple_patching.py` to make gender token indexing and target selection explicit, fail-fast, and reproducible.

## 1) Added strict gender token validation helper

### Why
- Prevent silent tokenizer behavior changes from corrupting analysis.
- Ensure only `male`/`female` are used for this pipeline.
- Ensure `" " + gender` is exactly one token for this setup.
- Ensure decoded token text still corresponds to expected gender terms.

### Before
```python
# No dedicated validation helper existed.
```

### After
```python
def _validated_gender_token_ids(llm: LanguageModel, gender: str) -> torch.Tensor:
    """Return token ids for ' {gender}' with strict validation for this analysis."""
    normalized = gender.strip().lower()
    assert normalized in {"male", "female"}, (
        f"Expected gender to be male/female, got {gender!r}"
    )
    token_ids = llm.tokenizer(
        " " + gender,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    assert int(token_ids.numel()) == 1, (
        f"Expected ' {gender}' to map to a single token, got {int(token_ids.numel())} "
        f"tokens: {token_ids.tolist()} (decoded={llm.tokenizer.decode(token_ids.tolist())!r})"
    )
    decoded = llm.tokenizer.decode(token_ids.tolist()).strip().lower()
    assert decoded in {"male", "female"}, (
        f"Token decode check failed for gender={gender!r}: decoded={decoded!r}, ids={token_ids.tolist()}"
    )
    return token_ids
```

---

## 2) Source patch token selection now enforces uniqueness

### Why
- Previous logic selected the first occurrence of `token_ids[-1]` in `clean_tokens`.
- If that token ID appears multiple times, source position can be wrong.
- New logic asserts exactly one match before selecting `patch_token_from`.

### Before
```python
token_ids = llm.tokenizer(
    " " + gender,
    return_tensors="pt",
    add_special_tokens=False,
)["input_ids"][0]
patch_token_from = torch.argwhere(clean_tokens == token_ids[-1])[0][0].item()
```

### After
```python
token_ids = _validated_gender_token_ids(llm, gender)
matches = torch.argwhere(clean_tokens == token_ids[-1])
assert matches.shape[0] == 1, (
    f"Expected exactly one source token match for gender={gender!r}, found {matches.shape[0]} "
    f"for token_id={int(token_ids[-1].item())}"
)
patch_token_from = matches[0][0].item()
```

---

## 3) `target_id` now uses the same validated token path

### Why
- Previously, `target_id` repeated raw tokenization logic independently.
- This could diverge from source indexing assumptions.
- New logic keeps one shared, validated path for both source/target behavior.

### Before
```python
target_id = llm.tokenizer(" " + gender, add_special_tokens=False)["input_ids"][-1]
```

### After
```python
target_id = int(_validated_gender_token_ids(llm, gender)[-1].item())
```

---

## Practical effect

- If tokenizer behavior matches expected assumptions, run behavior is unchanged.
- If assumptions fail (multi-token split, decode mismatch, duplicate source matches), the script now raises a clear assertion error instead of producing potentially misleading patching scores.
