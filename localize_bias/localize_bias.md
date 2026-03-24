## CoT Vignette
- Patch on residual stream 
- Run 2 prompt types ( A, C) x 10 prompt variations x 6 conditions = 120 runs
    - Conditions: 
### Results file
`results.pkl` contains:
rewrite_matrix — NumPy array of shape (28, n_tokens)
token_labels — list of token label strings for the column dimension
layer_hidden_states — the saved hidden states per layer (for offline logit lens analysis later without re-running nnsight traces)
prompt_name — which prompt type this run used (e.g. "VIGNETTE_PROMPT_A")
var — prompt variation
`var_{num}_layers_{num}_{num}.pkl` contains:
rewrite scores per batch. They serve as recovery files during the patching process, if the server crashes. They're not relevant for the aggregation and result analysis . 