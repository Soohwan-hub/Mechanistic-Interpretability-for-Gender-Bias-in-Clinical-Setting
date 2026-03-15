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
(3 Prompt Types x 5 Prompt Variations x 30 BHC cases ) x 2 conditions = 900 runs 
#### Simple:

### Metric: Average Rewrite Score Per Layer

Good, I have enough context. Let me write comprehensive notes on rewrite score aggregation for your specific setup.

---

## Rewrite Score Aggregation Notes
### Setup: 30 BHC cases per condition (Depression + Heart Failure), 3 prompt types × 10 paraphrases each


For a single run (one BHC case + one prompt variation), activation patching sweeps every **(layer, token)** pair. After patching at layer `l`, token position `i`, we have (28 layers x n tokens) rewrite scores, resulting in 28 reewrite scores.

Each of the 3 prompt types (A, B, C) has 5 prompt variations. For a single BHC case and a single prompt type, we run activation patching 5 times (once per paraphrase), each yielding a `(28,)` layer score vector.

We compute the average of the 28 rewrite scores across prompt variations within a prompt type.  

Finally,we compute the average of the 28 rewrite scoress across 30 BHC cases withina prompt type.

Each prompt type has 28 rewrite scores by layer.
