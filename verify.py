import pandas as pd
from vllm import LLM, SamplingParams

INPUT_FILE = "specified_count_sampled_candidates.jsonl"
OUTPUT_FILE = "verified_dataset.jsonl"
# CHANGE THIS if you are using the 70B model or GH200 (FP8)
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct" 

print("Loading data...")
df = pd.read_json(INPUT_FILE, lines=True)

# Filter out "Comorbid" cases so the LLM doesn't get confused
print(f"Original count: {len(df)}")
df = df[df['cohort'] != 'Comorbid']
print(f"Count after removing Comorbid: {len(df)}")

def create_prompt(text, cohort):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert clinical auditor.
Target Diagnosis: {cohort.upper()}

Your Job:
1. Read the Clinical Note below.
2. Determine if the patient was admitted primarily for {cohort}.
3. Answer NO if they have {cohort} but were admitted for something else (e.g., gallbladder surgery).
4. Answer NO if the text is about a different disease.
5. Answer YES only if {cohort} is the clear, primary reason for this hospital stay.

Reply ONLY with "YES" or "NO".

<|eot_id|><|start_header_id|>user<|end_header_id|>
Clinical Note:
{text[:3500]}

Is {cohort} the primary diagnosis?
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

print("Creating prompts...")
prompts = [create_prompt(row['text'], row['cohort']) for _, row in df.iterrows()]

print(f"Loading Model: {MODEL_NAME}...")
# Note: Add tensor_parallel_size=2 here if using 2x H100s
llm = LLM(model=MODEL_NAME)

print("Running inference on GPU...")
params = SamplingParams(temperature=0, max_tokens=5)
outputs = llm.generate(prompts, params)

results = [output.outputs[0].text.strip().upper() for output in outputs]
df['verification_result'] = results

# Filter: Keep only the "YES" rows
verified_df = df[df['verification_result'].str.contains("YES")]

verified_df.to_json(OUTPUT_FILE, orient='records', lines=True)

print(f"Done! Verified {len(verified_df)} out of {len(df)} candidates.")
print(f"Saved to {OUTPUT_FILE}")