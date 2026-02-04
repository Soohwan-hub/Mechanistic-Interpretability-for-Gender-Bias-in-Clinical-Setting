import pandas as pd

df = pd.read_json('filtered_dataset.jsonl', lines=True)
print(df['cohort'].value_counts())

randomly_sampled_df = df.groupby('cohort').apply(lambda x: x.sample(n=min(len(x), 700), random_state=42)).reset_index(drop=False)
OUTPUT_FILE = 'sampled_candidates.jsonl'
randomly_sampled_df.to_json(OUTPUT_FILE, orient='records', lines=True)

print(f"\nSaved {len(randomly_sampled_df)} entries to {OUTPUT_FILE}")
print(randomly_sampled_df['cohort'].value_counts())