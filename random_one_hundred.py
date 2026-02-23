import pandas as pd

df = pd.read_json('verified_dataset.jsonl', lines=True)
cols_to_drop = ['verification_result', 'original_text', 'level_1']
df = df.drop(columns=cols_to_drop)
df_sampled = df.sample(frac=1, random_state=42).groupby('cohort').head(100)
df_sampled = df_sampled.sort_values('cohort')
df_sampled.to_json('final_sampled_data.jsonl', orient='records', lines=True)