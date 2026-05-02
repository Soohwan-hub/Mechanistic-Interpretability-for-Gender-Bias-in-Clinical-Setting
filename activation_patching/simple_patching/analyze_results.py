import pickle
import numpy as np
import os

conditions = ['asthma', 'depression', 'multiple_sclerosis', 'rheumatoid_arthritis', 'sarcoidosis']
condition_means = {}

for condition in conditions:
    scores = []
    for pid in range(1, 32):
        fp = f'olmo31_rewrite_only/artifacts/{condition}_prompt{pid}.pkl'
        if os.path.exists(fp):
            with open(fp, 'rb') as f:
                data = pickle.load(f)
                # Truncate to first 38 tokens
                scores.append(data['rewrite_scores'][:, :38])
    if scores:
        all_scores = np.array(scores)  # (31, 32, 50)
        condition_means[condition] = np.mean(all_scores, axis=0)  # (32, 50)

# Best condition
overall_scores = {cond: np.mean(scores) for cond, scores in condition_means.items()}
best_condition = max(overall_scores, key=overall_scores.get)
print(f"Best condition overall: {best_condition} with mean rewrite score: {overall_scores[best_condition]:.6f}")

# Best layer across all conditions
layer_scores = np.mean(list(condition_means.values()), axis=(0, 2))  # average over conditions and tokens
best_layer = np.argmax(layer_scores)
print(f"Best layer overall: {best_layer} with mean rewrite score: {layer_scores[best_layer]:.6f}")

# Print all condition scores
for cond, score in sorted(overall_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{cond}: {score:.6f}")

# For plotting: average across conditions
avg_scores = np.mean(list(condition_means.values()), axis=0)  # (32, 50)
print(f"Average scores shape: {avg_scores.shape}")
print(f"Max score: {np.max(avg_scores):.6f}")
print(f"Min score: {np.min(avg_scores):.6f}")

# Create heatmap for conditions vs layers (excluding layer 0)
import pandas as pd

# Create dataframe: rows=layers (1-31), columns=conditions
layers = [f'Layer_{i}' for i in range(1, 32)]  # Exclude layer 0
conditions = list(condition_means.keys())

# Average across token positions for each condition and layer
heatmap_data = {}
for condition in conditions:
    scores = condition_means[condition]
    # Average across tokens (axis=1), then take layers 1-31
    layer_avgs = scores.mean(axis=1)[1:]  # Exclude layer 0
    heatmap_data[condition] = layer_avgs

df_heatmap = pd.DataFrame(heatmap_data, index=layers)
print("Heatmap data shape:", df_heatmap.shape)
print("Max value:", df_heatmap.max().max())
print("Min value:", df_heatmap.min().min())

# Find strongest condition-layer combination
max_val = df_heatmap.max().max()
max_loc = df_heatmap.stack().idxmax()
print(f"Strongest effect: {max_loc} = {max_val:.6f}")

# Save to CSV
df_heatmap.to_csv('condition_layer_heatmap.csv')
print("Saved condition vs layer heatmap to condition_layer_heatmap.csv")

# Try to create plot
try:
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Heatmap(
        z=df_heatmap.values,
        x=df_heatmap.columns,
        y=df_heatmap.index,
        colorscale='RdBu_r',
        zmid=0
    ))
    fig.update_layout(
        title='Rewrite Scores: Layers vs Medical Conditions (Excluding Layer 0)',
        xaxis_title='Medical Condition',
        yaxis_title='Layer',
        width=800,
        height=1000
    )
    fig.write_html('condition_layer_heatmap.html')
    print("Interactive heatmap saved to condition_layer_heatmap.html")
except ImportError:
    print("Plotly not available, data saved to CSV")