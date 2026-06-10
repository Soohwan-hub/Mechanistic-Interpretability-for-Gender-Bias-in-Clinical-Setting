#!/usr/bin/env python3
"""Summarize qwen_simple31_residual_rewrite: heatmaps + layer rankings."""
from __future__ import annotations

import argparse
import csv
import glob
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

CHAT_PAT = re.compile(r"im_start|im_end|assistant|user|system|redacted|^\.|^_$", re.I)
DEFAULT_RUN_DIR = Path("patching_results/qwen_simple31_residual_rewrite")


def is_template_token(label: str) -> bool:
    tok = label.rsplit("_", 1)[0]
    return bool(CHAT_PAT.search(tok)) or tok.strip() in ("", ".")


def load_artifacts(run_dir: Path) -> List[dict]:
    files = sorted((run_dir / "artifacts").glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No artifacts in {run_dir / 'artifacts'}")
    out = []
    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        data["_path"] = path
        out.append(data)
    return out


def aggregate_heatmap_matrix(cells: List[dict]) -> Tuple[np.ndarray, List[str], List[int]]:
    max_tokens = max(c["rewrite_scores"].shape[1] for c in cells)
    n_layers = cells[0]["rewrite_scores"].shape[0]
    stack = np.full((len(cells), n_layers, max_tokens), np.nan, dtype=float)
    ref_labels = cells[0]["token_labels"]
    for i, c in enumerate(cells):
        s = c["rewrite_scores"]
        stack[i, :, : s.shape[1]] = s
        if len(c["token_labels"]) > len(ref_labels):
            ref_labels = c["token_labels"]
    mean_mat = np.nanmean(stack, axis=0)
    layers = cells[0]["layer_labels"]
    token_labels = [ref_labels[t] if t < len(ref_labels) else f"tok_{t}" for t in range(max_tokens)]
    return mean_mat, token_labels, layers


def per_layer_ranking(cells: List[dict], layer_start: int, layer_end: int) -> List[dict]:
    """Rank layers by mean rewrite over content tokens, averaged across all cells."""
    layer_scores: Dict[int, List[float]] = {}
    for c in cells:
        s = c["rewrite_scores"]
        layers = c["layer_labels"]
        tokens = c["token_labels"]
        for li, layer in enumerate(layers):
            if layer < layer_start or layer > layer_end:
                continue
            row = s[li]
            content = [row[t] for t in range(len(row)) if not is_template_token(tokens[t])]
            if content:
                layer_scores.setdefault(layer, []).append(float(np.nanmean(content)))
    rows = []
    for layer in sorted(layer_scores):
        vals = layer_scores[layer]
        rows.append(
            {
                "layer": layer,
                "mean_rewrite": float(np.mean(vals)),
                "median_rewrite": float(np.median(vals)),
                "std_rewrite": float(np.std(vals)),
                "n_cells": len(vals),
            }
        )
    rows.sort(key=lambda r: -r["mean_rewrite"])
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def per_prompt_layer_ranking(cells: List[dict], layer_start: int, layer_end: int) -> List[dict]:
    """Best layer per prompt_id (mean content-token rewrite, pooled over cohorts)."""
    by_prompt: Dict[int, Dict[int, List[float]]] = {}
    for c in cells:
        pid = c["metadata"]["prompt_id"]
        s = c["rewrite_scores"]
        layers = c["layer_labels"]
        tokens = c["token_labels"]
        for li, layer in enumerate(layers):
            if layer < layer_start or layer > layer_end:
                continue
            row = s[li]
            content = [row[t] for t in range(len(row)) if not is_template_token(tokens[t])]
            if content:
                by_prompt.setdefault(pid, {}).setdefault(layer, []).append(float(np.nanmean(content)))
    rows = []
    for pid in sorted(by_prompt):
        layer_means = {L: float(np.mean(v)) for L, v in by_prompt[pid].items()}
        best_layer = max(layer_means, key=layer_means.get)
        rows.append(
            {
                "prompt_id": pid,
                "best_layer": best_layer,
                "best_layer_mean_rewrite": layer_means[best_layer],
                "layer18_mean_rewrite": layer_means.get(18, float("nan")),
            }
        )
    return rows


def layer_frequency_across_prompts(prompt_rows: List[dict], top_n: int = 10) -> List[dict]:
    from collections import Counter

    counts = Counter(r["best_layer"] for r in prompt_rows)
    total = len(prompt_rows)
    ranked = counts.most_common()
    return [
        {
            "layer": layer,
            "count_prompts": cnt,
            "fraction_prompts": cnt / total,
            "rank": i + 1,
        }
        for i, (layer, cnt) in enumerate(ranked[:top_n])
    ]


def save_heatmap(
    matrix: np.ndarray,
    token_labels: List[str],
    layers: List[int],
    out_base: Path,
    layer_start: int,
    title: str,
) -> None:
    layer_mask = [i for i, L in enumerate(layers) if L >= layer_start]
    mat = matrix[layer_mask, :]
    layer_names = [f"L{layers[i]}" for i in layer_mask]

    # Short x labels for readability
    short_x = []
    for i, lab in enumerate(token_labels[: mat.shape[1]]):
        tok = lab.rsplit("_", 1)[0]
        short_x.append(f"{i}:{tok[:12]}")

    csv_path = out_base.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + short_x)
        for name, row in zip(layer_names, mat):
            w.writerow([name] + [f"{v:.6f}" if np.isfinite(v) else "" for v in row])

    fig, ax = plt.subplots(figsize=(16, 10))
    vmax = np.nanpercentile(mat, 99) if np.any(np.isfinite(mat)) else 1.0
    vmin = np.nanpercentile(mat, 1) if np.any(np.isfinite(mat)) else -0.1
    im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(short_x, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.set_xlabel("Token position (index: decoded subtoken)")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Mean rewrite score")
    plt.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_base.with_suffix('.pdf')}")


def save_layer_rank_plots(layer_rows: List[dict], freq_rows: List[dict], out_dir: Path) -> None:
    # Full layer ranking bar chart
    layers = [r["layer"] for r in sorted(layer_rows, key=lambda x: x["layer"])]
    means = [r["mean_rewrite"] for r in sorted(layer_rows, key=lambda x: x["layer"])]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(layers, means, color="steelblue")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean rewrite (content tokens, all cells)")
    ax.set_title("Residual rewrite by layer (155 cells, L5–21 content tokens)")
    ax.axvline(18, color="crimson", linestyle="--", linewidth=1, label="L18 (paper)")
    ax.legend()
    plt.tight_layout()
    p = out_dir / "layer_mean_rewrite_bar"
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Top layers largest to smallest (ranked list)
    top = layer_rows[:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [f"L{r['layer']}" for r in reversed(top)],
        [r["mean_rewrite"] for r in reversed(top)],
        color="darkorange",
    )
    ax.set_xlabel("Mean rewrite score")
    ax.set_title("Top layers (L5–21, content tokens, pooled over 155 cells)")
    plt.tight_layout()
    p = out_dir / "top_layers_ranked"
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # How often each layer is #1 per prompt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [f"L{r['layer']}" for r in freq_rows],
        [r["count_prompts"] for r in freq_rows],
        color="seagreen",
    )
    ax.set_xlabel("Layer (best layer for prompt)")
    ax.set_ylabel("# prompts (out of 31)")
    ax.set_title("Most common best layer across prompt templates")
    plt.tight_layout()
    p = out_dir / "top_layers_across_prompts"
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote layer ranking figures to {out_dir}")


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--layer-start-heatmap", type=int, default=1)
    p.add_argument("--layer-start-rank", type=int, default=5)
    p.add_argument("--layer-end-rank", type=int, default=21)
    args = p.parse_args()

    run_dir = args.run_dir
    out_dir = run_dir / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_artifacts(run_dir)
    mat, token_labels, layers = aggregate_heatmap_matrix(cells)

    save_heatmap(
        mat,
        token_labels,
        layers,
        out_dir / "fig_token_layer_heatmap_all155",
        layer_start=args.layer_start_heatmap,
        title="Qwen simple prompts — residual rewrite (mean over 155 cells)",
    )

    layer_rows = per_layer_ranking(cells, args.layer_start_rank, args.layer_end_rank)
    prompt_rows = per_prompt_layer_ranking(cells, args.layer_start_rank, args.layer_end_rank)
    freq_rows = layer_frequency_across_prompts(prompt_rows, top_n=12)

    write_csv(
        out_dir / "top_layers_pooled.csv",
        layer_rows,
        ["rank", "layer", "mean_rewrite", "median_rewrite", "std_rewrite", "n_cells"],
    )
    write_csv(
        out_dir / "best_layer_per_prompt.csv",
        prompt_rows,
        ["prompt_id", "best_layer", "best_layer_mean_rewrite", "layer18_mean_rewrite"],
    )
    write_csv(
        out_dir / "top_layers_across_prompts.csv",
        freq_rows,
        ["rank", "layer", "count_prompts", "fraction_prompts"],
    )
    save_layer_rank_plots(layer_rows, freq_rows, out_dir)

    # Console summary
    top3 = layer_rows[:3]
    print("\n=== RESIDUAL RUN SUMMARY (155 cells) ===")
    print(f"Top layers L{args.layer_start_rank}-{args.layer_end_rank} (content-token mean):")
    for r in top3:
        print(f"  #{r['rank']} L{r['layer']}: mean={r['mean_rewrite']:.3f}")
    print("Most common best layer per prompt:")
    for r in freq_rows[:5]:
        print(f"  L{r['layer']}: {r['count_prompts']}/31 prompts")


if __name__ == "__main__":
    main()
