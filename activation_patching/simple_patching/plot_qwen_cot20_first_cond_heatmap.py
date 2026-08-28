#!/usr/bin/env python3
"""
Generate a heatmap for Qwen CoT-20 MLP using first condition-token rewrite scores.

Input:
- patching_results/qwen_cot20_mlp_rewrite/layer18_condition_token_first_last.csv

Output:
- patching_results/qwen_cot20_mlp_rewrite/qwen_cot20_mlp_rewrite_l18_first_cond_heatmap.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


PALETTE = [
    "#488f31",  # green dark
    "#6aaa96",
    "#aecdc2",
    "#f1f1f1",
    "#f8b9a1",
    "#f08056",
    "#de3e00",  # orange dark
]

COHORT_ORDER = [
    "asthma",
    "depression",
    "multiple_sclerosis",
    "rheumatoid_arthritis",
    "sarcoidosis",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_run_dir = script_dir / "patching_results" / "qwen_cot20_mlp_rewrite"

    p = argparse.ArgumentParser(description="Plot Layer-18 first condition-token heatmap for qwen_cot20_mlp_rewrite.")
    p.add_argument("--run-dir", type=Path, default=default_run_dir)
    p.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/layer18_condition_token_first_last.csv",
    )
    p.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/qwen_cot20_mlp_rewrite_l18_first_cond_heatmap.png",
    )
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--annotate", action="store_true", help="Annotate each cell with score value.")
    return p.parse_args()


def _clean_condition_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _display_condition(name: str) -> str:
    return name.replace("_", " ")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    input_csv = args.input_csv if args.input_csv is not None else run_dir / "layer18_condition_token_first_last.csv"
    output_png = (
        args.output_png
        if args.output_png is not None
        else run_dir / "qwen_cot20_mlp_rewrite_l18_first_cond_heatmap.png"
    )

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    needed = {"condition", "prompt_id", "l18_first"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    df = df.copy()
    df["condition"] = df["condition"].map(_clean_condition_name)
    df["prompt_id"] = df["prompt_id"].astype(int)
    df["l18_first"] = pd.to_numeric(df["l18_first"], errors="coerce")

    pivot = df.pivot_table(index="condition", columns="prompt_id", values="l18_first", aggfunc="mean")
    # Keep canonical order and prompts 1..20 where possible.
    index_order = [c for c in COHORT_ORDER if c in pivot.index] + [c for c in pivot.index if c not in COHORT_ORDER]
    col_order = sorted(pivot.columns.tolist())
    pivot = pivot.reindex(index=index_order, columns=col_order)

    matrix = pivot.to_numpy(dtype=float)
    if matrix.size == 0:
        raise ValueError("No data available to plot.")

    cmap = LinearSegmentedColormap.from_list("green_orange_div", PALETTE, N=256)
    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    if np.isclose(vmin, vmax):
        vmin = vmin - 1e-6
        vmax = vmax + 1e-6

    fig, ax = plt.subplots(figsize=(13, 4.2))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns.tolist()], fontsize=9)
    ax.set_xlabel("Prompt ID", fontsize=11)

    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([_display_condition(c) for c in pivot.index.tolist()], fontsize=10)
    ax.set_ylabel("Condition", fontsize=11)

    ax.set_title("Qwen CoT-20 MLP rewrite — Layer 18 first condition-token score", fontsize=13, pad=12)

    # Explicit cell boundaries (drawn above image) so both vertical and horizontal lines are visible.
    x_edges = np.arange(-0.5, matrix.shape[1] + 0.5, 1.0)
    y_edges = np.arange(-0.5, matrix.shape[0] + 0.5, 1.0)
    ax.vlines(x_edges, ymin=-0.5, ymax=matrix.shape[0] - 0.5, colors="#000000", linewidth=0.3, alpha=0.9, zorder=3)
    ax.hlines(y_edges, xmin=-0.5, xmax=matrix.shape[1] - 0.5, colors="#000000", linewidth=0.3, alpha=0.9, zorder=3)

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    if args.annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    text = "nan"
                else:
                    text = f"{val:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color="#111111")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote heatmap: {output_png}")


if __name__ == "__main__":
    main()
