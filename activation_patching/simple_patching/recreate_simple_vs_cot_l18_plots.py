#!/usr/bin/env python3
"""
Recreate the two Layer-18 "Simple vs CoT" comparison figures.

Data sources:
- Simple run: female5_patch_male/condition_token_analysis/per_unit_layer_condition_token_summary.csv
- CoT run:    patching_results/qwen_cot20_mlp_rewrite/layer18_condition_token_first_last.csv

Outputs (by default, written into the CoT run directory):
- simple_vs_cot_l18_comparison.csv
- simple_vs_cot_l18_overall.png
- simple_vs_cot_l18_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Divergent palette requested by user.
PALETTE = {
    "green_dark": "#488f31",
    "green_mid": "#6aaa96",
    "green_light": "#aecdc2",
    "neutral": "#f1f1f1",
    "orange_light": "#f8b9a1",
    "orange_mid": "#f08056",
    "orange_dark": "#de3e00",
}

COHORT_ORDER = [
    "asthma",
    "depression",
    "multiple_sclerosis",
    "rheumatoid_arthritis",
    "sarcoidosis",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_simple_dir = script_dir / "female5_patch_male"
    default_cot_dir = script_dir / "patching_results" / "qwen_cot20_mlp_rewrite"

    p = argparse.ArgumentParser(description="Recreate simple-vs-CoT Layer-18 plots.")
    p.add_argument("--simple-run-dir", type=Path, default=default_simple_dir)
    p.add_argument("--cot-run-dir", type=Path, default=default_cot_dir)
    p.add_argument("--layer", type=int, default=18, help="Layer index to aggregate (default: 18).")
    p.add_argument("--output-dir", type=Path, default=None, help="Defaults to --cot-run-dir.")
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def _normalize_cohort_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def load_simple_metrics(simple_run_dir: Path, layer: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_unit_csv = simple_run_dir / "condition_token_analysis" / "per_unit_layer_condition_token_summary.csv"
    if not per_unit_csv.is_file():
        raise FileNotFoundError(f"Simple per-unit summary not found: {per_unit_csv}")

    df = pd.read_csv(per_unit_csv)
    needed = {"cohort", "score_key", "layer", "condition_token_mean", "full_token_mean"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Simple CSV missing required columns: {missing}")

    df["cohort"] = df["cohort"].map(_normalize_cohort_name)
    df = df[(df["score_key"] == "rewrite_scores") & (df["layer"] == layer)].copy()
    if df.empty:
        raise ValueError(f"No Simple rows for score_key=rewrite_scores at layer={layer}")

    grouped = (
        df.groupby("cohort", as_index=False)
        .agg(
            simple_L18_condition_token=("condition_token_mean", "mean"),
            simple_L18_full_token_mean=("full_token_mean", "mean"),
            simple_n_units=("cohort", "size"),
        )
        .rename(columns={"cohort": "condition"})
    )
    return grouped, df


def load_cot_metrics(cot_run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cot_csv = cot_run_dir / "layer18_condition_token_first_last.csv"
    if not cot_csv.is_file():
        raise FileNotFoundError(f"CoT first/last summary not found: {cot_csv}")

    df = pd.read_csv(cot_csv)
    needed = {"condition", "l18_first", "l18_last", "l18_full_mean"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"CoT CSV missing required columns: {missing}")

    df["condition"] = df["condition"].map(_normalize_cohort_name)
    grouped = (
        df.groupby("condition", as_index=False)
        .agg(
            cot_L18_first_condition=("l18_first", "mean"),
            cot_L18_last_condition=("l18_last", "mean"),
            cot_L18_full_token_mean=("l18_full_mean", "mean"),
            cot_n_units=("condition", "size"),
        )
    )
    return grouped, df


def build_comparison_df(
    simple_grouped: pd.DataFrame,
    simple_raw_l18: pd.DataFrame,
    cot_grouped: pd.DataFrame,
    cot_raw: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.merge(simple_grouped, cot_grouped, on="condition", how="inner")
    if merged.empty:
        raise ValueError("No overlapping conditions between Simple and CoT sources.")

    merged = merged[
        [
            "condition",
            "simple_L18_condition_token",
            "cot_L18_first_condition",
            "cot_L18_last_condition",
            "cot_L18_full_token_mean",
            "simple_L18_full_token_mean",
            "simple_n_units",
            "cot_n_units",
        ]
    ]

    # Keep canonical ordering where available; append unknown conditions at end.
    order_map = {c: i for i, c in enumerate(COHORT_ORDER)}
    merged = merged.sort_values(
        by="condition",
        key=lambda s: s.map(lambda x: order_map.get(x, 10_000)),
    ).reset_index(drop=True)

    # Keep "ALL" consistent with prior analysis: aggregate directly from raw units.
    # This preserves behavior when some condition-token entries are NaN in one cohort.
    overall = {
        "condition": "ALL",
        "simple_L18_condition_token": float(simple_raw_l18["condition_token_mean"].mean()),
        "cot_L18_first_condition": float(cot_raw["l18_first"].mean()),
        "cot_L18_last_condition": float(cot_raw["l18_last"].mean()),
        "cot_L18_full_token_mean": float(cot_raw["l18_full_mean"].mean()),
        "simple_L18_full_token_mean": float(simple_raw_l18["full_token_mean"].mean()),
        "simple_n_units": int(len(simple_raw_l18)),
        "cot_n_units": int(len(cot_raw)),
    }

    return pd.concat([merged, pd.DataFrame([overall])], ignore_index=True)


def plot_overall(comparison_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    row = comparison_df[comparison_df["condition"] == "ALL"].iloc[0]
    labels = ["Simple\ncond-token", "CoT\nfirst cond", "CoT\nlast cond", "CoT\nall-token mean"]
    values = [
        float(row["simple_L18_condition_token"]),
        float(row["cot_L18_first_condition"]),
        float(row["cot_L18_last_condition"]),
        float(row["cot_L18_full_token_mean"]),
    ]
    colors = [
        PALETTE["green_dark"],
        PALETTE["orange_dark"],
        PALETTE["orange_mid"],
        PALETTE["green_mid"],
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Overall average: Simple vs CoT (Layer 18)", fontsize=14)
    ax.set_ylabel("Layer-18 MLP rewrite score", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.018, f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _display_name(condition: str) -> str:
    return condition.replace("_", " ")


def plot_by_condition(comparison_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    df = comparison_df[comparison_df["condition"] != "ALL"].copy()
    df["label"] = df["condition"].map(_display_name)

    x = np.arange(len(df))
    width = 0.20

    series: List[Dict[str, object]] = [
        {
            "name": "Simple · cond-token",
            "values": df["simple_L18_condition_token"].to_numpy(dtype=float),
            "color": PALETTE["green_dark"],
            "offset": -1.5 * width,
        },
        {
            "name": "CoT · first cond-token",
            "values": df["cot_L18_first_condition"].to_numpy(dtype=float),
            "color": PALETTE["orange_dark"],
            "offset": -0.5 * width,
        },
        {
            "name": "CoT · last cond-token",
            "values": df["cot_L18_last_condition"].to_numpy(dtype=float),
            "color": PALETTE["orange_mid"],
            "offset": 0.5 * width,
        },
        {
            "name": "CoT · mean over all tokens",
            "values": df["cot_L18_full_token_mean"].to_numpy(dtype=float),
            "color": PALETTE["green_mid"],
            "offset": 1.5 * width,
        },
    ]

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for s in series:
        ax.bar(
            x + float(s["offset"]),
            s["values"],
            width,
            label=str(s["name"]),
            color=str(s["color"]),
            edgecolor="#111111",
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist(), rotation=14)
    ax.set_ylabel("Layer-18 MLP rewrite score")
    ax.set_title("Simple vs CoT — Layer-18 MLP rewrite scores by condition")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.cot_run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    simple_grouped, simple_raw_l18 = load_simple_metrics(args.simple_run_dir, args.layer)
    cot_grouped, cot_raw = load_cot_metrics(args.cot_run_dir)
    comparison_df = build_comparison_df(simple_grouped, simple_raw_l18, cot_grouped, cot_raw)

    out_csv = output_dir / "simple_vs_cot_l18_comparison.csv"
    out_overall_png = output_dir / "simple_vs_cot_l18_overall.png"
    out_condition_png = output_dir / "simple_vs_cot_l18_comparison.png"

    comparison_df[
        [
            "condition",
            "simple_L18_condition_token",
            "cot_L18_first_condition",
            "cot_L18_last_condition",
            "cot_L18_full_token_mean",
            "simple_L18_full_token_mean",
        ]
    ].to_csv(out_csv, index=False)

    plot_overall(comparison_df, out_overall_png, dpi=args.dpi)
    plot_by_condition(comparison_df, out_condition_png, dpi=args.dpi)

    print(f"Wrote comparison CSV: {out_csv}")
    print(f"Wrote overall plot:    {out_overall_png}")
    print(f"Wrote condition plot:  {out_condition_png}")


if __name__ == "__main__":
    main()
