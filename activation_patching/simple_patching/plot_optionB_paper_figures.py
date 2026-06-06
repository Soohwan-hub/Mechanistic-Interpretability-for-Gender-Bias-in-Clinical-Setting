#!/usr/bin/env python3
"""Paper figures for Option B interchange run."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_RUN = Path("vignette_results/qwen_optionB_interchange_free_canonical_layer18_n35")
COHORTS = [
    "asthma",
    "depression",
    "multiple_sclerosis",
    "rheumatoid_arthritis",
    "sarcoidosis",
]
FACTORS = [0, 1, 2, 3, 4, 5]
TOP_N = 10
IA_THRESHOLD_PCT = 50.0


def _load_cohort_prompt_rows(summary_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(summary_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("scope") != "cohort_prompt":
                continue
            rows.append(
                {
                    "cohort": row["cohort"],
                    "prompt_id": int(row["prompt_id"]),
                    "factor": float(row["factor"]),
                    "n": int(row["n"]),
                    "male_n": int(row["male_n"]),
                    "female_n": int(row["female_n"]),
                    "unknown_n": int(row["unknown_n"]),
                    "ia": float(row["target_success_rate"]),
                }
            )
    return rows


def _ia_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, int], dict[float, float]]:
    cp: dict[tuple[str, int], dict[float, float]] = defaultdict(dict)
    for row in rows:
        cp[(str(row["cohort"]), int(row["prompt_id"]))][float(row["factor"])] = float(row["ia"])
    return cp


def _row_at(
    rows: list[dict[str, object]], cohort: str, prompt_id: int, factor: float
) -> dict[str, object]:
    for row in rows:
        if row["cohort"] == cohort and row["prompt_id"] == prompt_id and row["factor"] == factor:
            return row
    raise KeyError(f"Missing row: {cohort} p{prompt_id} f{factor}")


def _best_factor_row(prompt_rows: list[dict[str, object]]) -> dict[str, object]:
    return max(prompt_rows, key=lambda row: float(row["ia"]))


def _export_top_prompts(rows: list[dict[str, object]], out_dir: Path) -> pd.DataFrame:
    top_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        f5_rows = sorted(
            [row for row in rows if row["cohort"] == cohort and row["factor"] == 5.0],
            key=lambda row: (-float(row["ia"]), int(row["prompt_id"])),
        )
        for rank, row in enumerate(f5_rows[:TOP_N], start=1):
            f0 = _row_at(rows, cohort, int(row["prompt_id"]), 0.0)
            best = _best_factor_row([r for r in rows if r["cohort"] == cohort and r["prompt_id"] == row["prompt_id"]])
            top_rows.append(
                {
                    "cohort": cohort,
                    "rank_at_f5": rank,
                    "prompt_id": row["prompt_id"],
                    "ia_f0_pct": round(100 * float(f0["ia"]), 1),
                    "ia_f5_pct": round(100 * float(row["ia"]), 1),
                    "ia_gain_f0_to_f5_pp": round(100 * (float(row["ia"]) - float(f0["ia"])), 1),
                    "best_factor": int(best["factor"]),
                    "ia_best_pct": round(100 * float(best["ia"]), 1),
                    "ia_gain_f0_to_best_pp": round(100 * (float(best["ia"]) - float(f0["ia"])), 1),
                    "n": row["n"],
                    "male_n_f0": f0["male_n"],
                    "male_n_f5": row["male_n"],
                    "unknown_n_f5": row["unknown_n"],
                }
            )
    df = pd.DataFrame(top_rows)
    df.to_csv(out_dir / "top_prompts_by_disease.csv", index=False)
    return df


def _export_scaling_gains(rows: list[dict[str, object]], out_dir: Path) -> pd.DataFrame:
    gain_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        for prompt_id in range(32):
            prompt_rows = [row for row in rows if row["cohort"] == cohort and row["prompt_id"] == prompt_id]
            f0 = _row_at(rows, cohort, prompt_id, 0.0)
            best = _best_factor_row(prompt_rows)
            gain_rows.append(
                {
                    "cohort": cohort,
                    "prompt_id": prompt_id,
                    "ia_f0_pct": round(100 * float(f0["ia"]), 1),
                    "best_factor": int(best["factor"]),
                    "ia_best_pct": round(100 * float(best["ia"]), 1),
                    "ia_gain_pp": round(100 * (float(best["ia"]) - float(f0["ia"])), 1),
                    "male_n_f0": f0["male_n"],
                    "male_n_best": best["male_n"],
                    "n": f0["n"],
                }
            )
    df = pd.DataFrame(gain_rows)
    df.to_csv(out_dir / "prompt_scaling_gains.csv", index=False)
    return df


def _export_cross_disease_f5(rows: list[dict[str, object]], out_dir: Path) -> pd.DataFrame:
    cross_rows: list[dict[str, object]] = []
    for prompt_id in range(32):
        by_cohort = {
            str(row["cohort"]): row
            for row in rows
            if row["prompt_id"] == prompt_id and row["factor"] == 5.0
        }
        if len(by_cohort) != len(COHORTS):
            continue
        ia_values = [100 * float(by_cohort[cohort]["ia"]) for cohort in COHORTS]
        mean_ia = sum(ia_values) / len(ia_values)
        n_ge50 = sum(1 for ia in ia_values if ia >= IA_THRESHOLD_PCT)
        cross_rows.append(
            {
                "prompt_id": prompt_id,
                "mean_ia_f5_pct": round(mean_ia, 1),
                "n_diseases_ge50pct": n_ge50,
                **{f"ia_f5_{cohort}_pct": round(100 * float(by_cohort[cohort]["ia"]), 1) for cohort in COHORTS},
            }
        )
    df = pd.DataFrame(cross_rows).sort_values("mean_ia_f5_pct", ascending=False)
    df.to_csv(out_dir / "cross_disease_prompts_f5.csv", index=False)
    return df


def _export_cohort_factor_summary(rows: list[dict[str, object]], out_dir: Path) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for cohort in COHORTS:
        for factor in FACTORS:
            factor_rows = [row for row in rows if row["cohort"] == cohort and row["factor"] == float(factor)]
            ia_values = [float(row["ia"]) for row in factor_rows]
            summary_rows.append(
                {
                    "cohort": cohort,
                    "factor": factor,
                    "mean_ia_pct": round(100 * sum(ia_values) / len(ia_values), 1),
                    "prompts_ge50pct": sum(1 for ia in ia_values if ia >= IA_THRESHOLD_PCT / 100),
                    "n_prompts": len(factor_rows),
                }
            )
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / "cohort_factor_summary.csv", index=False)
    return df


def _print_analysis_summary(
    rows: list[dict[str, object]],
    run_dir: Path,
    out_dir: Path,
    top_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    cohort_factor_df: pd.DataFrame,
) -> None:
    n_per_cell = int(rows[0]["n"])
    n_cells = len(COHORTS) * 32 * len(FACTORS)
    total_gens = n_cells * n_per_cell

    print("\n=== GENERATION COUNTS ===")
    print(f"Run: {run_dir}")
    print(f"Total generations: {total_gens:,} ({len(COHORTS)} diseases × 32 prompts × {len(FACTORS)} factors × {n_per_cell}/cell)")

    print("\n=== TOP PROMPTS @ f5 (by disease) ===")
    for cohort in COHORTS:
        subset = top_df[top_df["cohort"] == cohort].head(5)
        print(f"\n{cohort.replace('_', ' ').title()}:")
        for _, row in subset.iterrows():
            print(
                f"  p{int(row['prompt_id']):2d}: {row['ia_f5_pct']:5.1f}% @f5  "
                f"(f0→f5 {row['ia_gain_f0_to_f5_pp']:+5.1f}pp, male {int(row['male_n_f0'])}→{int(row['male_n_f5'])}/{int(row['n'])})"
            )

    print(f"\n=== PROMPTS WITH IA ≥ {IA_THRESHOLD_PCT:.0f}% @ f5 ===")
    for cohort in COHORTS:
        hits = cross_df[cross_df[f"ia_f5_{cohort}_pct"] >= IA_THRESHOLD_PCT].sort_values(
            f"ia_f5_{cohort}_pct", ascending=False
        )
        if hits.empty:
            print(f"  {cohort}: none")
            continue
        parts = ", ".join(
            f"p{int(row['prompt_id'])} ({row[f'ia_f5_{cohort}_pct']:.0f}%)"
            for _, row in hits.iterrows()
        )
        print(f"  {cohort}: {len(hits)} prompts — {parts}")

    print("\n=== CROSS-DISEASE PROMPTS @ f5 (mean IA ≥ 40%) ===")
    strong = cross_df[cross_df["mean_ia_f5_pct"] >= 40.0]
    for _, row in strong.iterrows():
        parts = ", ".join(f"{cohort[:3]}={row[f'ia_f5_{cohort}_pct']:.0f}%" for cohort in COHORTS)
        print(f"  p{int(row['prompt_id']):2d}: mean={row['mean_ia_f5_pct']:.1f}%  [{parts}]")

    print("\n=== CANONICAL p0 SCALING ===")
    for cohort in COHORTS:
        p0_pts = [
            (int(f), round(100 * float(_row_at(rows, cohort, 0, float(f))["ia"]), 0))
            for f in FACTORS
        ]
        print(f"  {cohort:22s} " + " → ".join(f"f{f}={ia:.0f}%" for f, ia in p0_pts))

    print(f"\nWrote CSV tables to {out_dir}:")
    for name in (
        "top_prompts_by_disease.csv",
        "prompt_scaling_gains.csv",
        "cross_disease_prompts_f5.csv",
        "cohort_factor_summary.csv",
        "table_all_ia.csv",
    ):
        print(f"  - {out_dir / name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper figures for Option B IA run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN,
        help="Run directory containing summary_by_factor.tsv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to <run-dir>/paper_figures).",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Only export CSV tables and print summary (skip PDF/PNG generation).",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip printing analysis tables to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir
    out_dir = args.out_dir or (run_dir / "paper_figures")
    summary_path = run_dir / "summary_by_factor.tsv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_cohort_prompt_rows(summary_path)
    cp = _ia_lookup(rows)

    top_df = _export_top_prompts(rows, out_dir)
    _export_scaling_gains(rows, out_dir)
    cross_df = _export_cross_disease_f5(rows, out_dir)
    cohort_factor_df = _export_cohort_factor_summary(rows, out_dir)

    all_ia_rows = []
    for cohort in COHORTS:
        for prompt_id in range(32):
            for factor in FACTORS:
                row = _row_at(rows, cohort, prompt_id, float(factor))
                all_ia_rows.append(
                    {
                        "cohort": cohort,
                        "prompt_id": prompt_id,
                        "factor": factor,
                        "ia_pct": round(100 * float(row["ia"]), 1),
                        "male_n": row["male_n"],
                        "n": row["n"],
                    }
                )
    pd.DataFrame(all_ia_rows).to_csv(out_dir / "table_all_ia.csv", index=False)

    if not args.skip_figures:
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

        fig, ax = plt.subplots(figsize=(6, 4))
        for cohort in COHORTS:
            ys = [100 * cp[(cohort, 0)][float(factor)] for factor in FACTORS]
            ax.plot(FACTORS, ys, marker="o", label=cohort.replace("_", " "))
        ax.set_xlabel("Patch scale factor")
        ax.set_ylabel("Interchange accuracy (%)")
        ax.set_title("Canonical prompt (p0): IA vs scaling factor")
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(out_dir / "fig1_ia_vs_factor_prompt0.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig1_ia_vs_factor_prompt0.png", dpi=300, bbox_inches="tight")
        plt.close()

        for cohort in COHORTS:
            mat = []
            for prompt_id in range(32):
                mat.append(
                    [100 * cp[(cohort, prompt_id)].get(float(factor), float("nan")) for factor in FACTORS]
                )
            heatmap_df = pd.DataFrame(
                mat,
                index=[f"p{prompt_id}" for prompt_id in range(32)],
                columns=[f"f{factor}" for factor in FACTORS],
            )
            fig, ax = plt.subplots(figsize=(7, 10))
            sns.heatmap(
                heatmap_df, ax=ax, cmap="viridis", vmin=0, vmax=100, cbar_kws={"label": "IA (%)"}
            )
            ax.set_title(cohort.replace("_", " "))
            ax.set_xlabel("Factor")
            ax.set_ylabel("Prompt")
            fig.tight_layout()
            fig.savefig(out_dir / f"fig2_heatmap_{cohort}.pdf", bbox_inches="tight")
            fig.savefig(out_dir / f"fig2_heatmap_{cohort}.png", dpi=300, bbox_inches="tight")
            plt.close()

        mat = [[100 * cp[(cohort, 0)][float(factor)] for factor in FACTORS] for cohort in COHORTS]
        heatmap_df = pd.DataFrame(
            mat,
            index=[cohort.replace("_", " ") for cohort in COHORTS],
            columns=[f"f{factor}" for factor in FACTORS],
        )
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap="viridis", vmin=0, vmax=100, ax=ax)
        ax.set_title("Prompt 0: IA by disease and factor")
        fig.tight_layout()
        fig.savefig(out_dir / "fig3_cohort_x_factor_prompt0.pdf", bbox_inches="tight")
        fig.savefig(out_dir / "fig3_cohort_x_factor_prompt0.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Wrote figures to {out_dir}")

    if not args.skip_summary:
        _print_analysis_summary(rows, run_dir, out_dir, top_df, cross_df, cohort_factor_df)


if __name__ == "__main__":
    main()
