"""
Replication-focused SAE causal testing on top shortlisted features.

This script reuses core helper logic from `sae_localization.py` but only runs
targeted replication for top-N `(layer, feature_idx)` hits.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import sae_localization as base

try:
    import plotly.express as px

    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


DEFAULT_SHORTLIST = (
    "gh200_full_artifacts_x86/causal_shortlist.csv"
)


def _parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_feature_list(value: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if not value.strip():
        return pairs
    for item in value.split(","):
        entry = item.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"Invalid feature entry '{entry}'. Use layer:feature_idx format."
            )
        layer_s, feat_s = entry.split(":", 1)
        pairs.append((int(layer_s), int(feat_s)))
    return pairs


def _parse_sae_overrides(value: str) -> Dict[int, str]:
    if not value.strip():
        return {}
    raw = json.loads(value)
    return {int(k): str(v) for k, v in raw.items()}


def replication_paths(run_dir: Path) -> Dict[str, Path]:
    art = run_dir / "artifacts"
    return {
        "run_dir": run_dir,
        "artifacts_dir": art,
        "plots_dir": art / "plots",
        "progress": run_dir / "progress.json",
        "baseline": art / "replication_baselines.json",
        "trace_index": art / "replication_trace_index.json",
        "targets": art / "replication_targets.csv",
        "activation_cache": art / "replication_activation_cache.json",
        "raw": art / "replication_raw.parquet",
        "summary": art / "replication_summary.csv",
        "pass_fail": art / "replication_pass_fail.csv",
    }


def _expected_sign_from_value(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 1


def select_targets(args: argparse.Namespace, paths: Dict[str, Path]) -> pd.DataFrame:
    manual_pairs = _parse_feature_list(args.feature_list)
    if manual_pairs:
        rows = []
        for layer, feat in manual_pairs:
            rows.append(
                {
                    "layer": int(layer),
                    "feature_idx": int(feat),
                    "token_identity": "<manual>",
                    "n_conditions": np.nan,
                    "mean_norm_effect": np.nan,
                    "expected_sign": 1,
                    "source": "manual",
                }
            )
        targets = pd.DataFrame(rows)
    else:
        shortlist = pd.read_csv(args.shortlist_csv)
        required = {"layer", "feature_idx", "n_conditions", "mean_norm_effect"}
        missing = required - set(shortlist.columns)
        if missing:
            raise ValueError(
                f"Shortlist file is missing required columns: {sorted(missing)}"
            )
        shortlist = shortlist.sort_values(
            ["n_conditions", "mean_norm_effect"], ascending=[False, False]
        )
        shortlist = shortlist.drop_duplicates(["layer", "feature_idx"], keep="first")
        targets = shortlist.head(args.top_n).copy()
        targets["expected_sign"] = (
            targets["mean_norm_effect"].astype(float).map(_expected_sign_from_value)
        )
        targets["source"] = "shortlist"

    targets["layer"] = targets["layer"].astype(int)
    targets["feature_idx"] = targets["feature_idx"].astype(int)
    targets.to_csv(paths["targets"], index=False)
    return targets


def load_target_saes(
    target_layers: List[int], device: str, overrides: Dict[int, str]
) -> Dict[int, Any]:
    saes: Dict[int, Any] = {}
    for layer in sorted(set(target_layers)):
        sae, _ = base.load_sae_for_layer(layer, device, overrides)
        saes[layer] = sae
    return saes


def generate_baselines(
    args: argparse.Namespace,
    model,
    tokenizer,
    female_id: int,
    male_id: int,
    device: str,
    paths: Dict[str, Path],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if args.resume and base.checkpoint_exists(paths["baseline"]) and base.checkpoint_exists(
        paths["trace_index"]
    ):
        baseline = base.load_json(paths["baseline"])
        index = base.load_json(paths["trace_index"])
        print(
            f"Baselines loaded from cache: total={len(baseline)} eligible={len(index['eligible_keys'])}"
        )
        return baseline, index

    temperatures = _parse_float_list(args.temperatures)
    conditions = _parse_csv_list(args.conditions)
    prompt_vars = _parse_csv_list(args.prompt_vars)
    seeds = _parse_int_list(args.seeds)

    baseline: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    combos = [
        (seed, condition, variation, temp_idx)
        for seed in seeds
        for condition in conditions
        for variation in prompt_vars
        for temp_idx in range(len(temperatures))
    ]
    for seed, condition, variation, temp_idx in tqdm(combos, desc="Replication baselines"):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        prompt_str = base.build_prompt(tokenizer, condition, variation)
        trace = base._generate_trace(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt_str=prompt_str,
            temperature=temperatures[temp_idx],
            max_new_tokens=args.max_new_tokens,
            female_id=female_id,
            male_id=male_id,
        )
        key = f"{condition}|{variation}|temp{temp_idx}|seed{seed}"
        trace["condition"] = condition
        trace["variation"] = variation
        trace["temp_idx"] = temp_idx
        trace["seed"] = seed
        trace["prompt_str"] = prompt_str
        baseline[key] = trace

        delta = float(trace["logit_diff"])
        eligible = trace["gender_pos"] != -1 and np.isfinite(delta) and delta > 0
        rows.append(
            {
                "trace_key": key,
                "condition": condition,
                "variation": variation,
                "temp_idx": temp_idx,
                "seed": seed,
                "delta_base": delta,
                "gender_pos": int(trace["gender_pos"]),
                "eligible": bool(eligible),
            }
        )

    trace_df = pd.DataFrame(rows)
    eligible_keys = trace_df.loc[trace_df["eligible"], "trace_key"].tolist()
    index = {
        "conditions": conditions,
        "prompt_vars": prompt_vars,
        "temperatures": temperatures,
        "seeds": seeds,
        "eligible_keys": eligible_keys,
    }
    base.save_json(paths["baseline"], baseline)
    base.save_json(paths["trace_index"], index)
    print(f"Baselines generated: total={len(rows)} eligible={len(eligible_keys)}")
    return baseline, index


def cache_target_feature_activations(
    args: argparse.Namespace,
    model,
    tokenizer,
    baseline: Dict[str, Any],
    eligible_keys: List[str],
    targets: pd.DataFrame,
    saes: Dict[int, Any],
    paths: Dict[str, Path],
) -> Dict[str, Any]:
    if args.resume and base.checkpoint_exists(paths["activation_cache"]):
        cache = base.load_json(paths["activation_cache"])
        print("Activation cache loaded from disk.")
        return cache

    features_by_layer: Dict[int, List[int]] = defaultdict(list)
    for row in targets.to_dict("records"):
        features_by_layer[int(row["layer"])].append(int(row["feature_idx"]))
    features_by_layer = {
        layer: sorted(set(feats)) for layer, feats in features_by_layer.items()
    }

    latent_values_pos: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    latent_values_neg: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    per_trace: Dict[str, Any] = {}

    for trace_key in tqdm(eligible_keys, desc="Caching target activations"):
        row = baseline[trace_key]
        token_ids = row["full_token_ids"]
        tokens = torch.tensor(token_ids, device=model.cfg.device).unsqueeze(0)
        trace_payload = {
            "condition": row["condition"],
            "variation": row["variation"],
            "temp_idx": row["temp_idx"],
            "seed": row["seed"],
            "prompt_len": row["prompt_len"],
            "token_ids": token_ids,
            "tokens_decoded": [
                tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids
            ],
            "gender_pos": row["gender_pos"],
            "delta_base": row["logit_diff"],
            "layers": {},
        }
        with torch.no_grad():
            for layer, feat_list in features_by_layer.items():
                hook_name = f"blocks.{layer}.hook_resid_post"
                _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
                resid = cache[hook_name]
                feats = saes[layer].encode(resid).squeeze(0).detach().float().cpu().numpy()
                layer_payload: Dict[str, List[float]] = {}
                for feat_idx in feat_list:
                    fk = feats[:, feat_idx]
                    layer_payload[str(feat_idx)] = fk.tolist()
                    pos_vals = fk[fk > 0]
                    neg_vals = fk[fk < 0]
                    if len(pos_vals) > 0:
                        latent_values_pos[(layer, feat_idx)].extend(pos_vals.tolist())
                    if len(neg_vals) > 0:
                        latent_values_neg[(layer, feat_idx)].extend(np.abs(neg_vals).tolist())
                trace_payload["layers"][str(layer)] = layer_payload
        per_trace[trace_key] = trace_payload

    thresholds: Dict[str, Dict[str, float]] = {}
    all_pairs = set(latent_values_pos.keys()) | set(latent_values_neg.keys())
    for (layer, feat_idx) in all_pairs:
        pos_arr = np.array(latent_values_pos.get((layer, feat_idx), []), dtype=float)
        neg_arr = np.array(latent_values_neg.get((layer, feat_idx), []), dtype=float)
        thresholds[f"{layer}:{feat_idx}"] = {
            "positive": float(pos_arr.mean() + pos_arr.std()) if len(pos_arr) else 0.0,
            "negative": float(neg_arr.mean() + neg_arr.std()) if len(neg_arr) else 0.0,
        }

    activation_cache = {
        "thresholds": thresholds,
        "per_trace": per_trace,
    }
    base.save_json(paths["activation_cache"], activation_cache)
    return activation_cache


def run_replication_ablation(
    baseline: Dict[str, Any],
    eligible_keys: List[str],
    targets: pd.DataFrame,
    activation_cache: Dict[str, Any],
    tokenizer,
    female_id: int,
    male_id: int,
    model,
    saes: Dict[int, Any],
    gating_mode: str,
) -> pd.DataFrame:
    per_trace = activation_cache["per_trace"]
    thresholds = activation_cache["thresholds"]

    rows: List[Dict[str, Any]] = []
    target_rows = targets.to_dict("records")

    for target in tqdm(target_rows, desc="Replicating targets"):
        layer = int(target["layer"])
        feat_idx = int(target["feature_idx"])
        expected_sign = int(target.get("expected_sign", 1))
        threshold_map = thresholds.get(
            f"{layer}:{feat_idx}",
            {"positive": 0.0, "negative": 0.0},
        )

        for trace_key in eligible_keys:
            trace = baseline[trace_key]
            cached = per_trace[trace_key]
            gender_pos = int(trace["gender_pos"])
            if gender_pos <= 0:
                continue

            fk_list = (
                cached.get("layers", {})
                .get(str(layer), {})
                .get(str(feat_idx), [])
            )
            if not fk_list:
                continue

            tokens = torch.tensor(trace["full_token_ids"], device=model.cfg.device).unsqueeze(0)
            delta_base = float(trace["logit_diff"])
            if not np.isfinite(delta_base):
                continue

            for token_pos, f_value in enumerate(fk_list):
                f_value = float(f_value)
                gate_signs = (
                    ["positive", "negative"]
                    if gating_mode == "sign_aware"
                    else ["positive"]
                )
                for gate_sign in gate_signs:
                    threshold = float(threshold_map.get(gate_sign, 0.0))
                    if gate_sign == "positive" and f_value <= threshold:
                        continue
                    if gate_sign == "negative" and f_value >= -threshold:
                        continue

                    logits = base.run_single_ablation(
                        model,
                        saes[layer],
                        tokens,
                        layer=layer,
                        token_pos=token_pos,
                        feature_idx=feat_idx,
                        threshold=threshold,
                        gate_sign=gate_sign,
                    )
                    dec = logits[0, gender_pos - 1, :]
                    delta_abl = float((dec[female_id] - dec[male_id]).item())
                    delta_shift = delta_base - delta_abl
                    norm_effect = float(delta_shift / (abs(delta_base) + 1e-9))
                    token_text = tokenizer.decode(
                        [trace["full_token_ids"][token_pos]], skip_special_tokens=False
                    )
                    token_identity = token_text.strip() or "<blank>"
                    sign = int(np.sign(norm_effect))
                    sign_match = int(sign == expected_sign)

                    rows.append(
                        {
                            "trace_key": trace_key,
                            "condition": trace["condition"],
                            "variation": trace["variation"],
                            "temp_idx": trace["temp_idx"],
                            "seed": trace["seed"],
                            "layer": layer,
                            "feature_idx": feat_idx,
                            "token_pos": int(token_pos),
                            "token_text": token_text,
                            "token_identity": token_identity,
                            "f_value": f_value,
                            "threshold": threshold,
                            "gate_sign": gate_sign,
                            "delta_base": delta_base,
                            "delta_abl": delta_abl,
                            "delta_shift": delta_shift,
                            "norm_effect": norm_effect,
                            "expected_sign": expected_sign,
                            "effect_sign": sign,
                            "sign_match": sign_match,
                        }
                    )

    return pd.DataFrame(rows)


def build_replication_summary(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(raw_df) == 0:
        empty_summary = pd.DataFrame(
            columns=[
                "layer",
                "feature_idx",
                "gate_sign",
                "token_identity",
                "n_hits",
                "n_runs",
                "n_conditions",
                "n_prompt_vars",
                "n_temps",
                "n_seeds",
                "mean_norm_effect",
                "median_norm_effect",
                "std_norm_effect",
                "sign_consistency",
            ]
        )
        empty_pf = pd.DataFrame(
            columns=[
                "layer",
                "feature_idx",
                "n_hits",
                "n_runs",
                "n_conditions",
                "n_prompt_vars",
                "n_temps",
                "n_seeds",
                "mean_norm_effect",
                "median_norm_effect",
                "sign_consistency",
                "gate_sign",
                "replication_pass",
            ]
        )
        return empty_summary, empty_pf

    summary = (
        raw_df.groupby(["layer", "feature_idx", "gate_sign", "token_identity"], as_index=False)
        .agg(
            n_hits=("trace_key", "count"),
            n_runs=("trace_key", "nunique"),
            n_conditions=("condition", "nunique"),
            n_prompt_vars=("variation", "nunique"),
            n_temps=("temp_idx", "nunique"),
            n_seeds=("seed", "nunique"),
            mean_norm_effect=("norm_effect", "mean"),
            median_norm_effect=("norm_effect", "median"),
            std_norm_effect=("norm_effect", "std"),
            sign_consistency=("sign_match", "mean"),
        )
        .sort_values(["n_conditions", "sign_consistency", "mean_norm_effect"], ascending=[False, False, False])
    )

    pass_fail = (
        raw_df.groupby(["layer", "feature_idx", "gate_sign"], as_index=False)
        .agg(
            n_hits=("trace_key", "count"),
            n_runs=("trace_key", "nunique"),
            n_conditions=("condition", "nunique"),
            n_prompt_vars=("variation", "nunique"),
            n_temps=("temp_idx", "nunique"),
            n_seeds=("seed", "nunique"),
            mean_norm_effect=("norm_effect", "mean"),
            median_norm_effect=("norm_effect", "median"),
            sign_consistency=("sign_match", "mean"),
        )
        .sort_values(["sign_consistency", "median_norm_effect"], ascending=[False, False])
    )
    return summary, pass_fail


def _feature_label(layer: int, feature_idx: int, gate_sign: str) -> str:
    return f"L{int(layer)}:F{int(feature_idx)}:{gate_sign[0].upper()}"


def _safe_write_fig(fig, out_path: Path) -> None:
    try:
        fig.write_image(str(out_path))
    except Exception as exc:
        print(f"Plot write warning for {out_path.name}: {exc}")


def save_replication_plots(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pass_fail_df: pd.DataFrame,
    paths: Dict[str, Path],
    args: argparse.Namespace,
) -> None:
    if not args.save_plots:
        return
    if not _HAS_PLOTLY:
        print("Plotting skipped: plotly is unavailable.")
        return
    if len(raw_df) == 0:
        print("Plotting skipped: replication_raw is empty.")
        return

    paths["plots_dir"].mkdir(parents=True, exist_ok=True)

    df = raw_df.copy()
    df["feature"] = df.apply(
        lambda r: _feature_label(r["layer"], r["feature_idx"], r["gate_sign"]), axis=1
    )

    # 1) Feature x token identity mean effect heatmap (top token identities only).
    token_order = (
        df["token_identity"].value_counts().head(args.max_plot_tokens).index.tolist()
    )
    heat_effect = (
        df[df["token_identity"].isin(token_order)]
        .groupby(["feature", "token_identity"], as_index=False)["norm_effect"]
        .mean()
    )
    if len(heat_effect) > 0:
        pivot = (
            heat_effect.pivot(index="feature", columns="token_identity", values="norm_effect")
            .fillna(0.0)
        )
        fig = px.imshow(
            pivot.values,
            x=pivot.columns,
            y=pivot.index,
            aspect="auto",
            labels={"x": "Token identity", "y": "Feature", "color": "Mean norm effect"},
            title="Replication mean norm effect (feature x token identity)",
        )
        fig.update_layout(height=550)
        _safe_write_fig(
            fig, paths["plots_dir"] / f"feature_token_mean_effect_heatmap.{args.plot_format}"
        )

    # 2) Feature x token identity sign-consistency heatmap.
    heat_sign = (
        df[df["token_identity"].isin(token_order)]
        .groupby(["feature", "token_identity"], as_index=False)["sign_match"]
        .mean()
    )
    if len(heat_sign) > 0:
        pivot = (
            heat_sign.pivot(index="feature", columns="token_identity", values="sign_match")
            .fillna(0.0)
        )
        fig = px.imshow(
            pivot.values,
            x=pivot.columns,
            y=pivot.index,
            aspect="auto",
            labels={"x": "Token identity", "y": "Feature", "color": "Sign consistency"},
            title="Replication sign consistency (feature x token identity)",
            zmin=0.0,
            zmax=1.0,
        )
        fig.update_layout(height=550)
        _safe_write_fig(
            fig, paths["plots_dir"] / f"feature_token_sign_consistency_heatmap.{args.plot_format}"
        )

    # 3) Per-feature pass/fail diagnostics: median effect + sign consistency.
    pf = pass_fail_df.copy()
    if len(pf) > 0:
        pf["feature"] = pf.apply(
            lambda r: _feature_label(r["layer"], r["feature_idx"], r["gate_sign"]),
            axis=1,
        )
        pf["replication_pass_label"] = np.where(pf["replication_pass"], "pass", "fail")
        fig = px.scatter(
            pf,
            x="median_norm_effect",
            y="sign_consistency",
            size="n_runs",
            color="replication_pass_label",
            text="feature",
            hover_data=["n_hits", "n_conditions", "n_prompt_vars", "n_temps", "n_seeds"],
            title="Replication decision space (median effect vs sign consistency)",
        )
        fig.add_vline(x=args.min_median_effect)
        fig.add_hline(y=args.min_sign_consistency)
        fig.update_traces(textposition="top center")
        fig.update_layout(height=550)
        _safe_write_fig(
            fig, paths["plots_dir"] / f"replication_decision_scatter.{args.plot_format}"
        )

    # 4) Temperature profile by feature.
    temp_profile = (
        df.groupby(["feature", "temp_idx"], as_index=False)
        .agg(
            mean_norm_effect=("norm_effect", "mean"),
            median_norm_effect=("norm_effect", "median"),
            sign_consistency=("sign_match", "mean"),
        )
        .sort_values(["feature", "temp_idx"])
    )
    if len(temp_profile) > 0:
        fig = px.line(
            temp_profile,
            x="temp_idx",
            y="mean_norm_effect",
            color="feature",
            markers=True,
            title="Mean norm effect across temperature index",
        )
        fig.update_layout(height=520)
        _safe_write_fig(fig, paths["plots_dir"] / f"feature_temperature_profile.{args.plot_format}")

        fig = px.line(
            temp_profile,
            x="temp_idx",
            y="sign_consistency",
            color="feature",
            markers=True,
            title="Sign consistency across temperature index",
        )
        fig.add_hline(y=args.min_sign_consistency)
        fig.update_layout(height=520)
        _safe_write_fig(
            fig, paths["plots_dir"] / f"feature_temperature_sign_consistency.{args.plot_format}"
        )

    # 5) Seed stability profile by feature.
    seed_profile = (
        df.groupby(["feature", "seed"], as_index=False)
        .agg(
            mean_norm_effect=("norm_effect", "mean"),
            sign_consistency=("sign_match", "mean"),
        )
        .sort_values(["feature", "seed"])
    )
    if len(seed_profile) > 0:
        fig = px.line(
            seed_profile,
            x="seed",
            y="mean_norm_effect",
            color="feature",
            markers=True,
            title="Seed stability: mean norm effect",
        )
        fig.update_layout(height=520)
        _safe_write_fig(fig, paths["plots_dir"] / f"feature_seed_effect_stability.{args.plot_format}")

        fig = px.line(
            seed_profile,
            x="seed",
            y="sign_consistency",
            color="feature",
            markers=True,
            title="Seed stability: sign consistency",
        )
        fig.add_hline(y=args.min_sign_consistency)
        fig.update_layout(height=520)
        _safe_write_fig(
            fig, paths["plots_dir"] / f"feature_seed_sign_stability.{args.plot_format}"
        )

    # 6) Rich distribution view for each feature.
    if len(df["feature"].unique()) <= max(1, int(args.max_plot_features_for_box)):
        fig = px.box(
            df,
            x="feature",
            y="norm_effect",
            color="feature",
            points="outliers",
            title="Norm effect distribution by feature",
        )
        fig.add_hline(y=args.min_median_effect)
        fig.update_layout(height=560, showlegend=False)
        _safe_write_fig(fig, paths["plots_dir"] / f"feature_norm_effect_boxplot.{args.plot_format}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replication-only SAE ablation for top shortlisted features."
    )
    parser.add_argument("--run-id", type=str, default="replication_top_hits")
    parser.add_argument("--output-dir", type=str, default="sae_replication_runs")
    parser.add_argument(
        "--shortlist-csv",
        type=str,
        default=DEFAULT_SHORTLIST,
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--feature-list",
        type=str,
        default="",
        help="Manual features as 'layer:feat,layer:feat'. Overrides shortlist.",
    )

    parser.add_argument("--conditions", type=str, default=",".join(base.CONDITIONS))
    parser.add_argument("--prompt-vars", type=str, default="var1,var2,var3")
    parser.add_argument(
        "--temperatures",
        type=str,
        default=",".join([str(t) for t in base.DEFAULT_TEMPERATURES]),
    )
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--max-new-tokens", type=int, default=700)

    parser.add_argument("--runtime-profile", type=str, default="gh200", choices=["gh200", "a10"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--min-sign-consistency", type=float, default=0.70)
    parser.add_argument("--min-median-effect", type=float, default=0.10)
    parser.add_argument("--min-runs", type=int, default=5)
    parser.add_argument(
        "--gating-mode",
        type=str,
        default="sign_aware",
        choices=["sign_aware", "positive_only"],
        help="sign_aware evaluates positive and negative gates separately.",
    )
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--max-plot-tokens", type=int, default=40)
    parser.add_argument("--max-plot-features-for-box", type=int, default=20)
    parser.add_argument(
        "--sae-id-overrides",
        type=str,
        default="",
        help='JSON string, e.g. \'{"3":"resid_post_layer_3_trainer_1"}\'',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = replication_paths(run_dir)
    paths["artifacts_dir"].mkdir(parents=True, exist_ok=True)
    if args.save_plots:
        paths["plots_dir"].mkdir(parents=True, exist_ok=True)
    base.ensure_parquet()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = base._resolve_dtype(device, args.dtype, args.runtime_profile)
    print("=== SAE replication preflight ===")
    print(f"run_id={args.run_id}")
    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"shortlist_csv={args.shortlist_csv}")
    print(f"top_n={args.top_n}")
    print(f"gating_mode={args.gating_mode}")
    print("===============================")
    if args.dry_run:
        return

    targets = select_targets(args, paths)
    overrides = _parse_sae_overrides(args.sae_id_overrides)

    model, tokenizer, female_id, male_id, device = base.load_model_and_tokenizer(args)
    saes = load_target_saes(
        target_layers=targets["layer"].astype(int).tolist(),
        device=device,
        overrides=overrides,
    )

    baseline, index = generate_baselines(
        args=args,
        model=model,
        tokenizer=tokenizer,
        female_id=female_id,
        male_id=male_id,
        device=device,
        paths=paths,
    )
    eligible_keys = index["eligible_keys"]

    activation_cache = cache_target_feature_activations(
        args=args,
        model=model,
        tokenizer=tokenizer,
        baseline=baseline,
        eligible_keys=eligible_keys,
        targets=targets,
        saes=saes,
        paths=paths,
    )

    if args.resume and base.checkpoint_exists(paths["raw"]) and base.checkpoint_exists(paths["summary"]) and base.checkpoint_exists(paths["pass_fail"]):
        print("Replication outputs already exist; skipping compute due to --resume")
        return

    raw_df = run_replication_ablation(
        baseline=baseline,
        eligible_keys=eligible_keys,
        targets=targets,
        activation_cache=activation_cache,
        tokenizer=tokenizer,
        female_id=female_id,
        male_id=male_id,
        model=model,
        saes=saes,
        gating_mode=args.gating_mode,
    )
    raw_df.to_parquet(paths["raw"], index=False)

    summary_df, pass_fail_df = build_replication_summary(raw_df)
    pass_fail_df["replication_pass"] = (
        (pass_fail_df["sign_consistency"] >= args.min_sign_consistency)
        & (pass_fail_df["median_norm_effect"] >= args.min_median_effect)
        & (pass_fail_df["n_runs"] >= args.min_runs)
    )
    summary_df.to_csv(paths["summary"], index=False)
    pass_fail_df.to_csv(paths["pass_fail"], index=False)
    save_replication_plots(raw_df, summary_df, pass_fail_df, paths, args)

    print("Replication run complete.")
    print(f"targets={len(targets)} eligible_traces={len(eligible_keys)} raw_rows={len(raw_df)}")
    print(f"artifacts_dir={paths['artifacts_dir']}")


if __name__ == "__main__":
    main()
