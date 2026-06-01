"""
Clinical discharge notes → pooled SAE activations → sparse L1 logistic probe vs gender.

Reference alignment
------------------
Pooling matches sae_bias ``get_predictive_latents.get_max_agg_latents``:

  ``f[:, 1:, :].max(dim=1)[0]``  (max pool over tokens, skip sequence index 0 / BOS)

Our encoder uses TransformerLens + SAELens via ``sae_localization.get_max_latents_for_text``
(same slicing / reduction).

Probe training matches ``sae_bias/get_predictive_latents.py`` ``get_coeff`` (lines 14–30):

  - ``StandardScaler`` fit on **train**, transform train + test (no sklearn Pipeline required)
  - ``LogisticRegression(penalty='l1', solver='liblinear', max_iter=..., C=...)``
    with **no** ``random_state`` (same as reference)
  - Coefficients sorted by descending ``abs(coef)``; AUROC on **held-out** test logits

Modes
-----
``--probe-mode sae_bias`` (default): one stratified **group** holdout (no subject in both splits),
then probe exactly as ``get_coeff``. Optional ``--train-csv`` / ``--test-csv`` for two-table
runs like the paper's ``*_train.csv`` / ``*_test.csv``.

``--probe-mode group_cv``: StratifiedGroupKFold mean AUROC + top-|coef| on **full data**
(-extension for stability; coefficients are **not** the same object as ``get_coeff``).

Example
-------
  python "clinical_gender_linear_probe.py" ^
    --csv path/to/gender_cohort.csv ^
    --output-dir ./probe_runs ^
    --run-id mimic_discharge ^
    --probe-mode sae_bias ^
    --model-preset qwen2.5_7b_instruct ^
    --sae-layer-preset qwen_superset ^
    --max-seq-len 8192 ^
    --C 1 --max-iter 1000

Requires: same env as sae_localization (transformer_lens, sae_lens, sklearn, tqdm).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_COT_PATCHING = _THIS_DIR.parent
if str(_COT_PATCHING) not in sys.path:
    sys.path.insert(0, str(_COT_PATCHING))

import sae_localization as sl  # noqa: E402


# ---------------------------------------------------------------------------
# Mirrors sae_bias/src/get_predictive_latents.py (probe core)
# ---------------------------------------------------------------------------
def get_coeff(
    train_latents: np.ndarray,
    train_labels: np.ndarray,
    test_latents: np.ndarray,
    test_labels: np.ndarray,
    C: float,
    top_k: int = 100,
    max_iter: int = 1000,
) -> Tuple[list[int], list[float], float]:
    """
    Same computation as ``sae_bias.src.get_predictive_latents.get_coeff``.
    Returns (sorted feature indices [:top_k], coeffs [:top_k], test AUROC).
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_latents)
    X_test = scaler.transform(test_latents)

    clf = LogisticRegression(penalty="l1", solver="liblinear", max_iter=max_iter, C=C)
    clf.fit(X_train, train_labels)
    coeff = clf.coef_.flatten()

    sorted_ixs = np.argsort(-np.abs(coeff)).tolist()
    sorted_coeffs = coeff[sorted_ixs].tolist()

    y_prob = clf.predict_proba(X_test)
    auroc = float(roc_auc_score(test_labels, y_prob[:, 1]))

    return sorted_ixs[:top_k], sorted_coeffs[:top_k], auroc


def normalize_gender_labels(raw: pd.Series) -> pd.Series:
    s = raw.astype(str).str.strip().str.upper()
    mapped = s.map(
        {
            "M": 0,
            "F": 1,
            "MALE": 0,
            "FEMALE": 1,
            "0": 0,
            "1": 1,
        }
    )
    return mapped


def load_cohort(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"subject_id", "text", "gender"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {missing}; have {list(df.columns)}")
    df = df.copy()
    df["label"] = normalize_gender_labels(df["gender"])
    bad = df["label"].isna()
    if bad.any():
        raise ValueError(
            f"Unmapped gender values (need M/F or Male/Female): {df.loc[bad, 'gender'].unique().tolist()}"
        )
    dup = df["subject_id"].duplicated(keep=False)
    if dup.any():
        n = df.loc[dup, "subject_id"].nunique()
        raise ValueError(
            f"Found duplicate subject_id rows ({n} patients with repeats). "
            "Deduplicate to one row per patient before running."
        )
    return df


def apply_tok_count_filter(df: pd.DataFrame, tok_max: int, col_candidates: Tuple[str, ...]) -> pd.DataFrame:
    """Analog of reference ``df[TOK_COUNT < 2000]`` when a token-count column exists."""
    if tok_max <= 0:
        return df
    for col in col_candidates:
        if col in df.columns:
            out = df[df[col] < tok_max].copy()
            if len(out) < len(df):
                print(f"Filtered by {col} < {tok_max}: {len(df)} -> {len(out)} rows")
            return out
    return df


def cap_per_label(df: pd.DataFrame, col_label: str, max_per_class: int) -> pd.DataFrame:
    """Analog of ``.groupby([GT_COL]).head(N)`` in get_predictive_latents.main."""
    if max_per_class <= 0:
        return df
    parts = []
    for v in sorted(df[col_label].unique()):
        g = df[df[col_label] == v].head(max_per_class)
        parts.append(g)
    out = pd.concat(parts, axis=0)
    print(f"Capped rows per label to {max_per_class}: kept {len(out)} total")
    return out


def truncate_for_model(model, text: str, max_seq_len: int) -> str:
    if max_seq_len <= 0:
        return text
    toks = model.to_tokens(text, prepend_bos=False)
    n = int(toks.shape[1])
    if n <= max_seq_len:
        return text
    slc = toks[0, :max_seq_len].tolist()
    return model.tokenizer.decode(slc)


def encode_layer_max_pool(
    model,
    texts: list[str],
    layer: int,
    sae: torch.nn.Module,
    batch_report_every: int = 0,
) -> np.ndarray:
    """Max-pooled SAE vector per note; matches sae_bias get_max_agg_latents pooling."""
    feats_list: list[np.ndarray] = []
    for i, t in enumerate(
        tqdm(texts, desc=f"encode layer {layer}", disable=len(texts) < 500)
    ):
        vec = sl.get_max_latents_for_text(model, t, layer, sae)
        feats_list.append(vec.astype(np.float32, copy=False))
        del vec
        if batch_report_every and (i + 1) % batch_report_every == 0:
            torch.cuda.empty_cache()
    return np.stack(feats_list, axis=0)


def probe_one_layer_group_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    C: float,
    seed: int,
    max_iter: int,
    top_k: int,
) -> tuple[float, list[float], list[tuple[int, float]]]:
    """Extension: grouped stratified CV + full-data refit for ranked features."""
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_aucs: list[float] = []
    for train_idx, test_idx in kf.split(X, y, groups):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        max_iter=max_iter,
                        C=C,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        proba = pipe.predict_proba(X[test_idx])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[test_idx], proba)))

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=max_iter,
                    C=C,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    w = pipe.named_steps["clf"].coef_[0].astype(np.float64)
    order = np.argsort(-np.abs(w))
    ranked: list[tuple[int, float]] = []
    for idx in order[:top_k]:
        ranked.append((int(idx), float(w[int(idx)])))
    return float(np.mean(fold_aucs)), fold_aucs, ranked


def stratified_group_holdout_indices(
    y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """First fold of StratifiedGroupKFold: disjoint groups, stratified labels."""
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(iter(kf.split(np.zeros(len(y)), y, groups)))


def build_args_ns(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(dtype=args.dtype, runtime_profile=args.runtime_profile)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clinical gender linear probe (SAELens); mirrors sae_bias get_coeff when desired")
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="Single cohort CSV (subject_id, text, gender). Ignored if --train-csv and --test-csv are set.",
    )
    p.add_argument("--train-csv", type=str, default="", help="Train table (same schema as single --csv); sae_bias-style split.")
    p.add_argument("--test-csv", type=str, default="", help="Test table; must be disjoint subject_id from train.")
    p.add_argument("--output-dir", type=str, default="clinical_gender_probe_runs")
    p.add_argument("--run-id", type=str, default="default")
    p.add_argument("--max-seq-len", type=int, default=0, help="Truncate to this many model tokens per note (0 = no truncate)")
    p.add_argument(
        "--tok-count-max",
        type=int,
        default=2000,
        help="If CSV has TOK_COUNT or tok_count, drop rows >= this value (reference uses 2000). 0 = skip.",
    )

    p.add_argument(
        "--probe-mode",
        type=str,
        choices=["sae_bias", "group_cv"],
        default="sae_bias",
        help="sae_bias: get_coeff on train vs test. group_cv: StratifiedGroupKFold + full refit.",
    )
    p.add_argument(
        "--holdout-n-splits",
        type=int,
        default=5,
        help="For single-CSV sae_bias mode: use first fold of StratifiedGroupKFold(n_splits=...).",
    )
    p.add_argument("--n-folds", type=int, default=5, help="Used only when --probe-mode group_cv.")
    p.add_argument(
        "--max-per-class-train",
        type=int,
        default=0,
        help="After loading train CSV, apply groupby(label).head(N). 0 = no cap (reference uses 500).",
    )
    p.add_argument(
        "--max-per-class-test",
        type=int,
        default=0,
        help="Same for test set (reference uses 100). 0 = no cap.",
    )

    p.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="L1 inverse strength; reference get_predictive_latents.main uses C=1.",
    )
    p.add_argument("--max-iter", type=int, default=1000, help="Reference get_coeff uses max_iter=1000.")
    p.add_argument("--top-k", type=int, default=100, help="Top |coef| latent indices saved (reference default 100).")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--reuse-activations", action="store_true")

    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--runtime-profile", type=str, default="a10", choices=["gh200", "a10"])
    p.add_argument(
        "--model-preset",
        type=str,
        default="qwen2.5_7b_instruct",
        choices=sorted(sl.MODEL_PRESETS.keys()),
    )
    p.add_argument("--model-name", type=str, default="")
    p.add_argument("--sae-releases", type=str, default="")
    p.add_argument(
        "--sae-layer-preset",
        type=str,
        default="qwen_superset",
        choices=sorted(sl.SAE_LAYER_PRESETS.keys()),
    )
    p.add_argument("--sae-layers", type=str, default="")
    p.add_argument("--sae-id-overrides", type=str, default="")
    return p.parse_args()


@dataclass
class LayerResult:
    layer: int
    sae_id: str
    test_auroc: float | None
    mean_cv_auc: float | None
    fold_aucs: list[float]
    top_features: list[dict[str, float | int]]


def _validate_disjoint_subjects(train_subj: np.ndarray, test_subj: np.ndarray) -> None:
    inter = np.intersect1d(np.unique(train_subj), np.unique(test_subj))
    if len(inter) > 0:
        raise ValueError(f"Train/test share {len(inter)} subject_id values; disjoint split required.")


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir).expanduser().resolve()
    run_dir = out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "activation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_tsv_dir = run_dir / "reference_style_tsv"
    results_tsv_dir.mkdir(parents=True, exist_ok=True)

    if args.train_csv and args.test_csv:
        df_train_raw = load_cohort(Path(args.train_csv).expanduser().resolve())
        df_test_raw = load_cohort(Path(args.test_csv).expanduser().resolve())
        df_train_raw = apply_tok_count_filter(df_train_raw, args.tok_count_max, ("TOK_COUNT", "tok_count"))
        df_test_raw = apply_tok_count_filter(df_test_raw, args.tok_count_max, ("TOK_COUNT", "tok_count"))
        df_train = cap_per_label(df_train_raw, "label", args.max_per_class_train)
        df_test = cap_per_label(df_test_raw, "label", args.max_per_class_test)
        _validate_disjoint_subjects(df_train["subject_id"].to_numpy(), df_test["subject_id"].to_numpy())
        splits_desc = {"mode": "two_csv", "train_csv": args.train_csv, "test_csv": args.test_csv}
        combined_for_encode = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
        n_train = len(df_train)
        train_idx_np = np.arange(n_train)
        test_idx_np = np.arange(n_train, n_train + len(df_test))
    else:
        if not args.csv:
            raise ValueError("Provide --csv, or both --train-csv and --test-csv.")
        csv_path = Path(args.csv).expanduser().resolve()
        df = load_cohort(csv_path)
        df = apply_tok_count_filter(df, args.tok_count_max, ("TOK_COUNT", "tok_count"))
        splits_desc = {"mode": "single_csv_holdout", "csv": str(csv_path)}
        combined_for_encode = df.reset_index(drop=True)
        y_all = combined_for_encode["label"].to_numpy(dtype=np.int64)
        groups_all = combined_for_encode["subject_id"].to_numpy()
        train_idx_np, test_idx_np = stratified_group_holdout_indices(y_all, groups_all, args.seed, args.holdout_n_splits)
        n_train = int(len(train_idx_np))
        splits_desc["holdout_n_splits"] = args.holdout_n_splits
        splits_desc["n_train"] = int(len(train_idx_np))
        splits_desc["n_test"] = int(len(test_idx_np))

    texts_raw = combined_for_encode["text"].astype(str).tolist()

    sl.ACTIVE_MODEL_NAME, sl.ACTIVE_SAE_RELEASES, sl.ACTIVE_SAE_FAMILY = sl.resolve_model_config(
        SimpleNamespace(
            model_preset=args.model_preset,
            model_name=args.model_name or "",
            sae_releases=args.sae_releases or "",
        )
    )
    sae_args = SimpleNamespace(
        sae_layers=args.sae_layers,
        sae_layer_preset=args.sae_layer_preset,
    )

    ns = build_args_ns(args)
    print("Loading model…")
    model, _tokenizer, _f, _m, device = sl.load_model_and_tokenizer(ns)

    if args.sae_layer_preset == "model_all" and not (args.sae_layers or "").strip():
        sl.SAE_LAYERS = sl.resolve_sae_layers(sae_args, model_n_layers=int(model.cfg.n_layers))
        print(f"Resolved model_all → n_layers={len(sl.SAE_LAYERS)}")
    else:
        sl.SAE_LAYERS = sl.resolve_sae_layers(sae_args)

    print(f"Model={sl.ACTIVE_MODEL_NAME} device={device} layers={sl.SAE_LAYERS}")

    truncate_len = args.max_seq_len
    texts = texts_raw if truncate_len <= 0 else [truncate_for_model(model, t, truncate_len) for t in tqdm(texts_raw, desc="truncate")]

    overrides: dict[int, str] = {}
    if args.sae_id_overrides.strip():
        raw = json.loads(args.sae_id_overrides)
        overrides = {int(k): str(v) for k, v in raw.items()}

    y_all_arr = combined_for_encode["label"].to_numpy(dtype=np.int64)

    manifest: dict[str, object] = {
        **splits_desc,
        "n_rows_encoded": len(combined_for_encode),
        "probe_mode": args.probe_mode,
        "mirrors_reference": (
            "sae_bias/src/get_predictive_latents.py get_coeff (+ get_max_agg_latents pooling pattern)"
            if args.probe_mode == "sae_bias"
            else "extension: StratifiedGroupKFold + Pipeline refit on full matrix"
        ),
        "max_seq_len": truncate_len,
        "tok_count_max_filter": args.tok_count_max,
        "label_counts_all": pd.Series(y_all_arr).value_counts().to_dict(),
        "model_preset": args.model_preset,
        "model_name": sl.ACTIVE_MODEL_NAME,
        "sae_releases": list(sl.ACTIVE_SAE_RELEASES),
        "sae_family": sl.ACTIVE_SAE_FAMILY,
        "sae_layers": list(sl.SAE_LAYERS),
        "C": args.C,
        "max_iter": args.max_iter,
        "top_k": args.top_k,
    }

    layer_results: list[LayerResult] = []

    for layer in tqdm(sl.SAE_LAYERS, desc="per-layer SAE"):
        cache_path = cache_dir / f"layer_{layer}_maxpool_X.npy"
        cache_meta = cache_dir / f"layer_{layer}_meta.json"

        if args.reuse_activations and cache_path.is_file():
            X = np.load(cache_path)
            with open(cache_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            sae_id = str(meta.get("sae_id", ""))
        else:
            sae, sae_id = sl.load_sae_for_layer(layer, model.cfg.device, overrides)
            X = encode_layer_max_pool(model, texts, layer, sae)
            del sae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            np.save(cache_path, X)
            with open(cache_meta, "w", encoding="utf-8") as f:
                json.dump({"layer": layer, "sae_id": sae_id, "shape": list(X.shape)}, f, indent=2)

        if args.probe_mode == "sae_bias":
            X_tr, y_tr = X[train_idx_np], y_all_arr[train_idx_np]
            X_te, y_te = X[test_idx_np], y_all_arr[test_idx_np]
            sorted_ixs, sorted_coeffs, auroc = get_coeff(
                X_tr, y_tr, X_te, y_te, C=args.C, top_k=args.top_k, max_iter=args.max_iter
            )
            print(f"layer {layer} sae_id={sae_id} test_AUROC (get_coeff): {auroc:.4f}")
            top_features = [{"feature_idx": int(i), "coef": float(c)} for i, c in zip(sorted_ixs, sorted_coeffs)]
            layer_results.append(
                LayerResult(
                    layer=layer,
                    sae_id=sae_id,
                    test_auroc=auroc,
                    mean_cv_auc=None,
                    fold_aucs=[],
                    top_features=top_features,
                )
            )
            tsv_path = results_tsv_dir / f"top_pred_latents_layer_{layer}.tsv"
            pd.DataFrame({"latent": sorted_ixs, "coeff": sorted_coeffs}).to_csv(tsv_path, sep="\t", index=False)
        else:
            groups_all = combined_for_encode["subject_id"].to_numpy()
            mean_auc, fold_aucs, ranked = probe_one_layer_group_cv(
                X,
                y_all_arr,
                groups_all,
                n_splits=args.n_folds,
                C=args.C,
                seed=args.seed,
                max_iter=args.max_iter,
                top_k=args.top_k,
            )
            print(f"layer {layer} sae_id={sae_id} mean_CV_AUC: {mean_auc:.4f}")
            top_features = [{"feature_idx": i, "coef": c} for i, c in ranked]
            layer_results.append(
                LayerResult(
                    layer=layer,
                    sae_id=sae_id,
                    test_auroc=None,
                    mean_cv_auc=mean_auc,
                    fold_aucs=fold_aucs,
                    top_features=top_features,
                )
            )

    out_json = {
        "manifest": manifest,
        "layers": [
            {
                "layer": r.layer,
                "sae_id": r.sae_id,
                "test_auroc": r.test_auroc,
                "mean_cv_auc": r.mean_cv_auc,
                "fold_aucs": r.fold_aucs,
                "top_features": r.top_features,
            }
            for r in layer_results
        ],
    }
    summary_path = run_dir / "clinical_gender_probe_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"Wrote {summary_path}")
    if args.probe_mode == "sae_bias":
        print(f"Reference-style TSVs under {results_tsv_dir}")


if __name__ == "__main__":
    main()
