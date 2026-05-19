"""
Max-activating token contexts for one (layer, SAE feature) on clinical notes.

Bridges linear-probe ``feature_idx`` (a column in the max-pooled SAE design matrix)
to **which subwords spike** that feature on your text, using the same
``HookedTransformer`` + SAELens path as ``get_max_latents_for_text``.

Single-feature example::

  python sae_feature_token_mae.py `
    --csv unique_patient_gender_cohort.csv `
    --model-preset gemma_scope2_4b_it `
    --layer 18 `
    --feature-idx 11292 `
    --sae-meta-json probe_meta/layer_18_meta.json `
    --global-top 25

Batch (one run per layer × positive / negative coef from probe TSVs)::

  python sae_feature_token_mae.py `
    --csv cohort/cohort.csv `
    --model-preset gemma_scope2_4b_it `
    --probe-tsv-dir path/to/reference_style_tsv `
    --probe-meta-dir path/to/activation_cache `
    --layer-range 4-28 `
    --max-notes 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_LINEAR_PROBE = Path(__file__).resolve().parent
_ROOT = _LINEAR_PROBE.parent
# Repo layout: .../cot_patching/linear probe/this_script.py -> sae_localization.py in parent.
# Flat layout (e.g. ~/gemma_mae/linear probe + ~/gemma_mae/cot_patching): module in cot_patching subdir.
_PY_PATH_INSERTS: list[Path] = []
for _candidate in (_ROOT, _ROOT / "cot_patching"):
    if (_candidate / "sae_localization.py").is_file():
        _PY_PATH_INSERTS.append(_candidate)
_PY_PATH_INSERTS.append(_LINEAR_PROBE)
for _p in _PY_PATH_INSERTS:
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import torch

import sae_localization as sl  # noqa: E402
from clinical_gender_linear_probe import apply_tok_count_filter, load_cohort, truncate_for_model  # noqa: E402


def token_window_str(token_strs: list[str], pos: int, radius: int) -> str:
    lo = max(0, pos - radius)
    hi = min(len(token_strs), pos + radius + 1)
    return "".join(token_strs[lo:hi])


def parse_layer_range(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return []
    if "-" in spec:
        left, _, right = spec.partition("-")
        lo, hi = int(left.strip()), int(right.strip())
        if hi < lo:
            raise ValueError(f"Invalid --layer-range: high < low ({spec})")
        return list(range(lo, hi + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def pos_neg_latents_from_tsv(tsv_path: Path) -> tuple[int, float, int, float]:
    """
    From probe reference TSV with columns latent, coef: return
    (latent_max_positive_coef, coef_max_pos, latent_max_negative_coef, coef_max_neg).
    """
    df = pd.read_csv(tsv_path, sep="\t")
    if "latent" not in df.columns or "coeff" not in df.columns:
        raise ValueError(f"{tsv_path}: expected tabs and columns latent, coeff; got {list(df.columns)}")
    coeffs = df["coeff"].astype(float)
    latent = df["latent"].astype(int)
    i_pos = int(coeffs.values.argmax())
    i_neg = int(coeffs.values.argmin())
    lp, cp = int(latent.iloc[i_pos]), float(coeffs.iloc[i_pos])
    ln, cn = int(latent.iloc[i_neg]), float(coeffs.iloc[i_neg])
    return lp, cp, ln, cn


def build_sae_override_for_layer(meta_dir: Path | None, layer: int, meta_file: Path | None) -> dict[int, str]:
    if meta_file is not None:
        meta = json.loads(meta_file.expanduser().read_text(encoding="utf-8"))
        ml = int(meta["layer"])
        if ml != layer:
            raise ValueError(f"{meta_file}: layer mismatch (json has {ml}, requested {layer})")
        return {layer: str(meta["sae_id"])}
    if meta_dir is None:
        return {}
    p = meta_dir / f"layer_{layer}_meta.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p} (--probe-meta-dir)")
    meta = json.loads(p.read_text(encoding="utf-8"))
    ml = int(meta["layer"])
    if ml != layer:
        raise ValueError(f"{p}: layer mismatch (json has {ml}, requested {layer})")
    return {layer: str(meta["sae_id"])}


@dataclass
class MaeRunConfig:
    max_seq_len: int
    tok_count_max: int
    skip_first_token: bool
    local_top_per_note: int
    global_top: int
    class_top: int
    enrichment_top_n: int
    enrichment_min_count: int
    enrichment_show_k: int
    context_radius: int
    output_jsonl: str


def run_single_mae(
    *,
    cfg: MaeRunConfig,
    df: pd.DataFrame,
    model,
    layer: int,
    feature_idx: int,
    sae_override: dict[int, str],
    polarity_label: str = "",
    sae_obj=None,
    sae_id_cached: str = "",
) -> str:
    """
    Run MAE for one latent. If sae_obj is None, loads SAE (and deletes after if created here).
    Returns sae_id string.
    """
    own_sae = sae_obj is None
    if own_sae:
        sae, sae_id = sl.load_sae_for_layer(layer, model.cfg.device, sae_override)
    else:
        sae = sae_obj
        sae_id = sae_id_cached

    header = (
        f"Model={sl.ACTIVE_MODEL_NAME} layer={layer} feature={feature_idx} SAE={sae_id}"
        + (f"  [{polarity_label}]" if polarity_label else "")
    )
    print(header)
    print("=" * min(100, len(header) + 5))

    skip0 = 1 if cfg.skip_first_token else 0
    cands: list[tuple[float, int, int, str, int, str, str]] = []

    texts_raw = df["text"].astype(str).tolist()
    subjs = df["subject_id"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    for row_i, (subj, label, text_raw) in enumerate(zip(subjs, labels, texts_raw)):
        text = truncate_for_model(model, text_raw, cfg.max_seq_len)
        acts, token_strs = sl.per_token_sae_feature_acts(model, text, layer, sae, feature_idx)
        slc = acts[skip0:]
        if slc.size == 0:
            continue
        k = min(cfg.local_top_per_note, slc.size)
        part = np.argpartition(-slc, k - 1)[:k]
        order = part[np.argsort(-slc[part])]
        for j in order:
            pos = int(skip0 + j)
            act = float(acts[pos])
            token_str = token_strs[pos]
            win = token_window_str(token_strs, pos, cfg.context_radius)
            cands.append((act, row_i, pos, subj, label, token_str, win))

    cands.sort(key=lambda x: -x[0])
    ranked = cands[: cfg.global_top]
    ranked_f = [x for x in cands if x[4] == 1][: cfg.class_top]
    ranked_m = [x for x in cands if x[4] == 0][: cfg.class_top]

    print(f"\nTop {len(ranked)} token spikes (feature {feature_idx}, layer {layer}):")
    for rank, (act, row_i, pos, subj, label, token_str, win) in enumerate(ranked, start=1):
        repl_win = win.replace("\n", "\\n")
        if len(repl_win) > 240:
            repl_win = repl_win[:240] + "..."
        tok_disp = token_str.replace("\n", "\\n")
        print(f"  {rank:2d}. act={act:.4f}  row={row_i}  pos={pos}  label={label}  subject_id={subj}")
        print(f"      token={tok_disp!r}")
        print(f"      {repl_win!r}")

    if cfg.class_top > 0:
        print(f"\nTop {len(ranked_f)} spikes in female-labeled notes (label=1):")
        for rank, (act, row_i, pos, subj, _label, token_str, win) in enumerate(ranked_f, start=1):
            repl_win = win.replace("\n", "\\n")
            if len(repl_win) > 220:
                repl_win = repl_win[:220] + "..."
            tok_disp = token_str.replace("\n", "\\n")
            print(f"  F{rank:02d}. act={act:.4f}  row={row_i}  pos={pos}  subject_id={subj}")
            print(f"      token={tok_disp!r}  window={repl_win!r}")

        print(f"\nTop {len(ranked_m)} spikes in male-labeled notes (label=0):")
        for rank, (act, row_i, pos, subj, _label, token_str, win) in enumerate(ranked_m, start=1):
            repl_win = win.replace("\n", "\\n")
            if len(repl_win) > 220:
                repl_win = repl_win[:220] + "..."
            tok_disp = token_str.replace("\n", "\\n")
            print(f"  M{rank:02d}. act={act:.4f}  row={row_i}  pos={pos}  subject_id={subj}")
            print(f"      token={tok_disp!r}  window={repl_win!r}")

    if cfg.enrichment_top_n > 0:
        enrich_pool = cands[: cfg.enrichment_top_n]
        tok_f: Counter[str] = Counter()
        tok_m: Counter[str] = Counter()
        for _act, _row_i, _pos, _subj, label, token_str, _win in enrich_pool:
            key = token_str.replace("\n", "\\n")
            if label == 1:
                tok_f[key] += 1
            else:
                tok_m[key] += 1

        keys = set(tok_f.keys()) | set(tok_m.keys())
        rows = []
        for key in keys:
            f = tok_f.get(key, 0)
            m = tok_m.get(key, 0)
            total = f + m
            if total < cfg.enrichment_min_count:
                continue
            rows.append(
                {
                    "token": key,
                    "female_count": f,
                    "male_count": m,
                    "total": total,
                    "female_enrichment_ratio": (f + 1.0) / (m + 1.0),
                    "male_enrichment_ratio": (m + 1.0) / (f + 1.0),
                }
            )
        rows_f = sorted(rows, key=lambda r: (-r["female_enrichment_ratio"], -r["total"]))
        rows_m = sorted(rows, key=lambda r: (-r["male_enrichment_ratio"], -r["total"]))

        print(
            f"\nToken enrichment from top {len(enrich_pool)} global spikes "
            f"(min_count={cfg.enrichment_min_count}):"
        )
        print(f"  Female-enriched tokens (top {min(cfg.enrichment_show_k, len(rows_f))}):")
        for r in rows_f[: cfg.enrichment_show_k]:
            print(
                "   "
                f"{r['token']!r}  F={r['female_count']} M={r['male_count']} "
                f"ratio={r['female_enrichment_ratio']:.2f}"
            )
        print(f"  Male-enriched tokens (top {min(cfg.enrichment_show_k, len(rows_m))}):")
        for r in rows_m[: cfg.enrichment_show_k]:
            print(
                "   "
                f"{r['token']!r}  F={r['female_count']} M={r['male_count']} "
                f"ratio={r['male_enrichment_ratio']:.2f}"
            )

    out_path = cfg.output_jsonl.strip()
    if out_path:
        out = Path(out_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fp:
            for rank, (act, row_i, pos, subj, label, token_str, win) in enumerate(ranked, start=1):
                fp.write(
                    json.dumps(
                        {
                            "rank": rank,
                            "activation": act,
                            "row_index": row_i,
                            "token_pos": pos,
                            "subject_id": subj,
                            "label": label,
                            "token_str": token_str,
                            "token_window": win,
                            "layer": layer,
                            "feature_idx": feature_idx,
                            "sae_id": sae_id,
                            "polarity": polarity_label,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"Wrote {out}")

    if own_sae:
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return sae_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-token SAE feature activations (MAE-style) on cohort text. "
        "Single (layer + feature-idx) or batch from probe TSVs (--probe-tsv-dir + --layer-range)."
    )
    p.add_argument("--csv", type=str, required=True)
    p.add_argument(
        "--model-preset",
        type=str,
        default="qwen2.5_7b_instruct",
        choices=sorted(sl.MODEL_PRESETS.keys()),
    )
    p.add_argument("--model-name", type=str, default="")
    p.add_argument("--sae-releases", type=str, default="")
    # Single-feature mode
    p.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="SAE/transformer block index. Not used when --probe-tsv-dir is set.",
    )
    p.add_argument(
        "--feature-idx",
        type=int,
        default=-1,
        help="Sparse feature index. Not used when --probe-tsv-dir is set.",
    )
    p.add_argument(
        "--sae-meta-json",
        type=str,
        default="",
        help="activation_cache/layer_L_meta.json (single-feature mode preferred).",
    )
    # Batch-from-TSV mode
    p.add_argument(
        "--probe-tsv-dir",
        type=str,
        default="",
        help="Folder with top_pred_latents_layer_L.tsv files; runs coef-max and coef-min latent per layer.",
    )
    p.add_argument(
        "--probe-meta-dir",
        type=str,
        default="",
        help=(
            "Folder with layer_<L>_meta.json for each layer in batch (typically activation_cache). "
            "Required for batch mode."
        ),
    )
    p.add_argument(
        "--layer-range",
        type=str,
        default="",
        help='Inclusive layers, e.g. "4-28" or "3,7,11" (batch mode).',
    )
    p.add_argument("--sae-id-override", type=str, default="", help="Force sae_id (single-layer mode only).")
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--tok-count-max", type=int, default=2000)
    p.add_argument("--max-notes", type=int, default=0, help="0 = all rows after filters.")
    p.add_argument(
        "--skip-first-token",
        action="store_true",
        help="Ignore position 0 when ranking (matches max-pool over feats[:, 1:, :]).",
    )
    p.add_argument("--local-top-per-note", type=int, default=5)
    p.add_argument("--global-top", type=int, default=30)
    p.add_argument("--class-top", type=int, default=20)
    p.add_argument("--enrichment-top-n", type=int, default=400)
    p.add_argument("--enrichment-min-count", type=int, default=2)
    p.add_argument("--enrichment-show-k", type=int, default=12)
    p.add_argument("--context-radius", type=int, default=6)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--runtime-profile", type=str, default="a10", choices=["gh200", "a10"])
    p.add_argument(
        "--output-jsonl",
        type=str,
        default="",
        help="Single-feature: write path. Batch: optional directory prefix — each job writes "
        "layer{L}_{pos|neg}_feat{idx}.jsonl underneath if this is a directory, else appended suffix.",
    )
    args = p.parse_args()
    return args


def _cfg_from_args(args: argparse.Namespace) -> MaeRunConfig:
    return MaeRunConfig(
        max_seq_len=args.max_seq_len,
        tok_count_max=args.tok_count_max,
        skip_first_token=args.skip_first_token,
        local_top_per_note=args.local_top_per_note,
        global_top=args.global_top,
        class_top=args.class_top,
        enrichment_top_n=args.enrichment_top_n,
        enrichment_min_count=args.enrichment_min_count,
        enrichment_show_k=args.enrichment_show_k,
        context_radius=args.context_radius,
        output_jsonl=args.output_jsonl.strip(),
    )


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    df = load_cohort(csv_path)
    df = apply_tok_count_filter(df, args.tok_count_max, ("TOK_COUNT", "tok_count"))
    if args.max_notes > 0:
        df = df.head(args.max_notes)

    cfg = _cfg_from_args(args)

    sl.ACTIVE_MODEL_NAME, sl.ACTIVE_SAE_RELEASES, sl.ACTIVE_SAE_FAMILY = sl.resolve_model_config(
        SimpleNamespace(
            model_preset=args.model_preset,
            model_name=args.model_name or "",
            sae_releases=args.sae_releases or "",
        )
    )
    ns = SimpleNamespace(dtype=args.dtype, runtime_profile=args.runtime_profile)
    model, _tok, _f, _m, _device = sl.load_model_and_tokenizer(ns)

    # --- Batch mode ---
    if (args.probe_tsv_dir or "").strip():
        tsv_root = Path(args.probe_tsv_dir).expanduser().resolve()
        if not args.layer_range.strip():
            raise SystemExit("Batch mode: pass --layer-range (e.g. 4-28)")
        layers = parse_layer_range(args.layer_range)
        if not layers:
            raise SystemExit('Batch mode: --layer-range produced no layers (use e.g. "4-28" or "3,7,11").')
        meta_dir_path = Path(args.probe_meta_dir).expanduser().resolve() if args.probe_meta_dir.strip() else None
        if meta_dir_path is None or not meta_dir_path.is_dir():
            raise SystemExit(
                "Batch mode requires --probe-meta-dir pointing to activation_cache containing layer_<L>_meta.json files."
            )
        batch_out_parent: Path | None = None
        out_base = cfg.output_jsonl
        if out_base:
            p = Path(out_base).expanduser()
            if str(out_base).endswith("/") or p.is_dir():
                batch_out_parent = Path(out_base).expanduser().resolve()
                batch_out_parent.mkdir(parents=True, exist_ok=True)

        print(f"BATCH MAE  layers={layers[0]}..{layers[-1]} ({len(layers)} total)  tsv_root={tsv_root}")
        for layer in layers:
            tsv_path = tsv_root / f"top_pred_latents_layer_{layer}.tsv"
            if not tsv_path.is_file():
                print(f"SKIP layer {layer}: missing {tsv_path.name}")
                continue
            lp, cp, ln, cn = pos_neg_latents_from_tsv(tsv_path)
            print(
                f"\n>>> Layer {layer} from {tsv_path.name}: "
                f"pos latent={lp} (coef={cp:.6g}); neg latent={ln} (coef={cn:.6g})"
            )
            try:
                sae_ov = build_sae_override_for_layer(meta_dir_path, layer, None)
            except Exception as e:
                print(f"SKIP layer {layer}: {e}")
                continue

            for feat_idx, polarity, tag in [(lp, "coef_max_positive", "pos"), (ln, "coef_min_negative", "neg")]:
                jr = replace(cfg, output_jsonl="")
                if batch_out_parent is not None:
                    jr.output_jsonl = str(
                        batch_out_parent / f"layer{layer}_{tag}_feat{feat_idx}.jsonl"
                    )
                elif out_base and Path(out_base).suffix == ".jsonl":
                    jr.output_jsonl = str(Path(out_base).with_name(f"L{layer}_{tag}_{Path(out_base).name}"))

                run_single_mae(
                    cfg=jr,
                    df=df,
                    model=model,
                    layer=layer,
                    feature_idx=int(feat_idx),
                    sae_override=sae_ov,
                    polarity_label=polarity,
                )
                print("\n")

        print("Batch MAE finished.")
        return

    # --- Single-feature mode ---
    if args.layer < 0 or args.feature_idx < 0:
        raise SystemExit("Single mode requires --layer and --feature-idx (or use batch with --probe-tsv-dir).")

    overrides_final: dict[int, str] = {}
    single_meta = Path(args.sae_meta_json).expanduser() if args.sae_meta_json.strip() else None
    if single_meta is not None and single_meta.is_file():
        overrides_final.update(build_sae_override_for_layer(None, args.layer, single_meta))
    if args.sae_id_override.strip():
        overrides_final[int(args.layer)] = args.sae_id_override.strip()
    run_single_mae(
        cfg=cfg,
        df=df,
        model=model,
        layer=args.layer,
        feature_idx=args.feature_idx,
        sae_override=overrides_final,
        polarity_label="",
    )


if __name__ == "__main__":
    main()
