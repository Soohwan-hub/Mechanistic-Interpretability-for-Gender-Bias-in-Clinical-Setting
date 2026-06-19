#!/usr/bin/env python3
"""Analyze scaled vignette generations for patch audit, bias ratio, and quality.

This script is post-hoc and reads an existing run directory produced by:
`generate_scaled_vignettes.py`.

Outputs are written under:
  <run_dir>/analysis/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sorted_factor_keys(keys: Iterable[str]) -> List[str]:
    return sorted(set(keys), key=lambda x: _to_float(x))


def _male_female_ratio(male_n: int, female_n: int) -> float:
    if female_n == 0 and male_n > 0:
        return float("inf")
    if female_n == 0:
        return float("nan")
    return male_n / female_n


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return num / den


def _factor_aggregate(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["factor"])].append(row)

    summary: List[Dict[str, Any]] = []
    for factor in _sorted_factor_keys(grouped.keys()):
        group = grouped[factor]
        n = len(group)
        male_n = sum(1 for r in group if r.get("predicted_gender") == "Male")
        female_n = sum(1 for r in group if r.get("predicted_gender") == "Female")
        unknown_n = sum(1 for r in group if r.get("predicted_gender") == "Unknown")
        ambiguous_n = sum(1 for r in group if r.get("predicted_gender") == "Ambiguous")
        success_n = sum(1 for r in group if str(r.get("is_success")) == "True")
        summary.append(
            {
                "factor": factor,
                "n": n,
                "male_n": male_n,
                "female_n": female_n,
                "unknown_n": unknown_n,
                "ambiguous_n": ambiguous_n,
                "success_n": success_n,
                "male_rate": _safe_div(male_n, n),
                "female_rate": _safe_div(female_n, n),
                "target_success_rate": _safe_div(success_n, n),
                "male_to_female_ratio": _male_female_ratio(male_n, female_n),
            }
        )
    return summary


def _cohort_factor_aggregate(rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["cohort"]), str(row["factor"]))].append(row)

    summary: List[Dict[str, Any]] = []
    for (cohort, factor), group in sorted(grouped.items(), key=lambda x: (x[0][0], _to_float(x[0][1]))):
        n = len(group)
        male_n = sum(1 for r in group if r.get("predicted_gender") == "Male")
        female_n = sum(1 for r in group if r.get("predicted_gender") == "Female")
        unknown_n = sum(1 for r in group if r.get("predicted_gender") == "Unknown")
        ambiguous_n = sum(1 for r in group if r.get("predicted_gender") == "Ambiguous")
        success_n = sum(1 for r in group if str(r.get("is_success")) == "True")
        summary.append(
            {
                "cohort": cohort,
                "factor": factor,
                "n": n,
                "male_n": male_n,
                "female_n": female_n,
                "unknown_n": unknown_n,
                "ambiguous_n": ambiguous_n,
                "success_n": success_n,
                "male_rate": _safe_div(male_n, n),
                "female_rate": _safe_div(female_n, n),
                "target_success_rate": _safe_div(success_n, n),
                "male_to_female_ratio": _male_female_ratio(male_n, female_n),
            }
        )
    return summary


def _factor_change_report(factor_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_factor = {str(r["factor"]): r for r in factor_rows}
    by_factor_numeric = {round(_to_float(r["factor"]), 8): r for r in factor_rows}
    rows: List[Dict[str, Any]] = []
    if "1.0" in by_factor and "5.0" in by_factor:
        base = by_factor["1.0"]
        end = by_factor["5.0"]
    elif 1.0 in by_factor_numeric and 5.0 in by_factor_numeric:
        base = by_factor_numeric[1.0]
        end = by_factor_numeric[5.0]
    else:
        return rows

    rows.append(
        {
            "comparison": "factor1_to_factor5",
            "base_factor": str(base["factor"]),
            "target_factor": str(end["factor"]),
            "base_male_to_female_ratio": base["male_to_female_ratio"],
            "target_male_to_female_ratio": end["male_to_female_ratio"],
            "delta_ratio": _to_float(end["male_to_female_ratio"]) - _to_float(base["male_to_female_ratio"]),
            "base_male_rate": base["male_rate"],
            "target_male_rate": end["male_rate"],
            "delta_male_rate": _to_float(end["male_rate"]) - _to_float(base["male_rate"]),
            "base_female_rate": base["female_rate"],
            "target_female_rate": end["female_rate"],
            "delta_female_rate": _to_float(end["female_rate"]) - _to_float(base["female_rate"]),
            "base_success_rate": base["target_success_rate"],
            "target_success_rate": end["target_success_rate"],
            "delta_success_rate": _to_float(end["target_success_rate"]) - _to_float(base["target_success_rate"]),
        }
    )
    return rows


def _clean_piece(piece: str) -> str:
    return piece.strip().lower().replace(" ", "").replace("▁", "")


def _expected_pieces_heuristic(condition_name: str, patch_subtoken: str) -> List[str]:
    words = [x for x in condition_name.lower().split() if x]
    if not words:
        return []
    if patch_subtoken == "first":
        return [words[0]]
    if patch_subtoken == "last":
        return [words[-1]]
    return words


def _build_patch_audit(
    rows: Sequence[Dict[str, str]],
    patch_subtoken: str,
    tokenizer_decoders: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["cohort"]), str(row["prompt_id"]))].append(row)

    out_rows: List[Dict[str, Any]] = []
    for (cohort, prompt_id), group in sorted(grouped.items(), key=lambda x: (x[0][0], int(x[0][1]))):
        condition_name = str(group[0]["condition_name"])
        observed_unique = sorted(set(str(r.get("patch_token_to_decoded", "")) for r in group))
        observed = observed_unique[0] if observed_unique else ""
        observed_parts = [x for x in observed.split("|") if x]

        if tokenizer_decoders is not None:
            expected = tokenizer_decoders[condition_name]
            expected_parts = [x for x in expected.split("|") if x]
            expected_mode = "tokenizer_exact"
        else:
            expected_parts = _expected_pieces_heuristic(condition_name, patch_subtoken)
            expected = "|".join(expected_parts)
            expected_mode = "heuristic"

        observed_clean = [_clean_piece(x) for x in observed_parts]
        expected_clean = [_clean_piece(x) for x in expected_parts]
        is_exact_match = observed_clean == expected_clean

        out_rows.append(
            {
                "cohort": cohort,
                "prompt_id": prompt_id,
                "condition_name": condition_name,
                "n_rows": len(group),
                "patch_subtoken": patch_subtoken,
                "expected_mode": expected_mode,
                "expected_patch_token_to_decoded": expected,
                "observed_patch_token_to_decoded": observed,
                "n_unique_observed_tokens": len(observed_unique),
                "exact_match": is_exact_match,
            }
        )
    return out_rows


def _get_tokenizer_expected_map(model_name: str, condition_names: Iterable[str], patch_subtoken: str) -> Dict[str, str]:
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    out: Dict[str, str] = {}
    for condition_name in condition_names:
        ids = tokenizer(" " + condition_name, add_special_tokens=False)["input_ids"]
        if not ids:
            out[condition_name] = ""
            continue
        if patch_subtoken == "first":
            selected = [ids[0]]
        elif patch_subtoken == "last":
            selected = [ids[-1]]
        else:
            selected = ids
        out[condition_name] = "|".join(tokenizer.decode([tid]).replace("\n", "\\n") for tid in selected)
    return out


def _compute_perplexity(
    rows: Sequence[Dict[str, str]],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> List[Dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()

    indexed = list(enumerate(rows))
    out_rows: List[Dict[str, Any]] = []

    for start in range(0, len(indexed), batch_size):
        chunk = indexed[start : start + batch_size]
        texts = [str(row.get("generated_text", "")) for _, row in chunk]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(resolved_device)
        attention_mask = enc["attention_mask"].to(resolved_device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())
        token_nll = losses * shift_mask
        token_counts = shift_mask.sum(dim=1).clamp(min=1)
        avg_nll = token_nll.sum(dim=1) / token_counts
        ppl = torch.exp(avg_nll).detach().cpu().tolist()
        tok_counts = token_counts.detach().cpu().tolist()

        for (idx, row), row_ppl, row_toks in zip(chunk, ppl, tok_counts):
            out_rows.append(
                {
                    "row_index": idx,
                    "cohort": row["cohort"],
                    "prompt_id": row["prompt_id"],
                    "factor": row["factor"],
                    "predicted_gender": row["predicted_gender"],
                    "is_success": row["is_success"],
                    "ppl": float(row_ppl),
                    "token_count": int(row_toks),
                }
            )

    out_rows.sort(key=lambda r: int(r["row_index"]))
    return out_rows


def _percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    vals = sorted(xs)
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return vals[lo]
    w = idx - lo
    return vals[lo] * (1 - w) + vals[hi] * w


def _ppl_summary_by_factor(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["factor"])].append(float(row["ppl"]))

    out_rows: List[Dict[str, Any]] = []
    for factor in _sorted_factor_keys(grouped.keys()):
        vals = grouped[factor]
        out_rows.append(
            {
                "factor": factor,
                "n": len(vals),
                "mean_ppl": sum(vals) / len(vals),
                "median_ppl": median(vals),
                "p10_ppl": _percentile(vals, 0.10),
                "p90_ppl": _percentile(vals, 0.90),
            }
        )
    return out_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a scaled-vignette run for patch validity, gender-ratio changes, and PPL."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory that contains config.json and all_generations.tsv",
    )
    parser.add_argument(
        "--all-generations",
        type=Path,
        default=None,
        help="Optional explicit path to all_generations.tsv (defaults to <run-dir>/all_generations.tsv).",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Optional analysis output directory (defaults to <run-dir>/analysis).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Optional tokenizer/PPL model override (defaults to model_name in config.json).",
    )
    parser.add_argument(
        "--skip-tokenizer-audit",
        action="store_true",
        help="Skip tokenizer-based expected-token audit and use heuristic matching only.",
    )
    parser.add_argument(
        "--compute-perplexity",
        action="store_true",
        help="Compute PPL on generated text for quality drift checks.",
    )
    parser.add_argument(
        "--ppl-model",
        type=str,
        default="",
        help="Optional model for PPL scoring (defaults to --model-name / config model_name).",
    )
    parser.add_argument("--ppl-batch-size", type=int, default=8)
    parser.add_argument("--ppl-max-length", type=int, default=256)
    parser.add_argument("--ppl-device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--ppl-max-rows",
        type=int,
        default=0,
        help="If >0, score a random subset of this many rows for faster approximate PPL.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir
    all_generations_path = args.all_generations or (run_dir / "all_generations.tsv")
    analysis_dir = args.analysis_dir or (run_dir / "analysis")
    config_path = run_dir / "config.json"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not all_generations_path.exists():
        raise FileNotFoundError(f"Missing generations TSV: {all_generations_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    model_name = args.model_name.strip() or str(config.get("model_name", ""))
    patch_subtoken = str(config.get("patch_subtoken", "last"))

    rows = _read_tsv(all_generations_path)
    if not rows:
        raise ValueError(f"No rows found in {all_generations_path}")

    analysis_dir.mkdir(parents=True, exist_ok=True)

    factor_rows = _factor_aggregate(rows)
    cohort_factor_rows = _cohort_factor_aggregate(rows)
    factor_change_rows = _factor_change_report(factor_rows)
    _write_tsv(analysis_dir / "ratio_by_factor.tsv", factor_rows)
    _write_tsv(analysis_dir / "ratio_by_cohort_factor.tsv", cohort_factor_rows)
    _write_tsv(analysis_dir / "factor1_to_factor5_change.tsv", factor_change_rows)

    tokenizer_expected: Optional[Dict[str, str]] = None
    tokenizer_audit_mode = "heuristic"
    if not args.skip_tokenizer_audit and model_name:
        try:
            condition_names = sorted(set(str(r["condition_name"]) for r in rows))
            tokenizer_expected = _get_tokenizer_expected_map(
                model_name=model_name,
                condition_names=condition_names,
                patch_subtoken=patch_subtoken,
            )
            tokenizer_audit_mode = "tokenizer_exact"
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] tokenizer audit unavailable ({exc}); falling back to heuristic audit.",
                flush=True,
            )

    patch_audit_rows = _build_patch_audit(
        rows=rows,
        patch_subtoken=patch_subtoken,
        tokenizer_decoders=tokenizer_expected,
    )
    _write_tsv(analysis_dir / "patch_token_audit.tsv", patch_audit_rows)

    metadata = {
        "run_dir": str(run_dir),
        "all_generations_path": str(all_generations_path),
        "analysis_dir": str(analysis_dir),
        "n_rows": len(rows),
        "model_name": model_name,
        "patch_subtoken": patch_subtoken,
        "tokenizer_audit_mode": tokenizer_audit_mode,
        "files_written": [
            "ratio_by_factor.tsv",
            "ratio_by_cohort_factor.tsv",
            "factor1_to_factor5_change.tsv",
            "patch_token_audit.tsv",
        ],
    }

    if args.compute_perplexity:
        ppl_model = args.ppl_model.strip() or model_name
        if not ppl_model:
            raise ValueError("PPL requested but model_name is empty. Pass --ppl-model.")

        sampled_rows = list(rows)
        if args.ppl_max_rows > 0 and args.ppl_max_rows < len(sampled_rows):
            rng = random.Random(args.seed)
            sampled_rows = [sampled_rows[i] for i in sorted(rng.sample(range(len(sampled_rows)), args.ppl_max_rows))]

        ppl_rows = _compute_perplexity(
            rows=sampled_rows,
            model_name=ppl_model,
            batch_size=args.ppl_batch_size,
            max_length=args.ppl_max_length,
            device=args.ppl_device,
        )
        ppl_summary_rows = _ppl_summary_by_factor(ppl_rows)
        _write_tsv(analysis_dir / "ppl_rows.tsv", ppl_rows)
        _write_tsv(analysis_dir / "ppl_by_factor.tsv", ppl_summary_rows)

        metadata["ppl_model"] = ppl_model
        metadata["ppl_batch_size"] = args.ppl_batch_size
        metadata["ppl_max_length"] = args.ppl_max_length
        metadata["ppl_device"] = args.ppl_device
        metadata["ppl_rows_scored"] = len(ppl_rows)
        metadata["files_written"].extend(
            [
                "ppl_rows.tsv",
                "ppl_by_factor.tsv",
            ]
        )

    with open(analysis_dir / "analysis_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote analysis to: {analysis_dir}", flush=True)
    print("Files:", flush=True)
    for item in metadata["files_written"]:
        print(f"  - {analysis_dir / item}", flush=True)


if __name__ == "__main__":
    main()
