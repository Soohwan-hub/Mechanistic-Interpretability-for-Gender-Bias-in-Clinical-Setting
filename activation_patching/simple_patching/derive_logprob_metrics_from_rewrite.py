#!/usr/bin/env python3
"""Derive log-prob metrics from saved rewrite-score artifacts.

This is an offline utility: it does not import or load any model. It reconstructs
patched target probabilities from rewrite scores and the saved corrupted baseline
probability, then writes a derived run folder with extra score matrices.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


SCORE_KEYS = ("rewrite_scores", "logprob_scores", "logprob_delta_scores")
EPS = 1e-45


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _atomic_write_pickle(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def derive_artifact(src_path: Path, dst_path: Path) -> None:
    with open(src_path, "rb") as f:
        payload = pickle.load(f)

    if "rewrite_scores" not in payload:
        raise ValueError(f"{src_path} does not contain rewrite_scores")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{src_path} does not contain metadata")
    if "corrupted_prob" not in metadata:
        raise ValueError(f"{src_path} metadata does not contain corrupted_prob")

    rewrite_scores = np.asarray(payload["rewrite_scores"], dtype=float)
    corrupted_prob = float(metadata["corrupted_prob"])

    denom = 1.0 - corrupted_prob + 1e-8
    patched_prob = rewrite_scores * denom + corrupted_prob
    patched_prob = np.clip(patched_prob, EPS, 1.0)
    clipped_corrupted_prob = max(corrupted_prob, EPS)

    logprob_scores = np.log(patched_prob)
    logprob_delta_scores = logprob_scores - np.log(clipped_corrupted_prob)

    derived = dict(payload)
    derived["rewrite_scores"] = rewrite_scores
    derived["logprob_scores"] = logprob_scores
    derived["logprob_delta_scores"] = logprob_delta_scores

    derived_metadata = dict(metadata)
    derived_metadata["score_keys"] = list(SCORE_KEYS)
    derived_metadata["corrupted_logprob"] = float(np.log(clipped_corrupted_prob))
    derived_metadata["derived_from"] = str(src_path)
    derived_metadata["derivation"] = (
        "logprob_scores and logprob_delta_scores derived offline from rewrite_scores "
        "and metadata['corrupted_prob']; no model rerun."
    )
    derived["metadata"] = derived_metadata

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_pickle(dst_path, derived)


def build_progress(src_run: Path, dst_run: Path, artifact_count: int) -> None:
    src_progress_path = src_run / "progress.json"
    if src_progress_path.exists():
        with open(src_progress_path, encoding="utf-8") as f:
            progress = json.load(f)
    else:
        completed = []
        for p in sorted((src_run / "artifacts").glob("*.pkl")):
            stem = p.stem
            if "_prompt" not in stem:
                continue
            cohort, prompt_id = stem.rsplit("_prompt", 1)
            completed.append(f"{cohort}:prompt{prompt_id}")
        progress = {"completed": completed, "failed": {}, "config_hash": ""}

    progress["model_name"] = "allenai/OLMo-7B-0724-Instruct-hf"
    progress["derived_from"] = str(src_run)
    progress["derivation"] = (
        "Derived logprob_scores and logprob_delta_scores from rewrite_scores "
        "using saved corrupted_prob; male_female_logit_delta_scores not derived."
    )
    progress["derived_artifact_count"] = artifact_count
    progress["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_write_json(dst_run / "progress.json", progress)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive logprob metrics from rewrite-only patching artifacts."
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("activation_patching/simple_patching/olmo31_rewrite_only"),
        help="Run directory containing rewrite-only artifacts.",
    )
    parser.add_argument(
        "--dest-run",
        type=Path,
        default=Path("activation_patching/simple_patching/olmo31_derived_logprob_metrics"),
        help="Destination run directory for derived artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination run directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_run = args.source_run
    dst_run = args.dest_run
    src_artifacts = src_run / "artifacts"
    dst_artifacts = dst_run / "artifacts"

    if not src_artifacts.exists():
        raise FileNotFoundError(f"Source artifacts directory not found: {src_artifacts}")
    if dst_run.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_run}. Pass --overwrite to replace it."
            )
        shutil.rmtree(dst_run)

    files = sorted(src_artifacts.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl artifacts found in {src_artifacts}")

    for src_path in files:
        derive_artifact(src_path, dst_artifacts / src_path.name)

    build_progress(src_run, dst_run, len(files))
    print(f"Derived {len(files)} artifacts into {dst_run}")


if __name__ == "__main__":
    main()
