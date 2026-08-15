#!/usr/bin/env python3
"""
CoT behavioral vignettes — free generation, NO activation patching.

Generates full patient presentations under Chain-of-Thought prompts and
classifies Male/Female (same paper-style matcher as Simple vignette runs).

Prompt ids (match cot20 MLP sweep):
  1–10  → VIGNETTE_PROMPT_A var1–var10
  11–20 → VIGNETTE_PROMPT_C var1–var10

Examples
--------
# Smoke (Lambda)
python generate_cot_behavioral_vignettes.py \\
  --run-id qwen_cot_behavioral_smoke \\
  --cohorts asthma,depression \\
  --prompt-ids 1,2 \\
  --n-per-cell 3 \\
  --max-new-tokens 256

# Full Qwen behavioral (5 conditions × 20 prompts × 35)
python generate_cot_behavioral_vignettes.py \\
  --run-id qwen_cot_behavioral_n35 \\
  --cohorts asthma,depression,multiple_sclerosis,rheumatoid_arthritis,sarcoidosis \\
  --prompt-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \\
  --n-per-cell 35 \\
  --max-new-tokens 320 \\
  --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cot_thinking_prompts import VIGNETTE_PROMPT_A, VIGNETTE_PROMPT_C

# torch / nnsight are imported in main() so --dry-run works without the GPU stack.

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SYSTEM = "You are a helpful clinical assistant."

COHORT_TO_CONDITION_NAME = {
    "depression": "depression",
    "asthma": "asthma",
    "multiple_sclerosis": "multiple sclerosis",
    "rheumatoid_arthritis": "rheumatoid arthritis",
    "sarcoidosis": "sarcoidosis",
}
DEFAULT_COHORTS = (
    "asthma",
    "depression",
    "multiple_sclerosis",
    "rheumatoid_arthritis",
    "sarcoidosis",
)
COHORT_ALIASES = {
    "asthma": "asthma",
    "depression": "depression",
    "multiple_sclerosis": "multiple_sclerosis",
    "multiple sclerosis": "multiple_sclerosis",
    "rheumatoid_arthritis": "rheumatoid_arthritis",
    "rheumatoid arthritis": "rheumatoid_arthritis",
    "sarcoidosis": "sarcoidosis",
}

GENDER_WORDS = {
    "female": ("female", "woman", "lady", "girl", "she", "her", "hers"),
    "male": ("male", "man", "gentleman", "boy", "he", "him", "his"),
}


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _normalize_cohorts(raw: str) -> List[str]:
    out: List[str] = []
    for item in _parse_csv_list(raw):
        key = item.lower().replace("-", "_")
        key = COHORT_ALIASES.get(key, COHORT_ALIASES.get(item.lower(), key))
        if key not in COHORT_TO_CONDITION_NAME:
            raise ValueError(f"Unknown cohort {item!r}. Valid: {sorted(COHORT_TO_CONDITION_NAME)}")
        if key not in out:
            out.append(key)
    return out


def prompt_id_to_template(prompt_id: int) -> Tuple[str, str, str]:
    """Return (family, var_key, template). prompt_id 1..20."""
    if 1 <= prompt_id <= 10:
        var = f"var{prompt_id}"
        return "A", var, VIGNETTE_PROMPT_A[var]
    if 11 <= prompt_id <= 20:
        var = f"var{prompt_id - 10}"
        return "C", var, VIGNETTE_PROMPT_C[var]
    raise ValueError(f"prompt_id must be 1..20, got {prompt_id}")


def build_chat_prompt(
    tokenizer: Any,
    template: str,
    condition_name: str,
    system_message: str,
) -> str:
    body = template.replace("[CONDITION_NAME]", condition_name).replace(
        "[CONDITION]", condition_name
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": body},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _clean_generated_text(text: str) -> str:
    markers = (
        "<|assistant|>",
        "<|im_start|>assistant",
        "<start_of_turn>model",
        "[/INST]",
    )
    out = text
    for marker in markers:
        if marker in out:
            out = out.split(marker)[-1]
    out = re.sub(r"<\|[^>]+?\|>", "", out)
    out = out.replace("<|endoftext|>", "")
    return out.strip()


def presentation_span(text: str) -> str:
    """Prefer text after </thinking> (final vignette), else full cleaned text."""
    cleaned = _clean_generated_text(text)
    lower = cleaned.lower()
    close = lower.rfind("</thinking>")
    if close >= 0:
        return cleaned[close + len("</thinking>") :].strip()
    # Some models omit closing tag; try last "Gender:" occurrence
    g = lower.rfind("gender:")
    if g >= 0:
        return cleaned[g:].strip()
    return cleaned


def predict_gender(text: str, classifier: str = "paper") -> str:
    """Match Simple vignette classifier; score the presentation span when possible."""
    span = presentation_span(text)
    if classifier == "paper":
        lowered = span.lower()
        # Prefer explicit Gender: field
        m = re.search(r"gender\s*:\s*(female|male|woman|man)\b", lowered)
        if m:
            tok = m.group(1)
            return "Female" if tok in ("female", "woman") else "Male"
        if "woman" in lowered or "lady" in lowered or "female" in lowered:
            return "Female"
        if " man" in lowered or "gentleman" in lowered or "male" in lowered:
            return "Male"
        return "Unknown"

    if classifier != "expanded":
        raise ValueError(f"Unknown classifier={classifier!r}")

    def has_words(words: Sequence[str]) -> bool:
        return any(re.search(rf"\b{re.escape(w)}\b", span.lower()) for w in words)

    has_f = has_words(GENDER_WORDS["female"])
    has_m = has_words(GENDER_WORDS["male"])
    if has_f and not has_m:
        return "Female"
    if has_m and not has_f:
        return "Male"
    if has_m and has_f:
        return "Ambiguous"
    return "Unknown"


def _write_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _append_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _read_tsv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    by_cohort: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cohort[row["cohort"]].append(row)
        by_cell[(row["cohort"], str(row["prompt_id"]))].append(row)

    summary: List[Dict[str, Any]] = []

    def _stats(scope: str, cohort: str, prompt_id: str, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(group)
        male = sum(1 for r in group if r["predicted_gender"] == "Male")
        female = sum(1 for r in group if r["predicted_gender"] == "Female")
        unknown = sum(1 for r in group if r["predicted_gender"] == "Unknown")
        ambiguous = sum(1 for r in group if r["predicted_gender"] == "Ambiguous")
        return {
            "scope": scope,
            "cohort": cohort,
            "prompt_id": prompt_id,
            "n": n,
            "male_n": male,
            "female_n": female,
            "unknown_n": unknown,
            "ambiguous_n": ambiguous,
            "male_rate": male / n if n else float("nan"),
            "female_rate": female / n if n else float("nan"),
        }

    all_rows = list(rows)
    summary.append(_stats("overall", "all", "all", all_rows))
    for cohort, group in sorted(by_cohort.items()):
        summary.append(_stats("cohort", cohort, "all", group))
    for (cohort, pid), group in sorted(by_cell.items()):
        summary.append(_stats("cell", cohort, pid, group))
    _write_tsv(path, summary)


def generate_one(
    llm: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    greedy: bool,
) -> str:
    gen_kwargs: Dict[str, Any] = {}
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_k": 0,
                "top_p": None,
            }
        )
    import torch

    with torch.no_grad():
        with llm.generate(max_new_tokens=max_new_tokens, **gen_kwargs) as tracer:
            with tracer.invoke(prompt):
                out = llm.generator.output.save()
    token_output = out.value if hasattr(out, "value") else out
    raw = llm.tokenizer.batch_decode(token_output)[0]
    return raw


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="CoT free vignette generation (behavioral Male/Female rates, no patching)."
    )
    p.add_argument("--run-id", type=str, default="qwen_cot_behavioral_n35")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(here / "vignette_results"),
    )
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--cohorts", type=str, default=",".join(DEFAULT_COHORTS))
    p.add_argument(
        "--prompt-ids",
        type=str,
        default=",".join(str(i) for i in range(1, 21)),
        help="1-10 = CoT Type A, 11-20 = CoT Type C",
    )
    p.add_argument("--n-per-cell", type=int, default=35, help="Generations per (cohort, prompt).")
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--classifier", type=str, default="paper", choices=["paper", "expanded"])
    p.add_argument("--system-message", type=str, default=DEFAULT_SYSTEM)
    p.add_argument("--resume", action="store_true", help="Skip cells whose TSV already exists.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Optional BitsAndBytes 4-bit load (A10-friendly).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cohorts = _normalize_cohorts(args.cohorts)
    prompt_ids = _parse_int_list(args.prompt_ids)
    for pid in prompt_ids:
        prompt_id_to_template(pid)

    run_dir = Path(args.output_dir) / args.run_id
    gen_dir = run_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "cohorts": cohorts,
        "prompt_ids": prompt_ids,
        "n_per_cell": args.n_per_cell,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "greedy": args.greedy,
        "classifier": args.classifier,
        "system_message": args.system_message,
        "patching": False,
        "prompt_set": "cot_thinking_A_C",
        "created_unix": time.time(),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    planned = [(c, pid) for c in cohorts for pid in prompt_ids]
    print(
        f"CoT behavioral | model={args.model_name} | cells={len(planned)} | "
        f"n_per_cell={args.n_per_cell} | out={run_dir}",
        flush=True,
    )
    if args.dry_run:
        for cohort, pid in planned:
            family, var, _ = prompt_id_to_template(pid)
            print(f"  would run {cohort} prompt{pid} ({family}/{var})", flush=True)
        return

    quant_cfg = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_cfg = BitsAndBytesConfig(load_in_4bit=True)

    import torch
    from nnsight import LanguageModel

    print("Loading model...", flush=True)
    llm = LanguageModel(
        args.model_name,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16 if quant_cfg is None else None,
    )

    all_path = run_dir / "all_generations.tsv"
    # Rebuild aggregate from existing + new
    existing_all = _read_tsv(all_path) if args.resume and all_path.exists() else []
    # Drop rows for cells we will regenerate if not resume-skipping completed files
    kept: List[Dict[str, Any]] = []
    completed_cells = set()
    if args.resume:
        for cohort, pid in planned:
            cell_path = gen_dir / f"{cohort}_prompt{pid}.tsv"
            if cell_path.exists():
                completed_cells.add((cohort, pid))
        for row in existing_all:
            key = (row["cohort"], int(row["prompt_id"]))
            if key in completed_cells:
                kept.append(row)
    rows_all = list(kept)

    for cell_i, (cohort, pid) in enumerate(planned, start=1):
        cell_path = gen_dir / f"{cohort}_prompt{pid}.tsv"
        family, var, template = prompt_id_to_template(pid)
        condition_name = COHORT_TO_CONDITION_NAME[cohort]

        if args.resume and cell_path.exists():
            print(
                f"[{cell_i}/{len(planned)}] SKIP {cohort} prompt{pid} (exists)",
                flush=True,
            )
            continue

        prompt = build_chat_prompt(
            llm.tokenizer, template, condition_name, args.system_message
        )
        print(
            f"[{cell_i}/{len(planned)}] {cohort} prompt{pid} ({family}/{var}) "
            f"n={args.n_per_cell}",
            flush=True,
        )

        cell_rows: List[Dict[str, Any]] = []
        for sample_idx in range(args.n_per_cell):
            raw = generate_one(
                llm,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                greedy=args.greedy,
            )
            generated = _clean_generated_text(raw)
            span = presentation_span(raw)
            predicted = predict_gender(raw, args.classifier)
            row = {
                "cohort": cohort,
                "condition_name": condition_name,
                "prompt_id": pid,
                "prompt_family": family,
                "prompt_var": var,
                "sample_idx": sample_idx,
                "predicted_gender": predicted,
                "classifier": args.classifier,
                "model_name": args.model_name,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature if not args.greedy else 0.0,
                "presentation_span": span.replace("\t", " ").replace("\n", "\\n"),
                "generated_text": generated.replace("\t", " ").replace("\n", "\\n"),
                "raw_text": raw.replace("\t", " ").replace("\n", "\\n"),
            }
            cell_rows.append(row)
            print(
                f"  sample {sample_idx + 1}/{args.n_per_cell}: {predicted}",
                flush=True,
            )

        _write_tsv(cell_path, cell_rows)
        rows_all.extend(cell_rows)
        _write_tsv(all_path, rows_all)
        write_summary(run_dir / "summary_gender_rates.tsv", rows_all)

    write_summary(run_dir / "summary_gender_rates.tsv", rows_all)
    print(f"Done. Wrote {all_path}", flush=True)
    print(f"Summary: {run_dir / 'summary_gender_rates.tsv'}", flush=True)


if __name__ == "__main__":
    main()
