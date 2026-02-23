import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

from config import CONFIG
from prompts import build_all_active_prompt_triplets


# ============================================================
# 1. LOAD + EXTRACT HELPERS
# ============================================================

def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_case_text(row: Dict) -> str:
    # Prefer filtered text; fallback to original text if needed.
    text = (row.get("text") or "").strip()
    if text:
        return text
    return (row.get("original_text") or "").strip()


def _sample_rows(rows: List[Dict], max_cases: int, seed: int) -> Iterable[Dict]:
    if len(rows) <= max_cases:
        return rows

    rng = random.Random(seed)
    picked = rows[:]
    rng.shuffle(picked)
    return picked[:max_cases]


def _extract_condition(row: Dict) -> str:
    # Pull condition from configured fields in priority order.
    for field in CONFIG.condition_fields:
        value = (row.get(field) or "").strip()
        if value:
            return value
    return "the given condition"


def _build_case_variants(case_text: str) -> Dict[str, str]:
    # Task 3 needs three case versions: neutral, male, and female.
    base_text = case_text.strip()
    return {
        "neutral": f"{CONFIG.neutral_case_prefix} {base_text}",
        "male": f"{CONFIG.male_case_prefix} {base_text}",
        "female": f"{CONFIG.female_case_prefix} {base_text}",
    }


# ============================================================
# 2. BUILD AND SAVE PROMPT SETS
# ============================================================

def _build_output_row(row: Dict, case_id: int) -> Dict:
    case_text = _extract_case_text(row)
    condition = _extract_condition(row)
    case_variants = _build_case_variants(case_text)
    prompt_triplets = build_all_active_prompt_triplets(
        case_variants=case_variants,
        condition=condition,
        active_prompt_ids=CONFIG.active_prompt_ids,
    )

    return {
        "case_id": case_id,
        "condition": condition,
        "cohort": row.get("cohort"),
        "source_summary": row.get("summary", ""),
        "source_text": case_text,
        "case_variants": case_variants,
        "prompt_triplets": prompt_triplets,
    }


def build_case_triplets() -> int:
    if not CONFIG.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {CONFIG.input_file}")

    rows = _read_jsonl(CONFIG.input_file)
    sampled_rows = list(_sample_rows(rows, CONFIG.max_cases, CONFIG.seed))

    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with CONFIG.triplets_file.open("w") as f:
        for case_id, row in enumerate(sampled_rows):
            case_text = _extract_case_text(row)
            if not case_text:
                continue

            # Save format: prompt_id -> {neutral, male, female}.
            output_row = _build_output_row(row, case_id=case_id)
            f.write(json.dumps(output_row) + "\n")
            count += 1

    return count


def main() -> None:
    count = build_case_triplets()
    print(f"Saved {count} case triplets -> {CONFIG.triplets_file}")


if __name__ == "__main__":
    main()
