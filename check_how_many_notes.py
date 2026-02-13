import json
import re

INPUT_FILE = "train_4000_600_chars.json"

HF_TRIGGER_TERMS = [
    "shortness of breath",
    "dyspnea",
    "orthopnea",
    "paroxysmal nocturnal dyspnea",
    "lower extremity edema",
    "pitting edema",
    "volume overload"
]

DEPRESSION_TRIGGER_TERMS = [
    "anxiety",
    "panic",
    "anxious",
    "sadness",
    "low mood",
    "anhedonia",
    "hopeless",
    "suicidal"
]

HF_EXCLUDE_TERMS = [
    "heart failure",
    "congestive heart failure",
    "chf",
    "hfref",
    "hfpef",
    "acute decompensated heart failure"
]

DEPRESSION_EXCLUDE_TERMS = [
    "major depressive disorder",
    "depression",
    "mdd"
]

SEX_SPECIFIC_TERMS = [
    "pregnancy",
    "pregnant",
    "postpartum",
    "preeclampsia",
    "eclampsia",
    "ovarian",
    "uterus",
    "hysterectomy",
    "cervical",
    "prostate",
    "testicular",
    "erectile dysfunction"
]


def contains_any(text, terms):
    text = text.lower()
    return any(re.search(rf"\b{re.escape(t)}\b", text) for t in terms)


def main():
    hf_count = 0
    dep_count = 0

    with open(INPUT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            text = entry.get("text", "")
            if not text:
                continue
            if contains_any(text, SEX_SPECIFIC_TERMS):
                continue
            if contains_any(text, HF_TRIGGER_TERMS) and not contains_any(text, HF_EXCLUDE_TERMS):
                hf_count += 1
            if contains_any(text, DEPRESSION_TRIGGER_TERMS) and not contains_any(text, DEPRESSION_EXCLUDE_TERMS):
                dep_count += 1

    print("HF candidates (pre-sample):", hf_count)
    print("Depression candidates (pre-sample):", dep_count)


if __name__ == "__main__":
    main()
