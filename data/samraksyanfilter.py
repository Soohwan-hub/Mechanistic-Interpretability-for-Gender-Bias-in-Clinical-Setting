import json
import random
import re

# ============================================================
# 1. CONFIG (Ahsan-style risk-prediction filtering)
# ============================================================

INPUT_FILE = "train.json"
SEED = 42
SAMPLE_SIZE = 1000

# Disease trigger terms: notes must contain at least one trigger.
# These are proxy symptoms, not explicit disease names.
HF_TRIGGER_TERMS = [
    "dyspnea",
    "orthopnea",
    "edema",
    "shortness of breath",
]

DEPRESSION_TRIGGER_TERMS = [
    "anxiety"
]

# Explicit disease mentions to exclude (risk-prediction setup).
HF_EXCLUDE_TERMS = [
    "heart failure",
    "congestive heart failure",
    "chf",
    "hfref",
    "hfpef",
    "acute decompensated heart failure",
    "coronary artery disease",
    "coronary heart disease",
    "ischemic heart disease",
    "cad",
    "mi",
    "myocardial infarction",
    "nstemi",
    "stemi",
    "angina",
    "unstable angina",
    "stable angina",
    "acs",
    "acute coronary syndrome",
    "s/p cabg",
    "cabg",
    "stent",
    "pci",
    "percutaneous coronary intervention"
]

DEPRESSION_EXCLUDE_TERMS = [
    "major depressive disorder",
    "depression",
    "mdd"
]

# Sex-specific conditions to exclude.
SEX_SPECIFIC_TERMS = [
    "pregnancy",
    "pregnant",
    "postpartum",
    "antepartum",
    "peripartum",
    "preeclampsia",
    "pre-eclampsia",
    "eclampsia",
    "gravida",
    "para",
    "lmp",
    "amenorrhea",
    "dysmenorrhea",
    "menstruation",
    "menstrual",
    "ovarian",
    "ovary",
    "uterus",
    "uterine",
    "endometrium",
    "endometrial",
    "endometriosis",
    "hysterectomy",
    "cervical",
    "cervix",
    "fallopian",
    "pcos",
    "fibroid",
    "prostate",
    "prostatic",
    "bph",
    "testicular",
    "testes",
    "orchitis",
    "epididymitis",
    "erectile dysfunction",
    "hypogonadism",
]

# Female-only proxy: require one or more female-coded markers in BHC text.
REQUIRE_FEMALE_NOTES = True
FEMALE_MARKER_REGEX = re.compile(
    r"\b(she|her|hers|female|woman|women|lady|ms\.?|mrs\.?)\b",
    re.IGNORECASE,
)


def contains_any(text, terms):
    text = text.lower()
    return any(re.search(rf"\b{re.escape(t)}\b", text) for t in terms)


def count_matches(text, terms):
    text = text.lower()
    return sum(1 for t in terms if re.search(rf"\b{re.escape(t)}\b", text))


def has_female_marker(text):
    return bool(FEMALE_MARKER_REGEX.search(text))


# ============================================================
# 2. LOAD DATA
# ============================================================

with open(INPUT_FILE, "r") as f:
    records = [json.loads(line) for line in f if line.strip()]

print("Total records loaded:", len(records))


# ============================================================
# 3. FILTER (Ahsan-style)
# ============================================================

hf_candidates = []
dep_candidates = []

for entry in records:
    text = entry.get("text", "")
    summary = entry.get("summary", "")

    if not text:
        continue

    if REQUIRE_FEMALE_NOTES and not has_female_marker(text):
        continue

    if contains_any(text, SEX_SPECIFIC_TERMS):
        continue

    hf_trigger_count = count_matches(text, HF_TRIGGER_TERMS)

    if hf_trigger_count >= 1 and not contains_any(text, HF_EXCLUDE_TERMS):
        hf_candidates.append({
            "cohort": "Heart Failure",
            "text": text,
            "summary": summary,
            "original_text": text
        })

    if contains_any(text, DEPRESSION_TRIGGER_TERMS) and not contains_any(text, DEPRESSION_EXCLUDE_TERMS):
        dep_candidates.append({
            "cohort": "Depression",
            "text": text,
            "summary": summary,
            "original_text": text
        })

print("HF candidates (pre-sample):", len(hf_candidates))
print("Depression candidates (pre-sample):", len(dep_candidates))

with open("hf_candidates.jsonl", "w") as f:
    for row in hf_candidates:
        f.write(json.dumps(row) + "\n")

with open("depression_candidates.jsonl", "w") as f:
    for row in dep_candidates:
        f.write(json.dumps(row) + "\n")


# ============================================================
# 4. SAMPLE AND SAVE
# ============================================================

rng = random.Random(SEED)
rng.shuffle(hf_candidates)
rng.shuffle(dep_candidates)

hf_sample = hf_candidates[:SAMPLE_SIZE]
dep_sample = dep_candidates[:SAMPLE_SIZE]

with open("hf_risk_candidates.jsonl", "w") as f:
    for row in hf_sample:
        f.write(json.dumps(row) + "\n")

with open("depression_risk_candidates.jsonl", "w") as f:
    for row in dep_sample:
        f.write(json.dumps(row) + "\n")

print("Saved HF sample:", len(hf_sample))
print("Saved Depression sample:", len(dep_sample))
