import json
import re

# ============================================================
# 1. NEGATION HANDLING
# ============================================================

NEGATIONS = [
    "no",
    "denies",
    "denied",
    "negative",
    "negative for",
    "not",
    "absence of",
    "free of",
    "lack of",
    "absent",
    "without",
    "rule out",
    "ruled out",
    "r/o",
    "unlikely",
    "not consistent with",
    "no evidence of",
    "without evidence of"
]

def not_negated(text, term, window=50):
    """
    Returns True if `term` appears in `text` WITHOUT a negation
    within `window` characters to the left.
    """
    text = text.lower()
    term = term.lower()

    for match in re.finditer(re.escape(term), text):
        start = match.start()
        left_context = text[max(0, start - window):start]

        if not any(
            re.search(rf"\b{re.escape(neg)}\b", left_context)
            for neg in NEGATIONS
        ):
            return True

    return False


# ============================================================
# 2. HEART FAILURE DEFINITIONS
# ============================================================

HF_DIAGNOSIS_TERMS = [
    "heart failure",
    "congestive heart failure",
    "chf",
    "hfref",
    "hfpef",
    "acute decompensated heart failure"
]

HF_SUPPORT_TERMS = [
    "ejection fraction",
    "pulmonary edema",
    "volume overload",
    "fluid overload",
    "cardiomyopathy"
]

HF_MEDS = [
    "furosemide",
    "lasix",
    "bumetanide",
    "torsemide",
    "spironolactone"
]

def hf_score(text):
    score = 0
    if any(not_negated(text, t) for t in HF_DIAGNOSIS_TERMS):
        score += 2
    if any(not_negated(text, t) for t in HF_SUPPORT_TERMS):
        score += 1
    if any(not_negated(text, t) for t in HF_MEDS):
        score += 1
    return score


# ============================================================
# 3. ASTHMA DEFINITIONS
# ============================================================

ASTHMA_TERMS = [
    "asthma",
    "reactive airway disease",
    "exercise induced asthma"
]

ASTHMA_SUPPORT = [
    "wheezing",
    "bronchospasm"
]

ASTHMA_MEDS = [
    "albuterol",
    "advair",
    "symbicort"
]

def asthma_score(text):
    score = 0
    if any(not_negated(text, t) for t in ASTHMA_TERMS):
        score += 2
    if any(not_negated(text, t) for t in ASTHMA_SUPPORT):
        score += 1
    if any(not_negated(text, t) for t in ASTHMA_MEDS):
        score += 1
    return score


# ============================================================
# 4. DEPRESSION DEFINITIONS
# ============================================================

DEPRESSION_TERMS = [
    "major depressive disorder",
    "depression",
    "mdd"
]

DEPRESSION_SYMPTOMS = [
    "depressed mood",
    "anhedonia",
    "hopelessness",
    "suicidal ideation"
]

ANTIDEPRESSANTS = [
    "sertraline",
    "fluoxetine",
    "citalopram",
    "escitalopram",
    "venlafaxine",
    "bupropion",
    "mirtazapine"
]

def depression_score(text):
    score = 0
    if any(not_negated(text, t) for t in DEPRESSION_TERMS):
        score += 2
    if any(not_negated(text, t) for t in DEPRESSION_SYMPTOMS):
        score += 1
    if any(not_negated(text, t) for t in ANTIDEPRESSANTS):
        score += 1
    return score


# ============================================================
# 5. LOAD DATA
# ============================================================

INPUT_FILE = "train_4000_600_chars.json"

with open(INPUT_FILE, "r") as f:
    records = [json.loads(line) for line in f if line.strip()]

print("Total records loaded:", len(records))


# ============================================================
# 6. SCORE ALL RECORDS
# ============================================================

hf_cases = []
asthma_cases = []
depression_cases = []

for entry in records:
    text = entry.get("text", "")

    hf_cases.append((hf_score(text), entry))
    asthma_cases.append((asthma_score(text), entry))
    depression_cases.append((depression_score(text), entry))


# ============================================================
# 7. SORT BY CONFIDENCE
# ============================================================

hf_cases.sort(key=lambda x: x[0], reverse=True)
asthma_cases.sort(key=lambda x: x[0], reverse=True)
depression_cases.sort(key=lambda x: x[0], reverse=True)


# ============================================================
# 8. SELECT TOP 100 (THRESHOLDS)
# ============================================================

HF_THRESHOLD = 2
ASTHMA_THRESHOLD = 2
DEPRESSION_THRESHOLD = 2

HF_TOP_100 = [e for s, e in hf_cases if s >= HF_THRESHOLD][:100]
ASTHMA_TOP_100 = [e for s, e in asthma_cases if s >= ASTHMA_THRESHOLD][:100]
DEPRESSION_TOP_100 = [e for s, e in depression_cases if s >= DEPRESSION_THRESHOLD][:100]

print("Heart Failure shortlisted:", len(HF_TOP_100))
print("Asthma shortlisted:", len(ASTHMA_TOP_100))
print("Depression shortlisted:", len(DEPRESSION_TOP_100))


# ============================================================
# 9. SAVE OUTPUTS
# ============================================================

with open("hf_top_100.json", "w") as f:
    json.dump(HF_TOP_100, f, indent=2)

with open("asthma_top_100.json", "w") as f:
    json.dump(ASTHMA_TOP_100, f, indent=2)

with open("depression_top_100.json", "w") as f:
    json.dump(DEPRESSION_TOP_100, f, indent=2)

print("Saved high-confidence cohorts to disk.")


# ============================================================
# 10. PRINT SAMPLE CASES FOR MANUAL REVIEW
# ============================================================

def print_sample_cases(name, cases, n=5):
    print("\n" + "=" * 80)
    print(f"SAMPLE {name.upper()} CASES (showing {n})")
    print("=" * 80)

    for i, entry in enumerate(cases[:n], 1):
        text = entry.get("text", "")
        summary = entry.get("summary", "")

        print(f"\n--- {name} CASE {i} ---")
        print("TEXT:")
        print(text[:1000])
        print("\nSUMMARY:")
        print(summary[:500])


print_sample_cases("Heart Failure", HF_TOP_100)
print_sample_cases("Asthma", ASTHMA_TOP_100)
print_sample_cases("Depression", DEPRESSION_TOP_100)