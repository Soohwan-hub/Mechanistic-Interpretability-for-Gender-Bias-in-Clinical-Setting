import json
import re

INPUT_FILE = 'train_4000_600_chars.json'
OUTPUT_FILE = 'filtered_dataset.jsonl'

# We use specific medical abbreviations found in MIMIC notes
CLINICAL_KEYWORDS = {
    'Heart Failure': [
        r'\bchf\b',                   # Congestive Heart Failure
        r'\bheart failure\b',
        r'\bhfref\b',                 # Reduced Ejection Fraction
        r'\bhfpef\b',                 # Preserved Ejection Fraction
        r'\bcardiomyopathy\b',
        r'\bejection fraction\b',
        r'\bleft ventricular dysfunction\b',
        r'\bbnp\b'                    # Brain Natriuretic Peptide (marker for HF)
    ],
    'Depression': [
        r'\bdepression\b',
        r'\bdepressive\b',
        r'\bmdd\b',                   # Major Depressive Disorder
        r'\bsuicid',                  # Suicidal, suicide
        r'\bmood disorder\b',
        r'\bipolar\b',                # Often comorbid or distinct, usually relevant
        r'\bzoloft\b', r'\bsertraline\b',
        r'\bprozac\b', r'\bfluoxetine\b',
        r'\blexapro\b', r'\bescitalopram\b'
    ],
    'Asthma': [
        r'\basthma\b',
        r'\basthmatic\b',
        r'\bcopd\b',                  # Chronic Obstructive Pulmonary Disease (often grouped in respiratory studies)
        r'\bwheez',                   # Wheezing, wheeze
        r'\bbronchospasm\b',
        r'\balbuterol\b',
        r'\binhaler\b',
        r'\bnebulizer\b'
    ]
}

# 2. GENDER REMOVAL (The "How")
# Removes: Social gender markers (He, She, Mr, Ms)
# Keeps: Biological covariates (Prostate, Ovarian, Hysterectomy, Pregnant)
GENDER_MARKERS = [
    r"\bhe\b", r"\bshe\b", r"\bhim\b", r"\bher\b", r"\bhers\b", r"\bhis\b",
    r"\bman\b", r"\bwoman\b", r"\bmen\b", r"\bwomen\b",
    r"\bmale\b", r"\bfemale\b", 
    r"\bgentleman\b", r"\blady\b", r"\bguy\b", r"\bgirl\b", r"\boy\b",
    r"\bmr\.\b", r"\bmrs\.\b", r"\bms\.\b", r"\bmr\b", r"\bmrs\b", r"\bms\b",
    r"\bhusband\b", r"\bwife\b", r"\bfather\b", r"\bmother\b", 
    r"\bson\b", r"\bdaughter\b", r"\bsister\b", r"\bbrother\b",
    r"\bgrandmother\b", r"\bgrandfather\b"
]

# Compile Regex for speed
GENDER_REGEX = re.compile("|".join(GENDER_MARKERS), re.IGNORECASE)
REPLACEMENT_TOKEN = "[PATIENT]"


def get_cohort(text):
    """
    Scans text for keywords. Returns the cohort name if found.
    If multiple found, returns 'Comorbid'.
    If none, returns None.
    """
    text_lower = text.lower()
    found = []
    
    for cohort, patterns in CLINICAL_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found.append(cohort)
                break # Found this cohort, move to next
    
    found = list(set(found)) # Remove duplicates
    
    if len(found) == 0:
        return None
    elif len(found) == 1:
        return found[0]
    else:
        return "Comorbid" # You can decide to keep or drop these later

def degender_text(text):
    """
    Replaces gender pronouns with [PATIENT].
    """
    if not text: return ""
    return GENDER_REGEX.sub(REPLACEMENT_TOKEN, text)


def main():
    print(f"Processing {INPUT_FILE}...")
    stats = {"Heart Failure": 0, "Depression": 0, "Asthma": 0, "Comorbid": 0, "Dropped": 0}
    
    with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
        for line in f_in:
            try:
                entry = json.loads(line)
                original_text = entry.get('text', '')
                
                # 1. Identify Cohort
                cohort = get_cohort(original_text)
                
                if cohort:
                    # 2. De-gender
                    clean_text = degender_text(original_text)
                    clean_summary = degender_text(entry.get('summary', ''))
                    
                    # 3. Construct new entry
                    new_entry = {
                        "cohort": cohort,
                        "text": clean_text,         # The bias-neutral input
                        "summary": clean_summary,   # The bias-neutral ground truth
                        "original_text": original_text # Optional: Keep original for "Male vs Female" injection testing later
                    }
                    
                    f_out.write(json.dumps(new_entry) + '\n')
                    stats[cohort] += 1
                else:
                    stats["Dropped"] += 1
                    
            except json.JSONDecodeError:
                continue

    print("\nProcessing Complete!")
    print("Statistics:", json.dumps(stats, indent=2))
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()