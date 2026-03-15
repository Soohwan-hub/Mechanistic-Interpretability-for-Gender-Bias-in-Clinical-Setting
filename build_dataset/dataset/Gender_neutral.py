import json
import re


# Input sampled files
INPUT_FILES = [
    "hf_risk_candidates.jsonl",
    "depression_risk_candidates.jsonl",
]

# Output suffix for neutralized datasets
OUTPUT_SUFFIX = "_gender_neutral.jsonl"

REPLACEMENT_TOKEN = "patient"

FEMALE_GENERAL_REGEX = re.compile(
    r"\b(female|woman|women|lady|ms\.?|mrs\.?)\b",
    re.IGNORECASE,
    
)
SHE_REGEX = re.compile(r"\bshe\b", re.IGNORECASE)
HERS_REGEX = re.compile(r"\bhers\b", re.IGNORECASE)
# Handle possessive phrase first, e.g. "her pain" -> "patient's pain".
HER_POSSESSIVE_REGEX = re.compile(r"\bher\s+([A-Za-z][A-Za-z0-9_-]*)", re.IGNORECASE)
HER_OBJECT_REGEX = re.compile(r"\bher\b", re.IGNORECASE)


def neutralize_text(text):
    if not text:
        return ""
    text = FEMALE_GENERAL_REGEX.sub(REPLACEMENT_TOKEN, text)
    text = SHE_REGEX.sub(REPLACEMENT_TOKEN, text)
    text = HERS_REGEX.sub(f"{REPLACEMENT_TOKEN}'s", text)
    text = HER_POSSESSIVE_REGEX.sub(rf"{REPLACEMENT_TOKEN}'s \1", text)
    text = HER_OBJECT_REGEX.sub(REPLACEMENT_TOKEN, text)
    return text


def process_file(input_path):
    output_path = input_path.replace(".jsonl", OUTPUT_SUFFIX)
    count = 0

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            row["text"] = neutralize_text(row.get("text", ""))
            row["summary"] = neutralize_text(row.get("summary", ""))
            f_out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Saved {count} rows -> {output_path}")


def main():
    for input_path in INPUT_FILES:
        process_file(input_path)


if __name__ == "__main__":
    main()
