import json
import re

INPUT_FILE = "train_4000_600_chars.json"

female_terms = ["she", "her", "hers", "mrs", "ms", "female", "woman", "women"]
male_terms = ["he", "him", "his", "mr", "male", "man", "men"]

female_re = re.compile(r"\b(" + "|".join(map(re.escape, female_terms)) + r")\b", re.I)
male_re = re.compile(r"\b(" + "|".join(map(re.escape, male_terms)) + r")\b", re.I)


def is_female_only(text):
    return bool(female_re.search(text)) and not bool(male_re.search(text))


def main():
    total = 0
    female_only = 0
    with_any_female = 0

    with open(INPUT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            text = record.get("text", "") or ""

            if female_re.search(text):
                with_any_female += 1
            if is_female_only(text):
                female_only += 1

    print("Total notes:", total)
    print("With any female marker:", with_any_female)
    print("Female-only (female markers and no male markers):", female_only)


if __name__ == "__main__":
    main()
