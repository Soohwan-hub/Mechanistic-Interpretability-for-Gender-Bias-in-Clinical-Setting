"""
Redact explicit gender cues from clinical note text.

Primary use: produce a **new** CSV with the same columns as the input
(e.g. ``unique_patient_gender_cohort.csv``) but a scrubbed ``text`` column, so
downstream probes stress **implicit** correlates (pronouns, clinical phrasing)
rather than header lines like ``Sex:   F``.

The CLI **always writes a separate file** (default name
``<input_stem>_gender_scrubbed.csv``). It **refuses** to use the input path as
output so the original data is not overwritten.

Requires **pandas** (see ``requirements-scrub.txt``). On Windows, install into a **short**
venv path (e.g. ``C:\\venvs\\lp``) to avoid ``WinError 206`` when pip unpacks numpy
under a very long repo path.

Default scrubbing is **conservative** (structured fields only). Stronger modes
can remove pronouns, familial terms, or ``male``/``female`` tokens anywhere—those are easy to
misuse (they delete clinically meaningful phrases like “male breast cancer”).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd


def _match_case(src: str, repl: str) -> str:
    """Return ``repl`` with casing style adapted to ``src``."""
    if src.isupper():
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl
@dataclass
class GenderCueScrubConfig:
    """What to remove or replace. All replacements use ``placeholder`` unless noted."""

    placeholder: str = "[REDACTED]"
    # Structured header / form fields (recommended baseline)
    scrub_sex_colon_line: bool = True
    scrub_gender_colon_line: bool = True
    # Broader “explicit label” phrases (still mostly administrative)
    scrub_assigned_sex_at_birth_line: bool = True
    # Inline “Sex:” / “Gender:” anywhere on a line (not only start-of-line)
    scrub_inline_sex_or_gender_field: bool = True
    # Strong / optional (can distort clinical meaning)
    scrub_he_she_pronouns: bool = False
    scrub_familial_terms: bool = False
    scrub_male_female_tokens: bool = False
    scrub_short_sex_markers: bool = False
    short_sex_marker_replacement: str = "person"
    scrub_sex_specific_anatomy_terms: bool = False
    scrub_repro_medication_marker_terms: bool = False
    scrub_gendered_personal_items: bool = False
    gendered_personal_item_tampon_replacement: str = "absorbent products"
    gendered_personal_item_bra_replacement: str = "supportive garment"
    # Extra regexes applied after built-ins; each (pattern, repl) repl can be str or callable
    extra_subs: list[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]] = field(
        default_factory=list
    )


def scrub_clinical_gender_cues(text: str, cfg: GenderCueScrubConfig | None = None) -> str:
    """
    Return ``text`` with configured gender cues redacted.

    Built-in patterns target typical MIMIC-style discharge headers, e.g.
    a line ``Sex:   F`` → ``Sex: [REDACTED]``.
    """
    if cfg is None:
        cfg = GenderCueScrubConfig()
    out = text
    ph = cfg.placeholder

    if cfg.scrub_sex_colon_line:
        # Line optional leading space; capture Sex / Legal sex variants
        out = re.sub(
            r"(?im)^\s*((?:Legal\s+)?Sex:)\s*[^\n]*",
            lambda m: f"{m.group(1)} {ph}",
            out,
        )
    if cfg.scrub_gender_colon_line:
        out = re.sub(
            r"(?im)^\s*(Gender(?:\s+Identity)?):\s*[^\n]*",
            lambda m: f"{m.group(1)}: {ph}",
            out,
        )
    if cfg.scrub_assigned_sex_at_birth_line:
        out = re.sub(
            r"(?im)^\s*((?:Sex\s+)?(?:Assigned|Designated)\s+Sex\s+at\s+Birth:)\s*[^\n]*",
            lambda m: f"{m.group(1)} {ph}",
            out,
        )
    if cfg.scrub_inline_sex_or_gender_field:
        # Any remaining "Sex:" not at line start (rare in discharges)
        out = re.sub(r"(?i)\b(Sex:)\s*[^\n]+", r"\1 " + ph, out)
        out = re.sub(r"(?i)\b(Gender(?:\s+Identity)?)\s*:\s*[^\n]+", r"\1: " + ph, out)

    if cfg.scrub_he_she_pronouns:
        # Intentionally does NOT touch they/them/their (gender-neutral and structurally important).
        # Map sexed pronouns to patient-form phrases to avoid malformed subject-verb agreement.
        def _pron(m: re.Match[str]) -> str:
            t = m.group(0).lower().rstrip(".")
            if t in {"he", "she", "him", "her", "himself", "herself"}:
                return _match_case(m.group(0), "the patient")
            if t in {"his", "hers"}:
                return _match_case(m.group(0), "the patient's")
            if t in {"priest", "monk", "businessman", "businessmen"}:
                return _match_case(m.group(0), "patient")
            # Keep non-pronoun explicit gender terms/honorifics fully redacted.
            return ph

        out = re.sub(
            r"\b("
            r"mr|mrs|ms|miss|mx|"
            r"he|she|her|him|his|hers|herself|himself|"
            r"priest|monk|businessman|businessmen|"
            r"man|men|woman|women|"
            r"gentleman|gentlemen|lady|ladies|girl|girls|boy|boys"
            r")\.?\b",
            _pron,
            out,
            flags=re.IGNORECASE,
        )
        # Grammar repair for patient-possessive kinship phrases introduced by pronoun replacement.
        out = re.sub(
            r"\b(the patient)\s+("
            r"mother|father|sister|brother|daughter|son|"
            r"parent|parents|child|children|sibling|siblings|"
            r"aunt|uncle|grandmother|grandfather|grandma|grandpa|"
            r"stepmother|stepfather|stepdaughter|stepson|"
            r"mom|dad|baby"
            r")\b",
            lambda m: _match_case(m.group(1), "the patient's") + " " + m.group(2),
            out,
            flags=re.IGNORECASE,
        )

    if cfg.scrub_familial_terms:
        # Patient-role relationship terms that can directly reveal patient gender.
        # Keep kinship/family-history terms (mother/father/maternal/paternal/etc.) to
        # preserve clinically relevant family history semantics.
        out = re.sub(
            r"\b("
            r"wife|husband|spouse|partner|"
            r"girlfriend|boyfriend|fiance|fiancee"
            r"|widow|widower"
            r")\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )

    if cfg.scrub_male_female_tokens:
        out = re.sub(r"(?i)\b(male|female)\b", ph, out)

    if cfg.scrub_short_sex_markers:
        # Replace chart shorthand sex markers in gender-like contexts (e.g., "yo F with ...", "M, PMH ...").
        # Keep this strict to avoid non-gender uses like follow-up shorthand ("f/u").
        sex_repl = ph if cfg.short_sex_marker_replacement.lower() == "redacted" else "person"

        def _short_marker(token: str) -> str:
            return _match_case(token, sex_repl)

        def _age_ctx(m: re.Match[str]) -> str:
            age_phrase = m.group(1)
            marker = m.group(2)
            return f"{age_phrase} {_short_marker(marker)}"

        # Age phrase + shorthand marker (most common in clinical headers).
        out = re.sub(
            r"(?i)\b((?:___|\d{1,3})\s*(?:yo|y/o|y\.o\.|yr(?:s)?\s+old|year(?:s)?\s+old))\s+([MF])\b",
            _age_ctx,
            out,
        )

        def _marker_ctx(m: re.Match[str]) -> str:
            marker = m.group(1)
            return _short_marker(marker)

        # Standalone marker before clinical demographic cues.
        out = re.sub(
            r"(?i)\b([MF])\b(?=\s*,?\s*(?:with|who|present(?:s|ed)?|admitted|pmh|pmhx|h/o|history)\b)",
            _marker_ctx,
            out,
        )

        # Marker followed by pregnancy/gestational shorthand contexts.
        out = re.sub(
            r"(?i)\b([MF])\b(?=\s*(?:\d+\s*(?:wk|wks|week|weeks|mo|mos|month|months)|ga\b|ega\b|post[-\s]?partum|p/w\b|pw\b))",
            _marker_ctx,
            out,
        )

    if cfg.scrub_sex_specific_anatomy_terms:
        # Option B: scrub explicit sex-anatomy / reproductive terms that act as direct biological shortcuts.
        # Handle obstetric shorthand such as G3P1 / G2 P0.
        out = re.sub(r"\bG\s*\d+\s*P\s*\d+\b", ph, out, flags=re.IGNORECASE)

        out = re.sub(
            r"\b("
            # Male-specific
            r"prostate|prostatic|bph|psa|"
            r"testicular|testes|testis|testicle|scrotum|scrotal|"
            r"penis|penile|foreskin|circumcision|"
            r"epididymis|vas(?:\s+deferens)?|vasectomy|"
            # Female-specific
            r"uterus|uterine|cervix|cervical|endometrial|endometrium|hysterectomy|"
            r"ovary|ovaries|ovarian|fallopian|"
            r"vagina|vaginal|vaginosis|vulvovaginal|vulva|vulval|labia|"
            r"adnexa|adnexal|adenomyosis|endometrioma|"
            # Fibroid-family variants (beyond exact 'fibroid'/'fibroids').
            r"fibroid(?:s|al|ectomy)?|"
            r"leiomyoma(?:ta|s)?|"
            r"myomectomy|"
            r"uterine\s+mass(?:es)?|"
            # Advanced obstetrics/surgery + anatomy/hormone + devices + shorthand + identity/social cues.
            r"c-?section|cesarean|eclampsia|preeclampsia|placenta|amniotic|episiotomy|"
            r"tubal\s+ligation|tah-?bso|tah/?bso|\btah\b|\bbso\b|\bturp\b|"
            r"iud|mirena|nexplanon|\bocps?\b|birth\s+control\s+pills?|nuvaring|depo-?provera|"
            r"condoms?|diaphragm|"
            r"\bova\b|sperm|seminal|clitoral|labial|vaginitis|endometriosis|pcos|"
            r"gynecomastia|priapism|menarche|dysmenorrhea|amenorrhea|"
            r"estrogen|progesterone|progestin|prolactin|testosterone|androgens?|"
            r"transgender|cisgender|ftm|mtf|maiden\s+name|matriarch|patriarch|"
            r"paternity|prostitute|fraternity|sorority|"
            r"\bgyn\b|colposcopy|hysteroscopy|oophorectomy|abortion|"
            r"lactation|breastfeed\w*|\bbaby\b|"
            r"\blmp\b|\bbph\b|\bpid\b|"
            r"pregnan\w*|maternity|obstetric|obstetrics|ob/?gyn|gynecolog\w*|gynaecolog\w*|prenatal|post[-\s]?partum|"
            r"gravida|para|miscarriage\w*|"
            r"menstrual|menstruation|menopause|menopausal|menses|"
            r"mammogram|mastectomy|breast|breasts|nipple|nipples|areola\w*|"
            r"pap\s*smears?|salpingo-?oophorectomy|"
            r"erectile|erection"
            r")\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )
        # Contextual menstrual-period phrases without touching generic "time period".
        out = re.sub(
            r"\b(last\s+(?:menstrual\s+)?period|regular\s+periods?|menstrual\s+periods?)\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )

    if cfg.scrub_repro_medication_marker_terms:
        # Additional strict option for sex-coded medications/tests/pathologies.
        # Intentionally excludes "spironolactone" and "hrt" due to frequent non-gender clinical meanings.
        out = re.sub(
            r"\b("
            # A) BPH/ED and related medication cluster
            r"flomax|tamsulosin|finasteride|proscar|viagra|sildenafil|cialis|tadalafil|"
            r"lupron|leuprolide|bicalutamide|casodex|dutasteride|avodart|testim|androgel|"
            # B) Female-specific medication cluster
            r"levonorgestrel|tamoxifen|letrozole|"
            r"ortho\s+tri-?cyclen|premarin|yaz|estring|vagifem|estrace|provera|"
            r"medroxyprogesterone|clomiphene|"
            # C) Pregnancy testing/markers
            r"b-?hcg|bhcg|hcg|gravid\w*|amniocentesis|\bsab\b|\biup\b|"
            r"pitocin|lochia|"
            r"fetal|fetus|placental?|meconium|apgar|trimester|gestation\w*|chorionic|macrosomia|"
            # D) Granular genital/reproductive pathology/surgery
            r"salpingectomy|oophorectomy|orchiectomy|vasectomy|glans|"
            r"orchitis|epididymitis|balanitis|vulvovaginitis|cervicitis|salpingitis|"
            r"mastitis|hydrocele|spermatocele|varicocele|hypospadias|"
            r"leep|d\s*[&+]\s*c|d\s+and\s+c|paps?|endometritis|tubal|"
            r"\btvt\b|(?:bladder|vaginal|urethral|midurethral|suburethral)\s+sling|sling\s+(?:procedure|surgery)|"
            r"circumcised|uncircumcised|smegma|phallus|prostatism"
            r")\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )

    if cfg.scrub_gendered_personal_items:
        # Optional strict mode: replace strongly gender-coded personal items
        # with neutral phrases rather than redaction tokens.
        def _item_repl(m: re.Match[str]) -> str:
            src = m.group(0)
            term = src.lower()
            if "tampon" in term:
                return _match_case(src, cfg.gendered_personal_item_tampon_replacement)
            if term == "panty":
                return _match_case(src, "undergarment")
            if term == "panties":
                return _match_case(src, "undergarments")
            return _match_case(src, cfg.gendered_personal_item_bra_replacement)

        out = re.sub(
            r"\b("
            r"tampon|tampons|"
            r"panty|panties|"
            r"bra|bras|"
            r"sports?\s+bra|surgical\s+bra"
            r")\b",
            _item_repl,
            out,
            flags=re.IGNORECASE,
        )

    for pat, repl in cfg.extra_subs:
        out = pat.sub(repl, out)

    # Post-pass cleanup for repeated patient placeholders.
    out = re.sub(
        r"\b(the patient)(?:\s+the patient)+\b",
        lambda m: m.group(1),
        out,
        flags=re.IGNORECASE,
    )

    return out


def _require_distinct_output_csv(input_csv: Path, output_csv: Path) -> None:
    """Never overwrite the source CSV; always write a separate file."""
    if input_csv.resolve() == output_csv.resolve():
        raise ValueError(
            f"Output must be a new file path, not the input: {input_csv}\n"
            f"Pass --output-csv with a different name (e.g. {input_csv.stem}_gender_scrubbed.csv)."
        )


def scrub_cohort_csv(
    input_csv: Path,
    output_csv: Path,
    cfg: GenderCueScrubConfig,
    text_col: str = "text",
    chunksize: int = 5000,
) -> Path:
    """
    Load CSV (in chunks if large), scrub ``text_col``, write a **new** CSV at ``output_csv``.
    All other columns are copied unchanged. Returns ``output_csv`` resolved.
    """
    input_csv = input_csv.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    _require_distinct_output_csv(input_csv, output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    reader = pd.read_csv(input_csv, chunksize=chunksize)
    n_rows = 0
    first_chunk = True
    for chunk in reader:
        if text_col not in chunk.columns:
            raise ValueError(f"Column {text_col!r} not in {list(chunk.columns)}")
        chunk = chunk.copy()
        chunk[text_col] = chunk[text_col].astype(str).map(lambda s: scrub_clinical_gender_cues(s, cfg))
        chunk.to_csv(
            output_csv,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk,
        )
        first_chunk = False
        n_rows += len(chunk)

    print(f"Wrote new CSV: {output_csv} ({n_rows} data rows)")
    return output_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrub explicit gender labels and write a NEW CSV (never overwrites the input)."
    )
    p.add_argument("--input-csv", type=str, required=True)
    p.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Path for the new CSV (required to differ from input). Default: <input_stem>_gender_scrubbed.csv next to input.",
    )
    p.add_argument("--placeholder", type=str, default="[REDACTED]")
    p.add_argument("--no-inline-fields", action="store_true", help="Only start-of-line Sex/Gender")
    p.add_argument(
        "--pronouns",
        action="store_true",
        help="Replace sexed pronouns with patient-form phrases and redact explicit gendered titles/labels.",
    )
    p.add_argument(
        "--familial-terms",
        action="store_true",
        help="Redact familial/relationship terms like mother/father/wife/husband/son/daughter.",
    )
    p.add_argument("--male-female-tokens", action="store_true", help=r"Remove \b male|female \b anywhere")
    p.add_argument(
        "--short-sex-markers",
        action="store_true",
        help="Replace standalone sex shorthand markers (M/F), e.g. 'yo F with ...'.",
    )
    p.add_argument(
        "--sex-marker-replacement",
        type=str,
        default="person",
        choices=["person", "redacted"],
        help="Replacement for standalone M/F when --short-sex-markers is set.",
    )
    p.add_argument(
        "--sex-anatomy-terms",
        action="store_true",
        help="Redact sex-specific anatomy/reproductive terms (Option B scrub).",
    )
    p.add_argument(
        "--repro-medication-markers",
        action="store_true",
        help="Redact additional sex-coded medications, pregnancy tests, and granular reproductive pathologies.",
    )
    p.add_argument(
        "--gendered-personal-items",
        action="store_true",
        help="Replace gender-coded personal items with neutral phrases (e.g., tampons, panty/panties, bra/bras).",
    )
    p.add_argument("--dry-run", type=int, default=0, help="Print N before/after snippets and exit without writing")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GenderCueScrubConfig(
        placeholder=args.placeholder,
        scrub_inline_sex_or_gender_field=not args.no_inline_fields,
        scrub_he_she_pronouns=args.pronouns,
        scrub_familial_terms=args.familial_terms,
        scrub_male_female_tokens=args.male_female_tokens,
        scrub_short_sex_markers=args.short_sex_markers,
        short_sex_marker_replacement=args.sex_marker_replacement,
        scrub_sex_specific_anatomy_terms=args.sex_anatomy_terms,
        scrub_repro_medication_marker_terms=args.repro_medication_markers,
        scrub_gendered_personal_items=args.gendered_personal_items,
    )
    inp = Path(args.input_csv).expanduser().resolve()
    if args.output_csv.strip():
        out = Path(args.output_csv).expanduser().resolve()
    else:
        out = inp.with_name(f"{inp.stem}_gender_scrubbed.csv")
    _require_distinct_output_csv(inp, out)
    if args.dry_run > 0:
        df = pd.read_csv(inp, nrows=max(50, args.dry_run + 5))
        if "text" not in df.columns:
            raise ValueError("CSV needs a header with a 'text' column")
        for i in range(min(args.dry_run, len(df))):
            raw = str(df.iloc[i]["text"])
            clean = scrub_clinical_gender_cues(raw, cfg)
            print(f"=== row {i} ===")
            print("BEFORE:", raw[:500].replace("\n", "\\n"))
            print("AFTER :", clean[:500].replace("\n", "\\n"))
            print()
        print("Dry run only; no new CSV created.")
        print(f"Showed {min(args.dry_run, len(df))} row(s). Would write (example path): {out}")
        return
    scrub_cohort_csv(inp, out, cfg)


if __name__ == "__main__":
    main()
