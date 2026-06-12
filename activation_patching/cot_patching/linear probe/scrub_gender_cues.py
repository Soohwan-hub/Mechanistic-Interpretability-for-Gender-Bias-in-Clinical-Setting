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


_SECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(history\s+of\s+present\s+illness|hpi)\s*:?\s*$", re.IGNORECASE), "<SEC_HPI>"),
    (re.compile(r"^\s*(chief\s+complaint|cc)\s*:?\s*$", re.IGNORECASE), "<SEC_CC>"),
    (re.compile(r"^\s*(past\s+medical\s+history|pmh|pmhx)\s*:?\s*$", re.IGNORECASE), "<SEC_PMH>"),
    (re.compile(r"^\s*(past\s+surgical\s+history|psh)\s*:?\s*$", re.IGNORECASE), "<SEC_PSH>"),
    (re.compile(r"^\s*(social\s+history|shx)\s*:?\s*$", re.IGNORECASE), "<SEC_SHX>"),
    (re.compile(r"^\s*(family\s+history|fhx)\s*:?\s*$", re.IGNORECASE), "<SEC_FHX>"),
    (re.compile(r"^\s*(medications?|home\s+medications?)\s*:?\s*$", re.IGNORECASE), "<SEC_MEDS>"),
    (re.compile(r"^\s*(allerg(?:y|ies))\s*:?\s*$", re.IGNORECASE), "<SEC_ALLERGIES>"),
    (re.compile(r"^\s*(physical\s+exam(?:ination)?)\s*:?\s*$", re.IGNORECASE), "<SEC_PHYSICAL_EXAM>"),
    (re.compile(r"^\s*(review\s+of\s+systems|ros)\s*:?\s*$", re.IGNORECASE), "<SEC_ROS>"),
    (re.compile(r"^\s*(assessment(?:\s+and\s+plan)?|a/p|plan)\s*:?\s*$", re.IGNORECASE), "<SEC_ASSESSMENT_PLAN>"),
    (re.compile(r"^\s*(hospital\s+course|brief\s+hospital\s+course)\s*:?\s*$", re.IGNORECASE), "<SEC_HOSPITAL_COURSE>"),
    (re.compile(r"^\s*(discharge\s+diagnos(?:is|es))\s*:?\s*$", re.IGNORECASE), "<SEC_DISCHARGE_DIAGNOSIS>"),
    (re.compile(r"^\s*(discharge\s+medications?)\s*:?\s*$", re.IGNORECASE), "<SEC_DISCHARGE_MEDS>"),
    (re.compile(r"^\s*(follow-?up\s+instructions?)\s*:?\s*$", re.IGNORECASE), "<SEC_FOLLOWUP_INSTRUCTIONS>"),
    (re.compile(r"^\s*(discharge\s+disposition)\s*:?\s*$", re.IGNORECASE), "<SEC_DISCHARGE_DISPOSITION>"),
    (re.compile(r"^\s*(pertinent\s+results?)\s*:?\s*$", re.IGNORECASE), "<SEC_PERTINENT_RESULTS>"),
    (re.compile(r"^\s*(medications?\s+on\s+admission)\s*:?\s*$", re.IGNORECASE), "<SEC_MEDS_ON_ADMISSION>"),
    (
        re.compile(r"^\s*(major\s+surgical\s+or\s+invasive\s+procedure(?:s)?)\s*:?\s*$", re.IGNORECASE),
        "<SEC_MAJOR_SURGICAL_PROCEDURE>",
    ),
]

_LAB_CUE_RE = re.compile(
    r"\b("
    r"na|k|cl|co2|hco3|bun|cr|creat(?:inine)?|glucose|lactate|"
    r"wbc|hgb|hct|plt|platelets?|"
    r"ast|alt|alk(?:aline)?(?:\s*phos(?:phatase)?)?|bilirubin|"
    r"ca|calcium|mg|magnesium|phos|phosphate|"
    r"inr|ptt?|troponin|bnp|estrad\w*"
    r")\b",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:/\d+)?(?![A-Za-z])")
_ANTHRO_DOSE_RE = re.compile(
    r"(?<![A-Za-z])(?P<num>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>kg|kgs?|lb|lbs?|pounds?|cm|mm|in|inch(?:es)?|mg|mcg|g|gm|gms?|grams?|ml|cc|oz|ounces?)\b",
    re.IGNORECASE,
)
_MIMIC_UNDERSCORE_RE = re.compile(r"_{3,}")
_MIMIC_BRACKET_RE = re.compile(r"\[\s*(?:\*\*[^][]+\*\*|[A-Za-z][A-Za-z0-9 _/\-]{0,40})\s*\]")


def _canonicalize_section_header(line: str) -> str:
    stripped = line.strip()
    for pat, token in _SECTION_PATTERNS:
        if pat.match(stripped):
            return token
    return line


def _is_lab_style_line(line: str) -> bool:
    return bool(_LAB_CUE_RE.search(line) and re.search(r"\d", line))


def _lowercase_preserve_tags(text: str) -> str:
    tags: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        tags.append(m.group(0))
        return f"__TAG_{len(tags) - 1}__"

    staged = re.sub(r"<[^>\n]+>", _stash, text)
    staged = staged.lower()
    for i, tag in enumerate(tags):
        staged = staged.replace(f"__tag_{i}__", tag).replace(f"__TAG_{i}__", tag)
    return staged


def _normalize_style_artifacts(
    text: str,
    num_marker: str = "<NUM>",
    redacted_marker: str = "<REDACTED>",
    drop_redacted_tokens: bool = False,
    lowercase_text: bool = True,
) -> str:
    if not text:
        return text

    out = text.replace("\r\n", "\n").replace("\r", "\n")
    style_redacted_repl = "" if drop_redacted_tokens else redacted_marker
    out = _MIMIC_UNDERSCORE_RE.sub(style_redacted_repl, out)
    out = _MIMIC_BRACKET_RE.sub(style_redacted_repl, out)
    # First pass on full text to catch number+unit patterns split by newlines.
    out = _ANTHRO_DOSE_RE.sub(lambda m: f"{num_marker} {m.group('unit')}", out)

    normalized_lines: list[str] = []
    for raw_line in out.split("\n"):
        line = raw_line.strip()
        line = re.sub(r"[ \t]+", " ", line)
        line = _canonicalize_section_header(line)
        line = _ANTHRO_DOSE_RE.sub(lambda m: f"{num_marker} {m.group('unit')}", line)
        if _is_lab_style_line(line):
            line = _NUM_RE.sub(num_marker, line)
            line = re.sub(
                rf"{re.escape(num_marker)}(?:\s*[-–>]+\s*{re.escape(num_marker)})+",
                num_marker,
                line,
            )
        normalized_lines.append(line)

    out = "\n".join(normalized_lines)
    # Convert repeated comma artifacts into sentence-break dots to avoid keeping
    # punctuation runs that can become stylistic shortcuts.
    out = re.sub(r"\s*,\s*,+\s*", ". . ", out)
    out = re.sub(r"\.{2,}", ".", out)
    out = re.sub(r"([,;:!?])\1+", r"\1", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    if lowercase_text:
        out = _lowercase_preserve_tags(out)
    return out


@dataclass
class GenderCueScrubConfig:
    """What to remove or replace. All replacements use ``placeholder`` unless noted."""

    placeholder: str = "[REDACTED]"
    replacement_mode: str = "redacted"
    # Structured header / form fields (recommended baseline)
    scrub_sex_colon_line: bool = True
    scrub_gender_colon_line: bool = True
    # Broader “explicit label” phrases (still mostly administrative)
    scrub_assigned_sex_at_birth_line: bool = True
    # Inline “Sex:” / “Gender:” anywhere on a line (not only start-of-line)
    scrub_inline_sex_or_gender_field: bool = True
    # Strong / optional (can distort clinical meaning)
    scrub_he_she_pronouns: bool = False
    pronoun_replacement: str = "they"
    scrub_familial_terms: bool = False
    scrub_male_female_tokens: bool = False
    scrub_short_sex_markers: bool = False
    short_sex_marker_replacement: str = "person"
    scrub_sex_specific_anatomy_terms: bool = False
    scrub_repro_medication_marker_terms: bool = False
    scrub_mae_proxy_cues: bool = False
    scrub_style_artifacts: bool = False
    style_num_marker: str = "<NUM>"
    style_redacted_marker: str = "<REDACTED>"
    style_drop_redacted_tokens: bool = False
    style_lowercase: bool = True
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
    ph = "" if cfg.replacement_mode.lower() == "delete" else cfg.placeholder

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
        # Map sexed pronouns to either neutral forms ("they") or full redaction.
        use_redacted = cfg.pronoun_replacement.lower() == "redacted"

        def _pron(m: re.Match[str]) -> str:
            t = m.group(0).lower().rstrip(".")
            if use_redacted:
                return ph
            if t in {"he", "she"}:
                return _match_case(m.group(0), "they")
            if t == "him":
                return _match_case(m.group(0), "them")
            if t == "her":
                # Heuristic: treat "her <noun>" as possessive in clinical prose,
                # but keep common object-pronoun contexts as "them".
                tail = m.string[m.end() :]
                nxt = re.match(r"\s+([A-Za-z][A-Za-z0-9_-]*)", tail)
                if not nxt:
                    return _match_case(m.group(0), "them")
                next_word = nxt.group(1).lower()
                non_possessive_followers = {
                    "and",
                    "or",
                    "but",
                    "to",
                    "for",
                    "with",
                    "without",
                    "in",
                    "on",
                    "at",
                    "of",
                    "by",
                    "from",
                    "into",
                    "onto",
                    "as",
                    "because",
                    "if",
                    "when",
                    "while",
                    "that",
                    "who",
                    "whom",
                    "yesterday",
                    "today",
                    "now",
                    "then",
                    "there",
                    "here",
                }
                if next_word in non_possessive_followers:
                    return _match_case(m.group(0), "them")
                return _match_case(m.group(0), "their")
            if t in {"his", "hers"}:
                return _match_case(m.group(0), "their")
            if t in {"himself", "herself"}:
                return _match_case(m.group(0), "themselves")
            if t in {"priest", "monk", "businessman", "businessmen"}:
                return _match_case(m.group(0), "patient")
            # Keep non-pronoun explicit gender terms/honorifics fully redacted.
            return ph

        out = re.sub(
            r"\b("
            r"mr|mrs|ms|miss|mx|"
            r"he|she|her|him|his|hers|herself|himself|"
            r"wife|wives|husband|husbands|spouse|spouses|partner|partners|"
            r"girlfriend|girlfriends|boyfriend|boyfriends|"
            r"fiance|fiancee|fiances|fiancees|"
            r"widow|widower|widows|widowers|"
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
        # Menstrual-period phrases before the main token list so "menstrual" is not stripped alone
        # (which would leave "last period" as a leak).
        out = re.sub(
            r"\b("
            r"last\s+(?:menstrual\s+|normal\s+)?period|"
            r"regular\s+periods?"
            r")\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )
        # Handle obstetric shorthand such as G3P1 / G2 P0.
        out = re.sub(r"\bG\s*\d+\s*P\s*\d+\b", ph, out, flags=re.IGNORECASE)

        out = re.sub(
            r"\b("
            # Male-specific
            r"prostat\w*|bph|psa|"
            r"testicular|testes|testis|testicle|scrotum|scrotal|"
            r"penis|penile|foreskin|circumcision|"
            r"epididymis|vas(?:\s+deferens)?|vasectomy|"
            # Female-specific
            r"uterus|uterine|cervix|cervical|endocervical|endometrial|endometrium|hysterectomy|"
            r"intrauterine|bimanual|speculum|"
            r"ovary|ovaries|ovarian|fallopian|"
            r"vagina|vaginal|transvaginal|vaginosis|vulvovaginal|vulva|vulval|labia|"
            r"adnexa|adnexal|adnexae|adenomyosis|endometrioma\w*|endometrioid\w*|"
            r"menorrhagia|oligomenorrhea|galactorrhea|"
            r"cul-?de-?sac|(?<!laryngeal\s)vestibule|myometrium|myometrial|parametrial|"
            # Fibroid-family variants (beyond exact 'fibroid'/'fibroids').
            r"fibroid(?:s|al|ectomy)?|"
            r"leiomyoma(?:ta|s)?|"
            r"myomectomy|"
            r"uterine\s+mass(?:es)?|"
            # Advanced obstetrics/surgery + anatomy/hormone + devices + shorthand + identity/social cues.
            r"c(?:\s|-)?section|caesarean|cesarean|eclampsia|preeclampsia|placenta|amniotic|episiotomy|"
            r"tubal\s+ligation|tah-?bso|tah/?bso|\btah\b|\bbso\b|\bturp\b|"
            r"iud|mirena|nexplanon|\bocps?\b|birth\s+control\s+pills?|nuvaring|depo-?provera|"
            r"condoms?|diaphragm|"
            r"\bova\b|sperm|seminal|clitoral|labial|vaginitis|endometriosis|pcos|"
            r"gynecomastia|priapism|menarche|dysmenorrhea|amenorrhea|"
            r"estrogen\w*|estrad\w*|progesterone|progestin|prolactin|testosterone|androgens?|"
            r"transgender|cisgender|ftm|mtf|maiden\s+name|matriarch|patriarch|"
            r"paternity|prostitute|fraternity|sorority|"
            r"\bgyn\b|colposcopy|hysteroscopy|oophorectomy|abortion|"
            r"lactation|breastfeed\w*|\bbaby\b|"
            r"\blmp\b|\bbph\b|\bpid\b|"
            r"pregnan\w*|maternity|obstetric|obstetrics|ob/?gyn|gynecolog\w*|gynaecolog\w*|prenatal|post[-\s]?partum|"
            r"gravida|para|miscarriage\w*|\bsvd\b|"
            r"menstrual|menstruation|menstural|menstural\s+bleeding|menometrorrhagia|premenstrual\w*|"
            r"(?:pre|post|peri)?menopaus\w*|menses|periods?|withdrawal\s+bleed\w*|"
            r"cycle|flow|pads?|hot\s+flashes?|spotting|heavy\s+bleeding|"
            r"mammogram\w*|lumpectom\w*|[A-Za-z]*mastectom\w*|breast|breasts|nipple|nipples|areola\w*|"
            r"pap\s*smears?|salpingo-?oophorectomy|"
            r"rectovaginal(?:\s+fistula)?|"
            r"vaginectom\w*|vulvar?\s+cysts?|mullerian(?:\s+agenesis)?|"
            r"corpus\s+luteum(?:\s+cysts?)?|luteum|luteal|"
            r"salpingo\w*|salpingectom\w*|vag(?:inal)?\s+cream|miconazole(?:\s+nitrate)?\s+vag(?:inal)?|"
            r"pelvic\s+(?:pain|ultrasound|ct|exam|mesh|mass|us)|"
            r"pgynhx|pgyn|colpo\w*|prometrium|ovcon|myoma\w*|mammoplast\w*|"
            r"erectile|erection|impotence|childbirth"
            r")\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )
        # Salpingo-oophorectomy chart misspellings (salpingoopherectomy, oopherectomy, oophrectomy).
        out = re.sub(
            r"\bsalpingo\w*opher\w*ectom\w*\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            r"\booph(?:\w*rectom\w*|\w*ectom\w*)\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )
        # Obstetric "labor" phrases only (not generic labor/delivered).
        out = re.sub(
            r"\b("
            r"full[-\s]?term(?:\s+(?:pregnan\w*|delivery|labor|birth))?|"
            r"induction\s+of\s+labor|"
            r"(?:pre)?term\s+labor|"
            r"labor\s+(?:and|&)\s+delivery"
            r")\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            r"\bmenstrual\s+periods?\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )

    if cfg.scrub_repro_medication_marker_terms:
        # Additional strict option for sex-coded medications/tests/pathologies.
        # Intentionally excludes spironolactone; includes HRT-family per MAE leakage findings.
        out = re.sub(
            r"\b("
            # A) BPH/ED and related medication cluster
            r"flomax|tamsulosin|alfuzosin|finasteride|proscar|propecia|"
            r"viagra|sildenafil|cialis|tadalafil|levitra|vardenafil|"
            r"lupron|leuprolide|bicalutamide|casodex|dutasteride|avodart|testim|androgel|"
            # B) Female-specific medication cluster
            r"levonorgestrel|tamoxifen|letrozole|arimidex|anastrozole|"
            r"raloxifene|exemestane|fulvestrant|faslodex|herceptin|"
            r"ortho\s+tri-?cyclen|premarin|yaz|estring|vagifem|estrace|provera|"
            r"medroxyprogesterone|clomiphene|"
            r"norethindrone|drospirenone|norgestimate|desogestrel|"
            r"prempro|alendronate|fosamax|denosumab|zometa|bisphosphonates?|"
            r"\bhrt\b|hormone\s+replacement(?:\s+therapy)?|"
            # C) Pregnancy testing/markers
            r"b-?hcg|bhcg|hcg|gravid\w*|amniocentesis|\bsab\b|\biup\b|"
            r"pitocin|lochia|misoprostol|cytotec|"
            r"breech(?!\s+(?:rhythm|of\b))|"
            r"hellp|"
            r"fetal|fetus|placental?|meconium|apgar|trimester|gestation\w*|chorionic|macrosomia|"
            r"antepartum|peripartum|cerclage|chorioamnionitis|"
            r"placenta\s+previa|\bprevia\b|"
            r"choriocarcinoma|nulliparous|"
            r"\bega\b|"
            # D) Granular genital/reproductive pathology/surgery
            r"salpingectomy|oophorectomy|orchiectomy|vasectomy|glans|"
            r"orchitis|epididymitis|balanitis|vulvovaginitis|cervicitis|salpingitis|"
            r"mastitis|hydrocele|spermatocele|varicocele|hypospadias|"
            r"seminoma|phimosis|peyronie\w*|"
            r"bartholin\w*|dyspareunia|fibroadenoma|"
            r"leep|d\s*[&+]\s*c|d\s+and\s+c|paps?|endometritis|tubal|"
            r"\btvt\b|(?:pelvic|bladder|vaginal|urethral|midurethral|suburethral)\s+sling|"
            r"sling\s+(?:procedure|surgery)|"
            r"circumcised|uncircumcised|smegma|phallus|prostatism"
            r")\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )

    if cfg.scrub_mae_proxy_cues:
        # Extra aggressive proxy scrub for MAE-observed leakage clusters.
        # Intentionally excludes HIV/STI terms (STD/STI, gonorrhea, syphilis,
        # condyloma, herpes, etc.): those reflect epidemiological bias in model
        # weights, not direct biological gender keywords, and are kept for study.
        out = re.sub(
            r"\b("
            # Contraception / OB-GYN chart shorthand
            r"contracept\w*|birth\s+control|family\s+planning|"
            r"\bob\/?gyn\b|\bgyn(?:ecolog\w*|a?ecolog\w*)?\b|"
            r"\bob(?:hx|h|triage)?\b|\bgyn(?:hx|h)?\b|"
            r"c(?:\s|-)?section|caesarean|cesarean|\bsvd\b|"
            # BPH / prostate proxy cluster
            r"\bbph\b|"
            r"benign\s+prostat\w+\s+hyperplas\w*|"
            r"prostat(?:ic)?\s+hypertroph\w*|"
            r"protat\w*|protatic\w*|"
            r"benign\s+hyperplas\w*|benign\s+hypertroph\w*|"
            r"urinary\s+obstruction|bladder\s+stone\w*|"
            r"lower\s+urinary\s+tract\s+symptoms|"
            r"\bluts\b|enlarged\s+prostate|"
            # Sexual-risk social/behavioral cues (not epidemiological STI/HIV terms)
            r"sexual\s+orientation|homosexual(?:\s+exposure)?|"
            r"men\s+who\s+have\s+sex\s+with\s+men|\bmsm\b"
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
    if cfg.scrub_sex_specific_anatomy_terms:
        # Residual menstrual cues after partial token redaction.
        out = re.sub(
            r"\blast\s+(?:"
            + re.escape(ph)
            + r"\s+|normal\s+)?period\b",
            ph,
            out,
            flags=re.IGNORECASE,
        )

    if cfg.scrub_style_artifacts:
        out = _normalize_style_artifacts(
            out,
            num_marker=cfg.style_num_marker,
            redacted_marker=cfg.style_redacted_marker,
            drop_redacted_tokens=cfg.style_drop_redacted_tokens,
            lowercase_text=cfg.style_lowercase,
        )

    if cfg.replacement_mode.lower() == "delete":
        # Deletion mode: tidy artifacts introduced by span removal.
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r" *\n *", "\n", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r"\.{2,}", ".", out)

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
    p.add_argument(
        "--replacement-mode",
        type=str,
        default="redacted",
        choices=["redacted", "delete"],
        help="How to replace matched spans: keep a marker token ('redacted') or delete spans entirely ('delete').",
    )
    p.add_argument("--no-inline-fields", action="store_true", help="Only start-of-line Sex/Gender")
    p.add_argument(
        "--pronouns",
        action="store_true",
        help="Replace sexed pronouns with they/them/their forms and redact explicit gendered titles/labels.",
    )
    p.add_argument(
        "--pronoun-replacement",
        type=str,
        default="they",
        choices=["they", "redacted"],
        help="When --pronouns is enabled: replace with neutral they/them/their forms, or fully redact pronouns.",
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
        "--mae-proxy-cues",
        action="store_true",
        help="Aggressive scrub for MAE-observed proxy cues (contraception/OB-GYN/BPH/sexual-risk). Keeps HIV/STI disease terms for epidemiological-bias analysis.",
    )
    p.add_argument(
        "--style-normalize",
        action="store_true",
        help="Normalize formatting artifacts (punctuation/whitespace/section headers/lab numbers) without semantic scrubbing.",
    )
    p.add_argument(
        "--style-num-marker",
        type=str,
        default="<NUM>",
        help="Marker for numeric values on lab-style lines when --style-normalize is set.",
    )
    p.add_argument(
        "--style-redacted-marker",
        type=str,
        default="<REDACTED>",
        help="Canonical replacement token for MIMIC de-identification placeholders when --style-normalize is set.",
    )
    p.add_argument(
        "--style-redacted-delete",
        action="store_true",
        help="When --style-normalize is set, delete MIMIC de-identification placeholders instead of inserting a marker token.",
    )
    p.add_argument(
        "--style-keep-case",
        action="store_true",
        help="When --style-normalize is set, do not lowercase note text.",
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
        replacement_mode=args.replacement_mode,
        scrub_inline_sex_or_gender_field=not args.no_inline_fields,
        scrub_he_she_pronouns=args.pronouns,
        pronoun_replacement=args.pronoun_replacement,
        scrub_familial_terms=args.familial_terms,
        scrub_male_female_tokens=args.male_female_tokens,
        scrub_short_sex_markers=args.short_sex_markers,
        short_sex_marker_replacement=args.sex_marker_replacement,
        scrub_sex_specific_anatomy_terms=args.sex_anatomy_terms,
        scrub_repro_medication_marker_terms=args.repro_medication_markers,
        scrub_mae_proxy_cues=args.mae_proxy_cues,
        scrub_style_artifacts=args.style_normalize,
        style_num_marker=args.style_num_marker,
        style_redacted_marker=args.style_redacted_marker,
        style_drop_redacted_tokens=args.style_redacted_delete,
        style_lowercase=not args.style_keep_case,
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
