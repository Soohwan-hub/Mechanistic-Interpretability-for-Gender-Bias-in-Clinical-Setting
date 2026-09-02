# HANDOFF

We're continuing an ongoing task. Use this handoff as the current source of
truth unless I correct it.

**Last updated:** 2026-08-28 · branch `om` · see `git log` for HEAD

---

## Goal

Get the BlackboxNLP 2026 paper ("Does Chain-of-Thought Prompting Debias
Clinical LLMs?") submission-ready, and keep the supporting research repo
auditable. Two intertwined workstreams:

1. **Paper edits.** Apply teammate-specified revisions to
   `paper/rtar_blackbox_main.tex` so it can be pasted straight into Overleaf.
2. **Figures.** Regenerate paper figures in a standardized palette, always
   traced back to real committed source data.

---

## Current status

### Done

- **`paper/rtar_blackbox_main.tex`** contains the real ACL submission plus all
  assets (17 figures, `acl.sty`, `acl_natbib.bst`, `custom.bib`,
  `anthology.bib.txt`, `tests/`). Compiles clean: **10 pages, 0 errors, 0
  missing figures.**
- **All 8 of the teammate's instructed revisions applied and verified**
  line-by-line against the pristine original at
  `~/Downloads/rtar_blackbox_Sam/rtar_blackbox_main.tex`:
  1. Abstract CoT block
  2. Introduction contribution (2), plus 15 → 20 CoT prompts in §2.1
  3. `\paragraph{MLP intervention.}`
  4. CoT `\paragraph{Overview}`
  5. `\subsection{Comparison between Simple & CoT}` (3 paragraphs → 1)
  6. Discussion sentence
  7. Conclusion sentence
  8. Limitations sentence swap
- **6 mid-sentence em dashes removed** from body prose (lines 134, 201, 255,
  267, 277, 313).
- **`paper_figures/`** holds 12 regenerated figures in the standard palette,
  each documented with exact source data in `paper_figures/README.md`, plus one
  alternate-palette variant (`fig5_..._redpink.png`, pink/dark-red instead of
  green/orange). The variant was made by recolouring the unfiltered `f0a910e`
  fig5 render pixel-by-pixel via `paper_figures/recolor_fig5_palette.py` —
  nothing re-plotted, so values and layout are preserved exactly. **No fig5
  generating script exists anywhere in this repo's history**; only PNG outputs
  were ever committed, which is why recolouring is the only safe way to restyle
  it.
- **Figure 9 source numbers committed** at `raw_uploads/cot_residual_fig9/`
  (560 KB, extracted from a 14 GB bundle) with a working
  `regenerate_fig9.py`.
- **Repo organized**: per-folder READMEs, `_archive/` for relocated files,
  root README converted from a hand-maintained file tree to a pointer table.
- **`.vscode/settings.json`** configures LaTeX Workshop autobuild on save.
- **Three NeurIPS venue variants** now live in `paper/`: `neurips_most_recent.tex`
  (Interp4Discovery), `GenAI4health.tex`, and `AI4Good.tex`. Same content and
  figures, differing only in track option and `\workshoptitle`.
- **`GenAI4health.tex` and `AI4Good.tex` audited against
  `paper/neurips_formatting_instructions.tex`** and brought into compliance:
  track option plus `\workshoptitle`, 17 headings and 3 figure captions
  converted to sentence case, `\input{checklist.tex}` added. Both compile
  clean at 23 pages, 9 content pages, 0 errors. Only formatting changed, no
  prose or values.

### In progress / not started

- Nothing actively mid-edit. Working tree is clean and pushed.

### Still needs to happen

- **Lines 140 and 142 of `rtar_blackbox_main.tex` contradict the new framing.**
  They still read "The grid averages do hide a sparse surviving signal…" and
  "CoT MLP patching leaves only a weak, prompt-specific, and model-specific
  remainder…". The teammate's search-and-destroy list bans "grid averages,"
  but no FIND block covered these paragraphs. **Needs the teammate's call on
  replacement wording. Do not rewrite them unprompted.**
- **Figure 10** (`om_comparison/figures/fig10_logprob_delta_qwen_vs_olmo.png`)
  is invalid. See "Known data problems" below.
- **fig5 rheumatoid-arthritis panel** renders near-blank after the layer
  filter. Faithful, but not useful in print. Options are documented in
  `paper_figures/README.md`.

---

## Important context

### Working style the user expects

- **Follow instructions literally.** Apply exactly the specified edits, nothing
  adjacent. Scope creep has been the single biggest recurring problem in this
  project.
- **Never change formatting** that wasn't explicitly requested. This file is
  pasted into Overleaf, so whitespace, float specifiers, `\paragraph{}` line
  breaks, and preamble all matter.
- **Do not commit unless asked.** The user says when.
- **Be concise.** No verbose preambles or narration.
- **Flag, don't decide.** When something looks wrong but is out of scope, say
  so and leave it alone.

### Prose style rules (anti-AI-slop protocol)

Applies to **new or rewritten prose only**, not to the teammate's existing text:

- No em dashes in body prose. **Exception:** structural uses are fine, e.g. the
  `\textbf{Label} --- description` separators in the contribution list, and the
  file-header comment. Numeric en dashes like `0.08–0.28` are correct, leave
  them.
- No semicolons.
- Colons are acceptable when they read naturally (introducing a list, a quote,
  a clear setup). Remove only choppy connector colons.
- No choppy sentences, no 3+ clause pile-ups, no flowery language.
- Informative prose, strong consistent voice.

### Key paths

| Path | What |
|---|---|
| `paper/rtar_blackbox_main.tex` | The paper. Only `.tex` that matters. |
| `~/Downloads/rtar_blackbox_Sam/rtar_blackbox_main.tex` | **Pristine original.** Always `diff` against this before claiming a change set is clean. |
| `paper_figures/README.md` | Per-figure source data, filtering notes, caveats. |
| `raw_uploads/cot_residual_fig9/` | Fig 9 numbers + `regenerate_fig9.py`. |
| `docs/BUNDLE_DIGEST.md` | Schema and per-layer numbers for committed bundles. |

### Standard figure palette

```
MALE_GREEN    #488f31     FEMALE_ORANGE #de3e00
BAR_ORANGE    #f08056     BAR_MUTED     #f8b9a1
GRIDCOLOR     #e8e8e8
Divergent ramp: #de3e00 → #f08056 → #f8b9a1 → #f1f1f1 → #aecdc2 → #6aaa96 → #488f31
```

White background, horizontal gridlines only, ticks 16 / axis titles 18 / title
20, kaleido `scale=3`. Green is positive **except fig9**, where the user asked
for the ramp reversed (orange positive).

### Environment gotchas

- `plotly 5.22.0` requires `kaleido==0.2.1`. Newer kaleido breaks
  `write_image()`.
- Rendering at `scale=3` can exceed a 120 s foreground timeout. Run in
  background.

---

## Decisions already made

- **`rtar_blackbox_main.tex` is the canonical filename.** The zip's
  `acl_latex.tex` was blank upstream ACL boilerplate and was discarded; the
  real content was renamed into place.
- **Figure float specifiers stay as-is.** 13 figures use `[h]`, which LaTeX
  cannot honor, so floats queue and flush at section breaks. This causes the
  whitespace gap before Discussion. It exists in the original. User chose to
  keep it rather than change formatting. Fix would be `[h]` → `[tb]`, but
  figures would move.
- **Branch `om` is left alone.** It is 16+ commits ahead of `main`; `main` is
  only 4 ahead and has nothing relevant. No rebase, no merge, no force-push.
- **Figure 4 values come from the team, not the repo.** Repo-computed
  `condition_token_mean` values are close but do not match, and OLMo-sarcoidosis
  is `NaN` here. Documented in `paper_figures/README.md`.
- **Fig 9 panels are not all Prompt A.** Multiple sclerosis comes from
  **PROMPT_C**; the others are PROMPT_A. Verified label-for-label. Window
  positions are hard-coded per panel because no single rule reproduces all four.
- **Large source bundles are not committed.** Only the extracted numbers are.

---

## What to avoid

- **Do not run heavy recomputation.** Regenerating the CoT residual bundle is
  10+ GPU-hours per condition. Always look for saved artifacts first.
- **Do not confuse MLP and residual-stream bundles.** `raw_uploads/cot_patching_{qwen,olmo}/`
  are **MLP**. Fig 9 needs **residual-stream** data. Mixing these up produced
  two wrong figures earlier.
- **Do not invent or eyeball values.** A previous attempt to read heatmap cells
  visually was correctly rejected. If the data can't be found, stop and say so.
- **Do not broaden a punctuation or style pass into prose rewriting.** This
  happened once and required a full revert to the pristine original.
- **Do not touch the teammate's existing prose** unless an instruction names it.
- **Do not add `\graphicspath`.** It was added once, then removed. Figures live
  alongside the `.tex` and resolve fine.

---

## Known data problems

- **Figure 10 is invalid.** `raw_uploads/log_delta/` is labeled "Qwen" but is
  actually **OLMo-shaped**: 31 layers (L1–L31, i.e. a 32-layer model), and the
  tokenizer fingerprint shows `<|endoftext|>` plus `<|user|>` split into 4
  tokens and `sarcoidosis` → `sarc`/`oid`/`osis`. Qwen 2.5-7B has 28 layers and
  uses ChatML single tokens. The figure's caption claims Qwen peaks at layer 30,
  which is impossible. **No genuine Qwen logprob_delta run exists in this repo**
  (see README status table: "✗ pending Lambda run"). Recommendation: pull the
  figure and the §2.6 model-comparison paragraph until the Qwen run exists, and
  correct the mislabel in `docs/results_for_writeup.md` line 6 and root README
  line 99.
- **Fig 9's title is misleading.** "Top-15 tokens (max|RS|)" does not describe
  the actual selection, which is a contiguous 15-token window. Worth correcting
  in the caption.

---

## Open questions / blockers

1. **Lines 140/142 rewording** — needs the teammate's decision.
2. **Figure 10** — pull it, or wait for a real Qwen logprob_delta run?
3. **fig5 RA panel** — annotate as no-effect, rescale it, or swap the condition?
4. **Optional abstract addition** from instruction 6 (CoT next-token gender
   probabilities, ~0.94 Female / ~0.06 Male) was **not added**, since the
   numbers couldn't be verified against any source. Add only if the user
   confirms them.

---

## Next best step

Ask the user whether the teammate has ruled on **lines 140 and 142**. That is
the only thing standing between the current file and a fully consistent
Overleaf paste. If they have no answer yet, the file is already safe to paste
as-is; those two paragraphs are simply stale relative to the new framing.

---

## How to respond

Start with a short summary of your understanding. Continue from here instead of
restarting from scratch. If anything critical is missing, ask only the minimum
questions needed.

---

## Verification recipe

Before claiming the `.tex` is clean:

```bash
cd paper

# 1. Diff against the pristine original. Expect ONLY intended hunks.
diff ~/Downloads/rtar_blackbox_Sam/rtar_blackbox_main.tex rtar_blackbox_main.tex | grep -E "^[0-9]"

# 2. Confirm no structural drift.
diff <(sed -n '1,33p' ~/Downloads/rtar_blackbox_Sam/rtar_blackbox_main.tex) \
     <(sed -n '1,33p' rtar_blackbox_main.tex)          # preamble
grep -o 'begin{figure}\[[a-z]*\]' rtar_blackbox_main.tex | sort | uniq -c   # floats

# 3. Em dashes: only the file comment and contribution list should remain.
grep -n '\-\-\-\|—' rtar_blackbox_main.tex

# 4. Compile twice, check for real errors.
pdflatex -interaction=nonstopmode -file-line-error rtar_blackbox_main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode -file-line-error rtar_blackbox_main.tex >/dev/null 2>&1
grep "^!" rtar_blackbox_main.log ; grep -c "not found on" rtar_blackbox_main.log
```

Expected: 10 pages, 0 errors, 0 missing figures. The only warning should be
`` `h' float specifier changed to `ht' ``, which is pre-existing and intentional.
