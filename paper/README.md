# paper/

LaTeX sources for the paper. Three venue variants share the same content and
figures, differing only in the NeurIPS track option and workshop title.

| File | Venue | Class option |
|---|---|---|
| `GenAI4health.tex` | GenAI4Health workshop | `[dblblindworkshop,nonatbib]` + `\workshoptitle{GenAI4Health}` |
| `AI4Good.tex` | Trustworthy AI for Good (AI4GOOD) workshop | `[dblblindworkshop,nonatbib]` + `\workshoptitle{Trustworthy AI for Good (AI4GOOD)}` |
| `neurips_most_recent.tex` | Interp4Discovery workshop | `[dblblindworkshop,nonatbib]` + `\workshoptitle{Interpretability for Discovery (Interp4Discovery)}` |

## Required files

Every variant needs these present to compile:

- `neurips_2026.sty` — official NeurIPS 2026 style file, do not modify
- `checklist.tex` — pulled in via `\input`, has no preamble of its own
- The figure PNGs alongside the `.tex` (no `\graphicspath`, figures resolve locally)

`neurips_formatting_instructions.tex` is the official NeurIPS template, kept as
the reference for formatting audits. It is not compiled as part of any paper.

## Formatting compliance

`GenAI4health.tex` and `AI4Good.tex` were audited against
`neurips_formatting_instructions.tex` and conform to the rules it states:
sentence-case headings and captions, figure captions after figures, table
titles before tables, no vertical rules, graphics widths as multiples of
`\linewidth`, single-paragraph abstract, nine content pages, line numbers,
US Letter, Type 1 fonts, unmodified style parameters.

Two things to know when compiling:

- The submission PDF shows the generic "Submitted to..." footer for every
  track. `neurips_2026.sty` only substitutes the workshop name when the
  `final` option is set, so this is expected, not a misconfiguration.
- `neurips_most_recent.tex` emits one `natbib` error because it loads
  `natbib` while passing `nonatbib` to the class. It falls back to numerical
  style and compiles. The other two variants do not load `natbib`.

## Build

Three `pdflatex` passes to resolve references and the appendix tables:

```bash
pdflatex -interaction=nonstopmode <file>.tex   # x3
```

Build artifacts (`*.aux`, `*.log`, `*.pdf`, `*.synctex.gz`, ...) are
gitignored. `.vscode/settings.json` configures build-on-save.
