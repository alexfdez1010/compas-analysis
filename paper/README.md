# Journal Article — Overleaf Template

A ready-to-use Overleaf/LaTeX template for manuscript submissions. The
structure is deliberately kept in separate files so that the
journal-mandated formatting (fonts, margins, bibliography style, title
page) stays consistent across every submission, while authors only ever
touch the files inside `user/`.

## Quick start

1. In Overleaf, make sure **`main.tex`** is set as the main document
   (menu → Settings → Main document).
2. Edit `user/userdata.tex` — title, authors, affiliations, email,
   keywords, abstract.
3. Write your manuscript in `user/content.tex`.
4. Add your references to `user/biblio.bib` and cite them with
   `\cite{yourkey}`.
5. Put any figures in `user/figures/` and reference them from
   `user/content.tex` (see the commented example at the top of that
   file).
6. Recompile. On Overleaf this runs `pdflatex → bibtex → pdflatex →
   pdflatex` automatically. If you compile locally, run that same
   four-pass sequence — a single `pdflatex` pass will not resolve your
   citations.

## File structure — what to edit and what not to

```
main.tex                    Entry point. Do not edit.
sections/
  preamble.tex               Journal formatting rules. Do not edit.
  cover.tex                  Title page layout. Do not edit.
references/
  IEEEtranDOI.bst             Bibliography style (IEEEtran + DOI field). Do not edit.
user/
  userdata.tex                EDIT — title, authors, affiliations, email, keywords, abstract
  usercode.tex                EDIT — your own extra packages/macros, if any
  content.tex                 EDIT — the body of your manuscript
  biblio.bib                  EDIT — your reference list (BibTeX format)
  figures/                    EDIT — put your image files here
```

As a rule of thumb: **if a file is not inside `user/`, leave it alone.**
Everything outside `user/` defines the journal's required format, and
changing it risks producing a manuscript that no longer matches the
journal's typesetting standard. Every file also carries the same notice
at its top.

## About the bibliography setup

This template uses BibTeX (not biblatex) with a customized style file,
`references/IEEEtranDOI.bst` — a version of the standard `IEEEtran.bst`
that additionally prints each reference's DOI.

Two things worth knowing:

- **`\bibliographystyle{references/IEEEtranDOI}`** in `main.tex` must
  point to the exact filename of the `.bst` file in `references/`
  (minus the extension). *(This was previously
  `references/IEEEtran`, which did not match the actual file,
  `IEEEtranDOI.bst` — BibTeX would fail with "I couldn't open style
  file references/IEEEtran.bst". That has been corrected in this
  version; if you ever rename the `.bst` file, update this line to
  match.)*
- **`\bibliography{IEEEabrv,user/biblio}`** in `main.tex` references
  two `.bib` databases: your own `user/biblio.bib`, and `IEEEabrv`,
  which supplies abbreviated IEEE journal-name strings. `IEEEabrv.bib`
  is *not* a file in this project — it ships as part of any full TeX
  Live installation (including Overleaf's), so BibTeX finds it
  automatically. You don't need to add it, and you shouldn't remove it
  from that line even though it isn't in the file tree.

This was verified by compiling the project end-to-end (`pdflatex` →
`bibtex` → `pdflatex` → `pdflatex`) with the corrected style path;
citations, in-text references, and the bibliography now resolve
correctly.

## Known quirks worth knowing about

- **`\kwfirst`, `\kwsecond`, `\kwthird`, `\myEmailAddress`, and
  `\myDOI`** are all defined in `user/userdata.tex`, and a
  `\keywords{...}` macro is defined in `sections/preamble.tex` — but
  nothing in `sections/cover.tex` actually calls `\keywords{...}` or
  prints the email/DOI. In the current structure, filling in these
  values does **not** make keywords, the corresponding-author email,
  or the DOI appear anywhere in the compiled PDF. This was left as-is
  since the structure was not to be changed, but flagging it in case
  the journal expects keywords to be visible on the title page — in
  that case, one line would need to be added to `sections/cover.tex`
  (e.g. `\keywords{\kwfirst, \kwsecond, \kwthird}` after the abstract).
- **Draft watermark:** the "DRAFT" watermark on the first page comes
  from `\usepackage[firstpage]{draftwatermark}` in
  `sections/preamble.tex`. Comment out that line (and the
  `\SetWatermarkLightness` line beneath it) once the manuscript is
  ready for final submission.
- **Line numbers:** `sections/cover.tex` turns on continuous line
  numbering (`\linenumbers`) for peer review. Remove that line for a
  camera-ready version if the journal doesn't want line numbers in the
  final PDF.
- **`user/figures/placeholder.jpg`** is a sample image left in place so
  the folder isn't empty; delete it once you've added your own
  figures.

## Compiling locally (outside Overleaf)

If you ever need to compile outside Overleaf, you need a full TeX Live
installation (Overleaf itself already has everything below). From the
project root:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Pre-submission checklist

- [ ] Title, all authors, affiliations, and corresponding email filled
      in `user/userdata.tex`
- [ ] Abstract within the journal's word limit
- [ ] All `[bracketed placeholders]` in `user/content.tex` replaced
      with real text
- [ ] Ethics Statement, CRediT contribution statement, Generative-AI
      declaration, and Competing-interest declaration all completed
- [ ] All citations resolve (no `[?]` in the compiled PDF — if you see
      one, re-run the four-pass compile sequence above)
- [ ] Draft watermark and line numbers removed for the camera-ready
      version, if required
- [ ] Placeholder figure removed / all figures are your own
