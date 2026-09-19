# BeatWeaver IEEE conference draft

The draft is `beatweaver.tex`; its bibliography is `references.bib`. The compiled preview is `../../output/pdf/beatweaver.pdf`. `beatweaver-latex.zip` contains the source package for Overleaf.

The paper uses the standard `IEEEtran` conference class: 10-point text, US Letter paper, and two columns. It does not modify margins or squeeze spacing. The architecture figure is editable TikZ inside the LaTeX source. No external images or project data are required to build the paper.

## Scope

Written from the thesis and the current working-tree implementation inspected on 18 September 2026, including uncommitted post-thesis changes. The draft emphasizes:

- Music-conditioned hierarchical retrieval.
- Matcher v2 exit/entry search, nonlinear scoring, checked state-to-state Hermite bridges, and boundary-aware playback.
- GMR v2 state-aware smoothing and reuse of stored position/velocity/acceleration.

Experiments and Results and Discussion are explicit placeholders. No thesis-era evaluation or development benchmark numbers are presented as evidence for the current system. The draft is a methods paper scaffold, not a completed submission.

The author name and UCL affiliation are taken from the thesis. Confirm the final author list, affiliation, email, and target conference requirements before submission. No coauthor, email address, funding statement, conference name, or copyright notice has been invented.

## Build

### Overleaf

Upload `beatweaver-latex.zip` as a new project, select `beatweaver.tex` as the main document, and use pdfLaTeX. Standard Overleaf installations provide IEEEtran, TikZ, and the other packages. The ZIP includes a compiled bibliography fallback, but the editable `.bib` file remains the source of references.

### TeX Live or MiKTeX

From this directory:

```text
latexmk -pdf beatweaver.tex
```

Without latexmk:

```text
pdflatex beatweaver.tex
bibtex beatweaver
pdflatex beatweaver.tex
pdflatex beatweaver.tex
```

### Portable compiler used for this preview

From the repository root in PowerShell:

```powershell
./tmp/paper-tools/tectonic/tectonic.exe --keep-logs --outdir output/pdf docs/paper/beatweaver.tex
```

The portable Tectonic compiler is local build tooling, not part of the submission ZIP. Its first compilation downloads standard TeX packages. The source also builds through the ordinary pdfLaTeX/BibTeX workflow above.

## Editing and provenance

`SOURCE_NOTES.md` maps the methods to code and distinguishes current behavior from older thesis claims. It is an authoring aid, not part of the paper. It also lists the primary reference sources and unresolved submission details.

The main paper is currently five pages, including bibliography and section placeholders. Its eventual length will change when experiments are added; the target conference's page limit has not been specified.
