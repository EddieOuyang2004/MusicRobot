# Thesis review: fixes for a one-hour revision

Reviewed attachment: `C:/Users/Eddie/Downloads/_scale__5_ucl_logo_png (4).pdf` (97 PDF pages). Page references below use the thesis's printed page numbers; add six for the PDF viewer page number. Reviewed the thesis text, selected rendered pages, the relevant implementation, table exporter, and saved experiment results. This was a review: thesis sources and frozen experiment results were not edited.

The main research limitation is that the strongest evidence concerns integration and 60 Hz kinematic playback. The formal experiment does not establish improved musical timing, physical robot control, or the separate benefits of most proposed components. Chapters 6 and 7 already acknowledge this. Preserve those qualifications; the one-hour revision should improve the validity and clarity of the existing evidence.

## Recommended 60-minute schedule

| Minutes | Action |
|---|---|
| 0-20 | Correct fold normalisation and update retrieval results/provenance. |
| 20-30 | Repair the interpretation of Figure 6.5. |
| 30-35 | Remove misleading long-run confidence intervals from Table 6.5. |
| 35-40 | Add units and aggregation definitions to Tables 6.7 and 6.9. |
| 40-45 | Correct the genre name and a few specific wording errors. |
| 45-50 | Replace two historical reprint references with original publication records. |
| 50-60 | Rebuild, check cross-references, and visually inspect changed pages. |

Time estimates assume the existing build environment works. The diagnostic retrieval comparison already ran successfully in this workspace.

## 1. Held-out music still influences retrieval normalisation

**Priority: highest.** Abstract; Sections 6.2 and 6.4; Table 6.2 (printed pp. 61 and 66).

In `realtime/humanoid_robot/src/test/run_humanoid_matcher_experiments.py`, `catalog_without_music()` removes the query music's segments, track and motions, but lines 280-283 copy the original `embedding_mean`, `embedding_std`, `rhythm_mean` and `rhythm_std`. `music_motion_catalog.py` computes those arrays over all catalogue segments (lines 1883-1886), and its matcher uses them for query and reference standardisation (lines 841-849 and 926-936). Candidate exclusion therefore does not establish fully held-out preprocessing.

**Fix:** compute these four statistics from the retained arrays in each fold, using the existing normalisation conventions. Rerun only offline retrieval, save it as a new result with provenance, and update the relevant table/export inputs. Keep the archived evaluation identifiable. Also audit any catalogue-derived motion statistics before describing motion-ranking results as completely fold-isolated. This does not require repeating the 135 runtime runs.

A scratch comparison changed only audio normalisation in memory. It reproduced every aggregate retrieval score printed in the existing thesis before applying the change:

| Metric | Existing global statistics | Fold-only audio statistics |
|---|---:|---:|
| Recall@1 | 0.316667 | 0.316667 |
| Recall@3 | 0.533333 | 0.550000 |
| Recall@5 | 0.783333 | 0.783333 |
| Macro-F1 | 0.271912 | 0.271912 |
| MRR | 0.504755 | 0.506144 |
| NDCG@5 | 0.550108 | 0.551264 |

This is a methodological correction, not evidence that the current scores were substantially inflated. The diagnostic does not replace the frozen results or establish that all tuning was independent of evaluation data.

Diagnostic script: `tmp/pdfs/one_hour_review/check_retrieval_normalisation.py`; results: `tmp/pdfs/one_hour_review/retrieval_normalisation_check.json`.

If the result-export pipeline cannot be updated within the time available, accurately qualify the existing protocol instead: "The held-out music identity is excluded from candidate records, while descriptor normalisation uses the frozen full-catalogue statistics. This is candidate-excluded evaluation with shared preprocessing, rather than fully inductive leave-one-music-out evaluation."

## 2. Figure 6.5 visually implies an impossible response sequence

**Location:** printed p. 74; `chapter6_experimental_evaluation.tex`, figure caption near line 538; plot construction in `scripts/export_chapter6.py`, lines 225-238.

For the change at 7.75 s, the saved response record puts completion at 11.253 s but switch initiation at 15.220 s. These cannot describe the same transition. The existing discussion acknowledges that milestones can refer to different candidates, but the figure's stage labels still invite a sequential interpretation.

**Ten-minute fix:** rename the figure as independent event observations, and explicitly identify the first episode as unmatched. Suggested caption addition:

> Each marker is an independently detected event, not necessarily a stage of the same candidate transition. In the first episode, completion precedes the later initiation marker because transition identity is not linked. This plot therefore does not estimate a complete semantic response latency. Missing markers indicate no qualifying event before the next change or the end of observation.

Mark missing/censored events visibly if time permits. Do not reorder points, substitute another seed, or invent a consistent event chain. A complete technical fix requires linking candidate and transition identities in the analysis; that is a separate task.

## 3. Table 6.5 displays a degenerate interval as precision

**Location:** printed p. 68; `scripts/export_chapter6.py`, lines 107-119.

The F-long row reports BAS `0.335 [0.335, 0.335]` and PFC `1.456 [1.456, 1.456]`. Those source-bootstrap intervals collapse because five seeds share one source. Section 6.3 explains this, but the table remains easy to misread.

**Fix:** retain the means and display the source-level CI as unavailable, with a footnote: "One source, five repeated runs; a source-level confidence interval is not informative." Optionally report observed seed ranges, clearly labelled as ranges rather than CIs. The saved results give BAS 0.3282-0.3433 and PFC 1.1385-1.8189. Do not treat five seeds as five independent music sources.

## 4. Results tables omit information necessary to interpret numbers

**Locations:** Table 6.7, printed p. 70; Table 6.9, printed p. 72. Generated by `scripts/export_chapter6.py`, lines 128-152.

- Table 6.7: add `rad/s` to speed and `rad/s^2` to acceleration. State that speed/acceleration entries are maxima across runs and violation entries are sums across runs.
- Table 6.9: add milliseconds and explain that p50/p95/p99 columns are medians across runs of each run's percentile. "Worst max" is the largest recorded maximum across runs. The exporter confirms this aggregation; they are not percentiles pooled across every event.

Edit the generator and captions so regeneration preserves the correction.

## 5. Correct small but conspicuous terminology and prose problems

- Printed p. 47, Section 5.3.2: "middle- and low-hip-hop families" should be "middle hip-hop (MH) and LA-style hip-hop (LH)." The code maps hip-hop tags to MH and LH (`music_motion_catalog.py`, lines 872-874), and the thesis itself defines LH correctly on pp. 10 and 67. Edit `chapter5_realtime_matching_control.tex:191`.
- Printed p. 17, Section 2.6.2: "uses the term Weaver in a related but more hierarchical sense" follows a discussion of motion matching and appears to be an inappropriate wording substitution. Use "BeatWeaver applies motion matching in a hierarchical retrieval-and-control pipeline." Edit `chapter2_literature_review.tex:478`.
- Printed p. 6: join the fragments beginning "Streaming generation..." and "And retrieval-and-control..." into one parallel sentence (`chapter2_literature_review.tex:20`).
- Printed p. 54: "It improves immediate responsiveness" claims an isolated benefit that the formal evaluation does not test. Change to "It is intended to provide immediate motion variation between retrieval updates" (`chapter5_realtime_matching_control.tex:547`).

## 6. Bibliography uses later reprints for foundational papers

**Location:** references [21] and [22], printed p. 86; `references.bib:108` and `references.bib:116`.

The entries are 2023 *Seminal Graphics Papers* reprints. These are not fabricated publications, but citing the originals makes the historical discussion clearer:

- SMPL: Loper et al., 2015, ACM Transactions on Graphics 34(6), article 248, pp. 248:1-248:16. The [official SMPL site](https://smpl.is.tue.mpg.de/) supplies the authors' BibTeX.
- Motion Graphs: Kovar, Gleicher and Pighin, 2002, ACM Transactions on Graphics 21(3), pp. 473-482. The [authors' publication page](https://graphics.cs.wisc.edu/Papers/2002/KGP02/) supplies the record.

Existing BibTeX keys can be preserved while replacing their metadata, avoiding unnecessary citation edits. Protect acronym capitalisation such as `{SMPL}`, `{AIST++}`, `{GPT}`, and `{MuJoCo}` where appropriate.

## Optional substitutions if one item finishes early

- **Clarify genre Recall@K.** The evaluator deduplicates genre labels. Recall@5 means the correct genre appears among five ranked genres, not five returned motions. A uniformly random permutation of ten genre labels has expected Recall@1 0.10 and Recall@5 0.50. Label this as an analytical chance reference, not an executed retrieval baseline. Existing music-level bootstrap intervals can also be added without another runtime experiment.
- **Table 6.6, pp. 68-69:** the thesis already explains why standardised geometric FID/Div are ill-conditioned. Put that qualification directly in the caption or beside the affected cells, so `1.40e+12` is not mistaken for an interpretable dance-quality measure. Preserve the raw diagnostic values.
- **Introduction, p. 5:** add a short explicit contribution paragraph distinguishing your integration, selection/transition mechanisms, and reproducible evaluation from reused AIST++, Discogs EffNet, GMR and MuJoCo components. State that component improvements remain unestablished unless tested.

## Problems that need more than a one-hour edit

Demonstrating improved beat synchronisation, obtaining reliable 120 Hz operation, adding causal component ablations, improving three zero-Recall@1 genres, testing real microphone conditions, conducting a user study, and validating physical balance all require additional technical work or evidence. The thesis already acknowledges most of these. Keep the limitations rather than weakening them to make the results appear stronger.

Selected PDF pages were visually checked. Apparent corrupted dashes in the initial extraction were a console encoding artifact, not a verified PDF defect. Embedded literature figure extraction also includes text outside PDF crop boxes; that is not evidence that whole paper pages are visibly reproduced.
