# BeatWeaver draft: evidence and authoring notes

Inspected on 2026-09-18. Current source code takes precedence over older notes and the thesis. This document is outside the manuscript and is not an experimental report.

## Reading base

The thesis sources in `docs/thesis/`, especially Chapters 1--5 and the limitations in Chapter 7, supply the motivation, data preparation, retrieval design, and earlier architecture. Chapter 6 is deliberately not a results source for this draft. The project also contains an archived offline pipeline and robot-arm work; the paper focuses on the current humanoid pipeline, as requested.

## Claim-to-code map

| Paper content | Primary project evidence |
| --- | --- |
| Audio normalization, embeddings, DSP, segment/track/genre/motion scores | `realtime/humanoid_robot/src/music_motion_catalog.py`: `AudioFeatureExtractor`, `OnnxEffnetBackend`, `MusicMotionMatcher` |
| Causal growing history, 2-second start, 30-second cap, retrieval scheduling | `realtime/humanoid_robot/src/realtime_music_humanoid_matcher_v2.py`: `RetrievalAudioHistory`, `available_history`, `RetrievalWorker`, main loop |
| First-20%-of-target entry policy, last-two-seconds exit search | matcher v2: `select_motion_entry`, `prepare_state_bridge`; `src/test/v2_entry_window_20260916.md` |
| Nonlinear pose/velocity/contact/root penalty, fixed tolerances | matcher v2: `EntryScoringConfig`, `entry_difference_penalty`, `select_motion_entry` |
| Frozen shortlist, checked replay, expanded retry, holds, stale/late plans | matcher v2: `musical_shortlist`, `AuthoredPlayback.prepare`, `AuthoredPlayback.sample` |
| Quintic state bridge, duration refinement, optional Ruckig | `realtime/humanoid_robot/src/motion_bridges.py`: `coefficients`, `HermiteBridge`, `make_bridge` |
| Sampled rejection, Bernstein acceptance/subdivision, extrema fallback | `realtime/humanoid_robot/src/hermite_bounds.py`: `classify`; `HermiteBridge.within_limits` |
| Boundary fade, authored-speed approach, bounded clock integration | `motion_bridges.py`: `boundary_weight`, `AuthoredClock`; matcher v2 `AuthoredPlayback.sample` |
| Collision-aware positional guide, pipeline version and state digest | `realtime/humanoid_robot/src/gmr_collision_projection.py` |
| Receding-horizon state optimization, normalized derivative tracking, soft jerk cost, four-subinterval bounds, continuation and failure policy | `realtime/humanoid_robot/src/gmr_state_trajectory.py`: `solve_horizon`, `plan_state_trajectory`, `subdivided_bernstein_matrix` |
| Regeneration from original SMPL, source provenance and separate dataset selection | `realtime/humanoid_robot/src/build_gmr_v2.py`; `docs/gmr_v2_batch.md` |
| Saved state use and joint mapping | `realtime/humanoid_robot/src/realtime_music_humanoid_dancer.py`; matcher v2 `build_motion_entry_features`, `authored_features` |

## Important differences from the thesis and older development notes

1. Matcher v2's default state-bridge path is not the thesis's moving-clip smoothstep blend. A bridge satisfies authored q/v/a boundary states. Root alignment/blending remains separate.
2. V2 uses a growing causal audio history up to 30 seconds, with analysis permitted after 2 seconds. Catalogue references remain six-second windows. These should not be described as identical-duration query/reference windows.
3. Current candidate entries are every valid authored frame in the first 20% of the clip. Old four-/eight-second remaining-duration rules and key-pose-neighbourhood-only entry search are obsolete here.
4. Source exits are reachable valid authored frames in the final two seconds, with early commitment before fades. This supersedes terminal-only descriptions in early v2 notes.
5. Consecutive-winner hysteresis and bar-triggered switching from v1 are not the default v2 state-bridge selection lifecycle, even though inherited options and policy objects remain in the script.
6. Pair cost is `(0.45*pose + 0.20*velocity + 0.20*contact + 0.10*root)/0.95`. Each component uses the nonlinear normalized penalty. Acceleration constrains bridge feasibility; it is not another pair-cost term. Salience is not an additional score in this v2 path.
7. State-bridge search has a default maximum of 10 seconds. The inherited `--transition-max-seconds 1.20` is not the default Hermite bridge's search ceiling.
8. The default v2 foreground rate is 60 Hz. Neither a 60 Hz configuration nor inherited thesis measurements establish current end-to-end performance.
9. GMR v2 and matcher v2 are separate versions. The former uses pipeline version 5 and projection revision 2. The matcher still requires explicit selection of the corrected dataset root.
10. Current offline smoothing carries q/v/a and stores exact knot derivatives. Fitting playback splines after checking only linear pose interpolation would not provide the same assurance. The current planner also contains subinterval Bernstein bounds and tight DAQP tolerances; older diagnostic notes describing these changes as uninstalled are superseded by the inspected source.
11. Offline jerk is a five-point mean-squared physical-time regularizer (scaled by `smoothing_weight * 1e-9`), not a hard jerk limit or an exact integrated-jerk optimum. Equation (7) describes this objective up to a common factor and the small numerical diagonal regularizer.
12. Default Hermite bridge jerk limits are infinite unless configured; Ruckig requires finite limits. C2 does not mean continuous jerk.
13. The online shortlist's full nonfailed candidate set is awaited by preparation; the draft does not claim an implemented ready-first partial-shortlist policy. Pair search is background work, not an operation guaranteed to finish within a foreground tick.
14. A failed bridge search can leave a terminal hold. There is no claim of guaranteed recovery or indefinite uninterrupted dancing, and a nonzero terminal derivative need not be preserved on entering a hold.

## Scope of evidence

- The manuscript describes implemented mechanisms, not measured performance gains.
- No development benchmark, selection audit, pilot dataset count, or thesis result is imported into the Experiments/Results placeholders.
- Do not infer that all 411 original motions have been successfully regenerated under the current GMR v2 implementation. Dataset completion must be established from the final manifest and artifact provenance for the eventual experiment.
- Authored within-clip joint bounds, online bridge bounds, and final emitted-pose checks are different validation stages.
- Collision checks are sampled and model-specific. The bridge solver checks joint polynomials; contact/root ranking does not certify collision-free or dynamically feasible transitions.
- Time warping, modulation, downstream output limiting, and runtime collision handling may change the planned reference. Their effects belong in final-output evaluation.
- MuJoCo is used for kinematic playback and geometry here. No physical G1, torque tracking, balance, friction, or actuator certification is claimed.
- Root derivatives are not covered by the joint C2 construction.

## Primary literature and formatting sources

References were checked against primary papers, author sites, or proceedings rather than copied blindly from the thesis bibliography.

- [IEEE conference authoring tools and templates](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/): standard IEEEtran conference layout.
- [AI Choreographer, CVF proceedings](https://openaccess.thecvf.com/content/ICCV2021/html/Li_AI_Choreographer_Music_Conditioned_3D_Dance_Generation_With_AIST_ICCV_2021_paper.html): CVF pagination 13401--13412 is used. The thesis used the differently paginated IEEE record.
- [Bailando](https://arxiv.org/abs/2203.13055).
- [EDGE](https://arxiv.org/abs/2211.10658).
- [DiscoForcing](https://arxiv.org/abs/2605.28491).
- [RoboPerform / Do You Have Freestyle?](https://arxiv.org/abs/2512.23650): cite the identifiable arXiv work, first submitted in 2025, rather than carrying over unverified conference page numbers. Updated versions appeared in 2026.
- [Discogs representation learning, ISMIR 2022](https://archives.ismir.net/ismir2022/paper/000099.pdf).
- [librosa, SciPy proceedings](https://proceedings.scipy.org/articles/Majora-7b98e3ed-003.pdf): conference paper, pages 18--24.
- [Motion Graphs, authors' publication page](https://graphics.cs.wisc.edu/Papers/2002/KGP02/): original 2002 publication, rather than its 2023 reprint.
- [Learned Motion Matching, author page](https://theorangeduck.com/page/learned-motion-matching).
- [SMPL, official model page](https://smpl.is.tue.mpg.de/): original 2015 publication, rather than its 2023 reprint.
- [GMR / Retargeting Matters](https://arxiv.org/abs/2510.02252).
- [Ruckig, RSS 2021 proceedings](https://www.roboticsproceedings.org/rss17/p015.html).
- [MuJoCo](https://doi.org/10.1109/IROS.2012.6386109).

## Items left to the author

- Confirm target conference, page allowance, anonymous-review rules, final authors/affiliations, and any acknowledgement/funding text.
- Choose experiments and complete the two reserved sections; update the abstract and conclusion only after results exist.
- Freeze the matcher, motion dataset, catalogue, and configuration for evaluation. Distinguish retrieval relevance, within-clip smoothness, bridge behavior, and final emitted motion when interpreting evidence.
- The coefficients, tolerances, and weights in the draft are current software settings, not empirically calibrated or hardware-certified parameters.
