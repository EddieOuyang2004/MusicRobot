# Improving music-to-motion matching in humanoid matcher v2

Analysis date: 2026-09-21. This is a proposal supported by the completed 20-song
audit and a small new offline ablation. No production matcher code was changed.

## Main conclusion

Improve candidate selection, motion compatibility and preparation reliability
before replacing the ONNX encoder. The existing encoder recognizes all ten
catalog controls, while hard style filtering can discard their matching motions.
However, the external-song embedding results are concentrated in Jazz Ballet, so
removing style guidance entirely also produces undesirable concentration.

The objective is an appropriate, rhythmically aligned, feasible dance with useful
variation. Maximizing the number of different motion IDs is not that objective.

## New controlled probe

The probe uses the first 26 seconds of each of the same 20 tracks, before any
audio repetition. It extracts each descriptor once and compares three scorers.
The preceding weak-music streak is reconstructed from saved diagnostic windows.
Baseline acceptance decisions and complete top-track/top-motion lists match the
saved diagnostic at 26 seconds for all 20 songs.

The no-hard-gate comparison preserves the baseline rejection decision; it does
not treat ambient rejection as part of the genre-filter ablation. This is not
the production `legacy` mode proposed for deployment.

| Scoring condition | AIST inputs whose top motion belongs to their own music ID | External rhythmic inputs whose top motion is Jazz Ballet |
|---|---:|---:|
| Current baseline | 5/10 | 1/9 |
| Remove tag-derived genre prior; retain genre filter | 9/10 | 7/9 |
| Remove hard genre filter; preserve baseline rejection | 9/10 | 7/9 |

JB5 remains rejected in all three conditions, explaining the tenth catalog case.
The two ablations happen to give identical top motion recommendations at these sampled windows;
they are not generally equivalent. Same-music association is a useful positive
control, not ground-truth aesthetic quality. External songs have no uniquely
correct motion label. These 20 snapshots identify mechanisms and tradeoffs; they
do not demonstrate an improvement across entire songs or unseen music.

Probe script: `tmp/probe_matcher_improvements.py`.
Raw results: `realtime/humanoid_robot/src/test/output/v2_selection_audit_20/improvement_probe/probe.json`.

## Priority 1: preserve strong music matches with soft style guidance

Current behavior in `music_motion_catalog.py`:

- Track similarity combines embedding, rhythm/timbre and tags with weights
  0.70, 0.20 and 0.10.
- Genre aggregation averages the three leading tracks of each genre, which can
  dilute a very strong individual track match.
- Genre preference then combines normalized similarity (0.35) with the manually
  mapped tag prior (0.65). Min-max normalization discards the absolute scale of
  the genre scores; a normalized winner is not a calibrated probability.
- Only the winning genre, plus a nearly tied second genre, can contribute
  motions. Generic Electronic tags have a fallback contribution to PO.

Proposed candidate pool: take the union of the strongest individual music-track
candidates and candidates supported by plausible style families. Do not allow a
single tag-derived genre to erase a strong individual match. Bound per-track
candidate counts so many near-duplicate motions cannot consume the whole pool.

Use a soft, bounded style bonus within this pool. Separate track-recognition
confidence from style confidence and danceability; the current `confidence`
combination of beat quality and genre margin does not estimate recognition
accuracy. Calibrate any high-confidence branch with absolute similarity, the
runner-up margin and held-out query distributions. Do not set a threshold by
examining only the ten recognizable catalog controls.

Test tag-prior weights 0, 0.2, 0.4 and the existing 0.65 as an initial diagnostic
sweep, separately from hard versus soft filtering. These are test values, not
claimed optimal settings. Freeze temporal windows, rejection, motion weights and
catalog contents during this comparison.

Success: preserve relevant known-track candidates, improve blinded external-song
ratings and avoid merely replacing Popping concentration with Jazz Ballet
concentration. Keep recognition, recommendation and executed-motion metrics
separate.

## Priority 2: match motion timing and character more directly

The current final ranking is 65% music similarity and 35% motion compatibility.
Within compatibility, tempo receives 50%, keypoint compatibility 35% and activity
15% (equivalent to 17.5%, 12.25% and 5.25% of the final score). The keypoint term
compares event density and regularity; it does not align particular accents with
particular musical beats. Density alone can prefer the same rhythmically regular
motion across many unrelated inputs.

Add the following incrementally rather than replacing the entire score at once:

1. Short-window onset strength and rhythmic accent patterns compared with motion
   accents at feasible playback rates. Distinguish phase alignment from event
   density and test half/double-tempo ambiguity explicitly.
2. Energy and energy change, upper/lower-body activity, movement amplitude,
   smoothness and contacts. Use longer history for style and shorter history for
   changing rhythm/energy. Compare, for example, 6-second rhythm and 24-second
   style windows against the current growing 30-second history.
3. Robot-space features from the actual retargeted and processed trajectories.
   The catalog currently profiles source AIST++ keypoints/velocity and separately
   validates GMR motion. Retargeting and smoothing can change the accents the
   viewer sees. Preserve human/source features for style while adding robot
   features for output compatibility; version and rebuild the catalog.

Use a shared score among physically feasible candidates: music relevance plus
rhythm/energy fit plus a calibrated style bonus, minus transition cost and recent
motion repetition. Physical feasibility remains an eligibility condition, not a
penalty that a strong music match can overcome. Normalize score components before
tuning their relative weights. The current bridge search primarily orders by
entry-state cost and uses music relevance for ties, so evaluate ranking quality
both before and after this stage.

## Priority 3: make preparation meet its deadline and recover from holds

Across the 20 saved runtime trials, the median of per-run waveform-to-match p95
times is about 282 ms; the largest per-run p95 is 581 ms. All trials record zero
busy retrieval submissions at the one-second retrieval interval. The median
per-run matcher-only p95 is about 3.15 ms. Therefore nearest-neighbor acceleration
or a faster encoder is not the first fix for the observed freezes.

Candidate readiness reaches 11.18 seconds in Bollywood Groove and 6.63 seconds
in Battle Ready. Both stay in loading until terminal hold and record no entry
scoring work. The code waits for every frozen candidate that has not failed to
become ready before beginning bridge scoring. A slow candidate can therefore
block all otherwise useful candidates.

Recommended changes:

- Cache validated grounding, authored-state and entry features with keys including
  motion content, robot model, limits and preprocessing version.
- Prepare the strongest candidates first and begin scoring a sufficient ready
  subset, instead of waiting for the whole shortlist. Use an explicit deadline
  based on remaining authored time and possible playback speed.
- Refresh candidates while preparation is still speculative when music evidence
  changes substantially; preserve a committed feasible bridge to avoid churn.
- Allow bounded recovery attempts after a terminal hold. Replan from the actual
  held output state and its derivatives, retaining collision and dynamics checks.
  Simply clearing `self.held` or applying a late plan from the old trajectory is
  not a valid recovery implementation.
- Record separate preparation, missing-artifact, infeasibility, stale-plan and
  deadline-miss reasons. Assert that an accepted/relevant ready candidate is not
  indefinitely blocked by an unrelated slow candidate.

Reproduce the two zero-switch songs first, then repeat the runtime suite. Track
hold duration, readiness latency, successful transitions and constraints together;
more transitions alone do not establish a better result.

## Priority 4: separate ambient evidence from a hard music rejection

The current rule can reject a window after two high ambient-related tag scores,
even with useful rhythm or a recognizable catalog match. JB5 was rejected 55/57
times; Hippety Hop 10/57; Isolation Waltz 24/57. The ambient control was correctly
rejected 57/57, so indiscriminately disabling this gate is not justified.

Calibrate a music/danceability decision from multiple signals: raw activity and
signal level, beat/onset evidence, tag evidence, persistence and retrieval
confidence. Distinguish uncertain audio from clear silence/non-music and permit
appropriate low-energy movement when supported. Use hysteresis over time and
evaluate recovery separately from rejection. Tune against real music, ambient,
speech, noise and silence; retain the current negative-set tests. Report false
rejection per song and false acceptance per negative recording, not per frame.

## Priority 5: expand and rebalance the catalog, then learn compatibility

Discogs EffNet is trained for 400 music-style labels, not paired robot-dance
appropriateness. That explains why its tags are useful evidence without making
them authoritative dance-style labels. This is a task mismatch, not proof the
encoder is defective. See [Essentia model documentation](https://essentia.upf.edu/models.html).

The 83.7% Jazz Ballet share of external embedding-only top tracks motivates
testing current catalog-standardized cosine against plain L2-normalized cosine
and regularized normalization. Measure tracks that become nearest neighbors for
many unrelated queries. Treat excessive nearest-neighbor frequency as a symptom
to investigate, not a reason to blindly penalize popular candidates.

FineDance preparation already exists locally. Its paired music/motion and finer
style annotations can broaden the candidate collection and later support a
lightweight learned compatibility model. The dataset introduces 22 fine-grained
genres; see the [official FineDance paper](https://arxiv.org/abs/2212.03741).

For integration:

- Preserve original recording identity, dataset source and native labels. Do not
  force every new dance into the existing ten-class hand-written mapping.
- Group segments/duplicates by original music recording for retrieval, training,
  weighting and splits. A recording with 50 segments must not have 50 times the
  influence merely because it was segmented more densely.
- Fit normalization and any learned mappings only on the training partition,
  then freeze them. The existing leave-music-out diagnostic retains statistics
  fitted to the original catalog and is not a fully independent held-out test.
- Verify synchronization, retargeting and output-space motion descriptors before
  indexing new motions.

A later ranker could learn music-motion compatibility from paired examples plus
hard negatives that share tempo but differ in style or accent structure. Allow
multiple acceptable motions per song: absence of an authored pairing does not
automatically make another motion a negative. Blinded human pairwise preferences
can evaluate and refine the ranking without demanding a single correct dance.

## Evaluation sequence

1. Keep the original 20-song suite as a fixed diagnostic/regression set. Use new
   source recordings for validation and a separate untouched final test set.
   Group all windows, segments and alternate versions of one recording together.
2. Cache complete audio descriptors, not just top-five labels/results, so scoring
   ablations share identical inputs and need no repeated ONNX inference.
3. Run isolated ablations: tag prior, genre pool, rejection, motion features,
   feature normalization. Change one mechanism at a time; combine only promising
   changes after inspecting tradeoffs.
4. Rate external-song outputs blindly for style appropriateness, energy fit,
   rhythmic fit and unwanted repetition. Record candidate Recall@K on labeled
   controls, beat/phase alignment, song-weighted motion-family concentration,
   rejection errors, transition success, holds and latency. Do not optimize
   entropy alone.
5. Test complete playback with several startup seeds on the validation split,
   including both steady songs and style/tempo changes. Bootstrap or otherwise
   aggregate by original song, never by overlapping windows as independent units.

Suggested implementation order: soft candidate selection and preparation/recovery
as separate changes; rejection calibration next; temporal/robot motion features;
then FineDance catalog expansion and a learned compatibility model. No numeric
improvement target or specific weight is established as validated by this probe.
