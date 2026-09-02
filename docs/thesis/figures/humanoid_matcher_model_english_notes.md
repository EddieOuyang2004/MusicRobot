# English Humanoid Matcher Model Figure

## Suggested figure title

**Hierarchical Music-to-Motion Matching and Beat-Synchronous Humanoid Control**

## Suggested thesis caption

Overview of the proposed realtime Humanoid Matcher. (A) A robust music-representation stage converts a six-second rolling audio window into content, rhythm–timbre, tempo, and activity descriptors, while the realtime analyser supplies beat and low-level feature events. (B) A hierarchical matcher first retrieves catalog tracks by segment-level similarity and Top-3 mean pooling, then ranks preflight-passed AIST++ motions using tempo, keypoint-density, and activity compatibility. (C) A stable selection policy, keypoint-constrained entry search, beat–phase controller, root-aligned blending, and output safety layer generate continuous 29-DoF Unitree G1 poses for MuJoCo kinematic playback.

## Editable formats

- `humanoid_matcher_model_english.drawio`: fully editable diagrams.net source.
- `humanoid_matcher_model_english.svg`: editable vector source for Inkscape, Illustrator, Affinity Designer, or PowerPoint.
- `humanoid_matcher_model_english.png`: rendered preview.

The visual structure follows an academic three-panel model diagram rather than a software flowchart. The formulas and default thresholds are taken from the current implementation. The figure identifies the present DSP embedding as the active catalog backend and Discogs-EffNet as an optional alternative.
