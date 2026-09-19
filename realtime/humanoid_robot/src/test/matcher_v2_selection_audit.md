# Matcher v2: 20-song selection concentration experiment

Purpose: determine whether the same motion IDs are recommended across different
songs, and whether concentration first appears in ONNX feature retrieval,
motion compatibility ranking, or actual v2 playback. This experiment describes
the current matcher; it does not alter its ranking, transitions, or catalog.

## Run

From `C:\Users\Eddie\Desktop\MusicRobot` in PowerShell:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/run_v2_selection_audit.py --stage run
```

The 20-song experiment is prepared but has not been run. A separate eight-second
smoke check validates the harness. Allow at least 20 minutes of audio playback
plus ONNX diagnostics and per-process startup (potentially much longer on a busy
machine). Runs are serial, headless and silent, with no physical robot connection.
Progress appears in the terminal; each song has a runtime log.

The same command resumes completed songs only after verifying their artifact
hashes. An interrupted/failed song is moved into `failed_attempts` and rerun.
Ctrl+C stops the run. Do not edit the matcher, models, catalog, source audio or
GMR files during the experiment: their hashes are checked before starting or
resuming. Existing local matcher changes are included in the frozen snapshot.

Outputs are under `realtime/humanoid_robot/src/test/output/v2_selection_audit_20/`:

- `report.md`: readable concentration summary, generated automatically at completion.
- `selection_frequencies.csv`: ranked IDs, equal-song shares, number of songs containing each ID.
- `per_song.csv`: retrieval counts, rejections and dominant recommendations for each song.
- `runtime_per_song.csv`: completed plays, motion changes, replays and observed duration.
- `report.json`: full aggregate data with an explicit 20/20 completion flag.
- `runs/song_XX/`: command, log, trace, timing, every delivered retrieval and diagnostic rankings.
- `songs.csv`, `manifest.json`, `catalog_inventory.csv`: frozen inputs, audio construction, versions, hashes and library composition.

To rebuild a partial report without running inference:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/run_v2_selection_audit.py --stage analyse
```

To verify inputs and inspect the 20 runtime commands without running them:

```powershell
.\.venv\Scripts\python.exe realtime/humanoid_robot/src/test/run_v2_selection_audit.py --stage dry-run
```

If intentionally changing the implementation or inputs, use a fresh directory
with `--stage prepare --output-dir <new-directory>`, then pass that directory to
`--stage run`. This prevents mixing configurations in one result.

## Fixed sample and controls

Twenty different musical tracks, selected before observing results:

| Group | Tracks |
|---|---|
| External CC0 music (9) | Backbeat; Hippety Hop; BeBop for Joey; Bollywood Groove; Cumbish; Bar Brawl; Battle Ready; Isolation Waltz; Funky Energy Loop |
| External ambient control (1) | Alien Spaceship Atmosphere |
| AIST++ catalog songs (10) | Longest original audio variant in each of BR, HO, JB, JS, KR, LH, LO, MH, PO and WA; exact IDs and filenames in songs.csv |

The ambient track tests rejection/holding and is reported separately from the
nine external rhythmic tracks. Catalog songs are in-sample positive controls,
not evidence of generalization. The deterministic diagnostic additionally removes
**every segment, variant and motion associated with the query music ID** for each
AIST++ song. It retains existing catalog normalization statistics. Actual v2
runtime always uses the unchanged production catalog.

Each runtime receives 60 seconds at 16 kHz mono. Longer audio is cropped from
time zero; shorter audio is repeated without crossfades, gain edits or time
stretching. This provides time for transitions without inventing songs. Repeat
boundaries, original lengths and hashes are recorded. Repetition is a confound:
diagnostic `*_first_pass` summaries use only windows ending before the first loop,
so concentration can also be examined on original audio alone. This is a
convenience sample of local music, not a random sample of all genres.

## Measurements

1. **Deterministic feature diagnostic:** growing causal history from 2 to 30
   seconds, then a trailing 30-second window; one observation every two seconds
   through 60 seconds. Uses v2's extractor/settings, including Discogs EffNet ONNX
   and speed range 0.55–1.3. State resets per song. Records top five embedding-only
   tracks (same standardized similarity and top-three-segment averaging), combined
   track scores, tags, rejection reasons, and top 20 motions with music, tempo,
   keypoint and activity score components.
2. **Actual v2 runtime:** normal file-input mode, one-second retrieval interval,
   60 Hz control, default beat-sync/transitions, require-GMR policy and startup
   seed 20260917. Normal file mode pre-analyses beats; this is not the separate
   experimental causal/live microphone mode. The wrapper records every completed
   retrieval once when delivered, before a same-frame transition can overwrite
   its trace event. It returns the original result unchanged.
3. **Playback:** count each `switch_complete` once, with same-ID replays separate.
   Exclude the initial seeded motion from initial play counts. Time occupancy
   starts after the first completed switch; holds remain included and blends are
   attributed to the current source motion. Songs without a completed switch
   remain visible as having no observations.

The diagnostic uses fixed windows while asynchronous runtime may skip submissions
when busy. These are separate measurements. Track rankings use all retrievals;
motion rankings only use accepted results with compatible motions. Rejections and
no-compatible-motion counts are explicit. Top-five motion statistics count ranking
slots, not actual plays or v2's shortlist.

Within each group, normalize counts within each song, then average over songs with
observations. Report contributing song count, top-one/top-five shares, unique IDs,
inverse concentration (`1 / sum(share^2)`) and cross-song prevalence. Overlapping
windows and 60 Hz frames are **not independent experimental samples**. There are
no frame/window-based p-values. One fixed startup seed controls initial conditions;
this experiment does not estimate variation over startup seeds.

The >=20% share / >=5-song flag is a predeclared descriptive review threshold,
not a significance test (smaller groups require presence in every song). Unequal
numbers of motions per song/genre, style priors, tempo gates, near-ties and transition
feasibility make uniform motion selection an inappropriate null.

## Assessing the suspicion after completion

- Repeated embedding-only **track IDs** across unrelated external songs suggest
  concentration at the feature/catalog similarity stage. ONNX predicts features
  and tags; it does not directly choose a motion ID.
- Diverse embeddings but concentrated combined/style/motion results point toward
  downstream scoring, style priors or compatibility filters. Inspect component
  scores and library inventory before attributing the result to ONNX.
- Diverse recommendations but concentrated completed plays point toward transition
  readiness, fallback/replay behavior or insufficient switching time.
- Compare external music, ambient rejection, in-sample controls, leave-music-out
  diagnostics and first-pass results separately. The pooled number is insufficient.

When finished, tell Codex the experiment is complete. Saved artifacts allow checking
which exact motions dominate, in how many songs, and at which selection stage.
