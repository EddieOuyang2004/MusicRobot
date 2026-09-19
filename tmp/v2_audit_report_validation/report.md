# Matcher v2: 20-song selection audit

Completed songs: 1/20.

Shares give equal weight to each song with observations; missing/rejected songs are reported separately.
Startup motion is excluded from play counts. Dwell includes holds and attributes blends to the current source motion.
The embedding-only ranking is a diagnostic track ranking, not the full motion policy. AIST catalog songs are in-sample; use leave-music-out and CC0 separately.

| Group | Metric | Songs with observations | Unique IDs | Top share | Leading ID |
|---|---|---:|---:|---:|---|
| all | live.combined_track_all | 1/1 | 1 | 100.0% | HO5 |
| all | live.motion_top1_accepted | 1/1 | 1 | 100.0% | gHO_sBM_cAll_d20_mHO5_ch10 |
| all | live.genre_accepted | 1/1 | 1 | 100.0% | HO |
| all | diagnostic.embedding_track_all | 1/1 | 1 | 100.0% | HO5 |
| all | diagnostic.combined_track_all | 1/1 | 1 | 100.0% | HO5 |
| all | diagnostic.motion_top1_accepted | 1/1 | 1 | 100.0% | gHO_sBM_cAll_d20_mHO5_ch10 |
| all | diagnostic.genre_accepted | 1/1 | 1 | 100.0% | HO |
| all | diagnostic_first_pass.embedding_track_all | 1/1 | 1 | 100.0% | HO5 |
| all | diagnostic_first_pass.combined_track_all | 1/1 | 1 | 100.0% | HO5 |
| all | diagnostic_first_pass.motion_top1_accepted | 1/1 | 1 | 100.0% | gHO_sBM_cAll_d20_mHO5_ch10 |
| all | diagnostic_first_pass.genre_accepted | 1/1 | 1 | 100.0% | HO |
| all | runtime.completed_plays_excluding_startup | 0/1 | 0 | n/a | none |
| all | runtime.dwell_after_first_switch | 0/1 | 0 | n/a | none |
| cc0_music | live.combined_track_all | 1/1 | 1 | 100.0% | HO5 |
| cc0_music | live.motion_top1_accepted | 1/1 | 1 | 100.0% | gHO_sBM_cAll_d20_mHO5_ch10 |
| cc0_music | live.genre_accepted | 1/1 | 1 | 100.0% | HO |
| cc0_music | diagnostic.embedding_track_all | 1/1 | 1 | 100.0% | HO5 |
| cc0_music | diagnostic.combined_track_all | 1/1 | 1 | 100.0% | HO5 |
| cc0_music | diagnostic.motion_top1_accepted | 1/1 | 1 | 100.0% | gHO_sBM_cAll_d20_mHO5_ch10 |
| cc0_music | diagnostic.genre_accepted | 1/1 | 1 | 100.0% | HO |
| cc0_music | diagnostic_first_pass.embedding_track_all | 1/1 | 1 | 100.0% | HO5 |
| cc0_music | diagnostic_first_pass.combined_track_all | 1/1 | 1 | 100.0% | HO5 |
| cc0_music | diagnostic_first_pass.motion_top1_accepted | 1/1 | 1 | 100.0% | gHO_sBM_cAll_d20_mHO5_ch10 |
| cc0_music | diagnostic_first_pass.genre_accepted | 1/1 | 1 | 100.0% | HO |
| cc0_music | runtime.completed_plays_excluding_startup | 0/1 | 0 | n/a | none |
| cc0_music | runtime.dwell_after_first_switch | 0/1 | 0 | n/a | none |
| cc0_ambient | live.combined_track_all | 0/0 | 0 | n/a | none |
| cc0_ambient | live.motion_top1_accepted | 0/0 | 0 | n/a | none |
| cc0_ambient | live.genre_accepted | 0/0 | 0 | n/a | none |
| cc0_ambient | diagnostic.embedding_track_all | 0/0 | 0 | n/a | none |
| cc0_ambient | diagnostic.combined_track_all | 0/0 | 0 | n/a | none |
| cc0_ambient | diagnostic.motion_top1_accepted | 0/0 | 0 | n/a | none |
| cc0_ambient | diagnostic.genre_accepted | 0/0 | 0 | n/a | none |
| cc0_ambient | diagnostic_first_pass.embedding_track_all | 0/0 | 0 | n/a | none |
| cc0_ambient | diagnostic_first_pass.combined_track_all | 0/0 | 0 | n/a | none |
| cc0_ambient | diagnostic_first_pass.motion_top1_accepted | 0/0 | 0 | n/a | none |
| cc0_ambient | diagnostic_first_pass.genre_accepted | 0/0 | 0 | n/a | none |
| cc0_ambient | runtime.completed_plays_excluding_startup | 0/0 | 0 | n/a | none |
| cc0_ambient | runtime.dwell_after_first_switch | 0/0 | 0 | n/a | none |
| aist_catalog | live.combined_track_all | 0/0 | 0 | n/a | none |
| aist_catalog | live.motion_top1_accepted | 0/0 | 0 | n/a | none |
| aist_catalog | live.genre_accepted | 0/0 | 0 | n/a | none |
| aist_catalog | diagnostic.embedding_track_all | 0/0 | 0 | n/a | none |
| aist_catalog | diagnostic.combined_track_all | 0/0 | 0 | n/a | none |
| aist_catalog | diagnostic.motion_top1_accepted | 0/0 | 0 | n/a | none |
| aist_catalog | diagnostic.genre_accepted | 0/0 | 0 | n/a | none |
| aist_catalog | diagnostic_first_pass.embedding_track_all | 0/0 | 0 | n/a | none |
| aist_catalog | diagnostic_first_pass.combined_track_all | 0/0 | 0 | n/a | none |
| aist_catalog | diagnostic_first_pass.motion_top1_accepted | 0/0 | 0 | n/a | none |
| aist_catalog | diagnostic_first_pass.genre_accepted | 0/0 | 0 | n/a | none |
| aist_catalog | runtime.completed_plays_excluding_startup | 0/0 | 0 | n/a | none |
| aist_catalog | runtime.dwell_after_first_switch | 0/0 | 0 | n/a | none |

A review flag means >=20% song-balanced share and presence in >=5 songs (or every song in a smaller group). It is a descriptive flag, not statistical significance or proof of a model defect.
Inspect selection_frequencies.csv and per_song.csv, then compare diagnostic embedding tracks, combined tracks, motion recommendations and completed plays. Nonuniform catalog sizes, style/tempo compatibility, loop boundaries and transition feasibility can all affect concentration.
Runtime uses one fixed startup seed and the normal file-input path; it is not a live-microphone or hardware trial. A held startup with no switches is not evidence that ONNX selected that motion.
