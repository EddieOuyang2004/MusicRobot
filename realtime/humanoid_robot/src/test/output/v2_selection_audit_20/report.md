# Matcher v2: 20-song selection audit

Completed songs: 20/20.

Shares give equal weight to each song with observations; missing/rejected songs are reported separately.
Startup motion is excluded from play counts. Dwell includes holds and attributes blends to the current source motion.
The embedding-only ranking is a diagnostic track ranking, not the full motion policy. AIST catalog songs are in-sample; use leave-music-out and CC0 separately.

| Group | Metric | Songs with observations | Unique IDs | Top share | Leading ID |
|---|---|---:|---:|---:|---|
| all | live.combined_track_all | 20/20 | 16 | 23.1% | JB5 |
| all | live.motion_top1_accepted | 19/20 | 29 | 17.1% | gPO_sBM_cAll_d12_mPO4_ch02 |
| all | live.genre_accepted | 19/20 | 8 | 31.8% | PO |
| all | diagnostic.embedding_track_all | 20/20 | 16 | 16.8% | JB5 |
| all | diagnostic.combined_track_all | 20/20 | 16 | 22.0% | JB5 |
| all | diagnostic.motion_top1_accepted | 20/20 | 26 | 16.1% | gPO_sBM_cAll_d12_mPO4_ch02 |
| all | diagnostic.genre_accepted | 20/20 | 8 | 39.3% | PO |
| all | diagnostic_first_pass.embedding_track_all | 20/20 | 16 | 16.8% | JB5 |
| all | diagnostic_first_pass.combined_track_all | 20/20 | 16 | 22.0% | JB5 |
| all | diagnostic_first_pass.motion_top1_accepted | 20/20 | 26 | 15.7% | gPO_sBM_cAll_d12_mPO4_ch02 |
| all | diagnostic_first_pass.genre_accepted | 20/20 | 8 | 38.6% | PO |
| all | runtime.completed_plays_excluding_startup | 17/20 | 40 | 7.9% | gPO_sBM_cAll_d12_mPO4_ch02 |
| all | runtime.dwell_after_first_switch | 17/20 | 40 | 7.5% | gPO_sBM_cAll_d12_mPO4_ch02 |
| all | leave_music_out.embedding_track_all | 10/10 | 15 | 17.0% | JS1 |
| all | leave_music_out.combined_track_all | 10/10 | 20 | 13.0% | MH4 |
| all | leave_music_out.motion_top1_accepted | 10/10 | 17 | 23.7% | gPO_sBM_cAll_d12_mPO4_ch02 |
| all | leave_music_out.genre_accepted | 10/10 | 7 | 54.3% | PO |
| all | leave_music_out_first_pass.embedding_track_all | 10/10 | 15 | 17.8% | JS1 |
| all | leave_music_out_first_pass.combined_track_all | 10/10 | 19 | 11.2% | WA0 |
| all | leave_music_out_first_pass.motion_top1_accepted | 10/10 | 17 | 22.6% | gPO_sBM_cAll_d12_mPO4_ch02 |
| all | leave_music_out_first_pass.genre_accepted | 10/10 | 7 | 53.3% | PO |
| cc0_music | live.combined_track_all | 9/9 | 8 | 29.0% | JB5 |
| cc0_music | live.motion_top1_accepted | 9/9 | 19 | 20.4% | gPO_sBM_cAll_d12_mPO4_ch02 |
| cc0_music | live.genre_accepted | 9/9 | 6 | 32.0% | WA |
| cc0_music | diagnostic.embedding_track_all | 9/9 | 7 | 23.3% | JB1 |
| cc0_music | diagnostic.combined_track_all | 9/9 | 8 | 26.7% | JB5 |
| cc0_music | diagnostic.motion_top1_accepted | 9/9 | 19 | 20.3% | gPO_sBM_cAll_d12_mPO4_ch02 |
| cc0_music | diagnostic.genre_accepted | 9/9 | 6 | 30.4% | WA |
| cc0_music | diagnostic_first_pass.embedding_track_all | 9/9 | 7 | 23.3% | JB1 |
| cc0_music | diagnostic_first_pass.combined_track_all | 9/9 | 8 | 26.7% | JB5 |
| cc0_music | diagnostic_first_pass.motion_top1_accepted | 9/9 | 19 | 20.3% | gPO_sBM_cAll_d12_mPO4_ch02 |
| cc0_music | diagnostic_first_pass.genre_accepted | 9/9 | 6 | 30.4% | WA |
| cc0_music | runtime.completed_plays_excluding_startup | 7/9 | 22 | 8.6% | gPO_sBM_cAll_d12_mPO4_ch01 |
| cc0_music | runtime.dwell_after_first_switch | 7/9 | 22 | 9.4% | gPO_sBM_cAll_d12_mPO4_ch01 |
| cc0_music | leave_music_out.embedding_track_all | 0/0 | 0 | n/a | none |
| cc0_music | leave_music_out.combined_track_all | 0/0 | 0 | n/a | none |
| cc0_music | leave_music_out.motion_top1_accepted | 0/0 | 0 | n/a | none |
| cc0_music | leave_music_out.genre_accepted | 0/0 | 0 | n/a | none |
| cc0_music | leave_music_out_first_pass.embedding_track_all | 0/0 | 0 | n/a | none |
| cc0_music | leave_music_out_first_pass.combined_track_all | 0/0 | 0 | n/a | none |
| cc0_music | leave_music_out_first_pass.motion_top1_accepted | 0/0 | 0 | n/a | none |
| cc0_music | leave_music_out_first_pass.genre_accepted | 0/0 | 0 | n/a | none |
| cc0_ambient | live.combined_track_all | 1/1 | 1 | 100.0% | JB5 |
| cc0_ambient | live.motion_top1_accepted | 0/1 | 0 | n/a | none |
| cc0_ambient | live.genre_accepted | 0/1 | 0 | n/a | none |
| cc0_ambient | diagnostic.embedding_track_all | 1/1 | 2 | 96.7% | JB5 |
| cc0_ambient | diagnostic.combined_track_all | 1/1 | 1 | 100.0% | JB5 |
| cc0_ambient | diagnostic.motion_top1_accepted | 1/1 | 1 | 100.0% | gPO_sBM_cAll_d10_mPO0_ch01 |
| cc0_ambient | diagnostic.genre_accepted | 1/1 | 1 | 100.0% | PO |
| cc0_ambient | diagnostic_first_pass.embedding_track_all | 1/1 | 2 | 96.7% | JB5 |
| cc0_ambient | diagnostic_first_pass.combined_track_all | 1/1 | 1 | 100.0% | JB5 |
| cc0_ambient | diagnostic_first_pass.motion_top1_accepted | 1/1 | 1 | 100.0% | gPO_sBM_cAll_d10_mPO0_ch01 |
| cc0_ambient | diagnostic_first_pass.genre_accepted | 1/1 | 1 | 100.0% | PO |
| cc0_ambient | runtime.completed_plays_excluding_startup | 0/1 | 0 | n/a | none |
| cc0_ambient | runtime.dwell_after_first_switch | 0/1 | 0 | n/a | none |
| cc0_ambient | leave_music_out.embedding_track_all | 0/0 | 0 | n/a | none |
| cc0_ambient | leave_music_out.combined_track_all | 0/0 | 0 | n/a | none |
| cc0_ambient | leave_music_out.motion_top1_accepted | 0/0 | 0 | n/a | none |
| cc0_ambient | leave_music_out.genre_accepted | 0/0 | 0 | n/a | none |
| cc0_ambient | leave_music_out_first_pass.embedding_track_all | 0/0 | 0 | n/a | none |
| cc0_ambient | leave_music_out_first_pass.combined_track_all | 0/0 | 0 | n/a | none |
| cc0_ambient | leave_music_out_first_pass.motion_top1_accepted | 0/0 | 0 | n/a | none |
| cc0_ambient | leave_music_out_first_pass.genre_accepted | 0/0 | 0 | n/a | none |
| aist_catalog | live.combined_track_all | 10/10 | 10 | 10.0% | BR0 |
| aist_catalog | live.motion_top1_accepted | 10/10 | 16 | 30.0% | gPO_sFM_cAll_d12_mPO1_ch16 |
| aist_catalog | live.genre_accepted | 10/10 | 6 | 44.8% | PO |
| aist_catalog | diagnostic.embedding_track_all | 10/10 | 10 | 10.0% | BR0 |
| aist_catalog | diagnostic.combined_track_all | 10/10 | 10 | 10.0% | BR0 |
| aist_catalog | diagnostic.motion_top1_accepted | 10/10 | 13 | 30.0% | gPO_sFM_cAll_d12_mPO1_ch16 |
| aist_catalog | diagnostic.genre_accepted | 10/10 | 6 | 52.0% | PO |
| aist_catalog | diagnostic_first_pass.embedding_track_all | 10/10 | 10 | 10.0% | BR0 |
| aist_catalog | diagnostic_first_pass.combined_track_all | 10/10 | 10 | 10.0% | BR0 |
| aist_catalog | diagnostic_first_pass.motion_top1_accepted | 10/10 | 13 | 30.0% | gPO_sFM_cAll_d12_mPO1_ch16 |
| aist_catalog | diagnostic_first_pass.genre_accepted | 10/10 | 6 | 50.6% | PO |
| aist_catalog | runtime.completed_plays_excluding_startup | 10/10 | 25 | 11.4% | gPO_sBM_cAll_d12_mPO4_ch02 |
| aist_catalog | runtime.dwell_after_first_switch | 10/10 | 25 | 11.5% | gPO_sBM_cAll_d12_mPO4_ch02 |
| aist_catalog | leave_music_out.embedding_track_all | 10/10 | 15 | 17.0% | JS1 |
| aist_catalog | leave_music_out.combined_track_all | 10/10 | 20 | 13.0% | MH4 |
| aist_catalog | leave_music_out.motion_top1_accepted | 10/10 | 17 | 23.7% | gPO_sBM_cAll_d12_mPO4_ch02 |
| aist_catalog | leave_music_out.genre_accepted | 10/10 | 7 | 54.3% | PO |
| aist_catalog | leave_music_out_first_pass.embedding_track_all | 10/10 | 15 | 17.8% | JS1 |
| aist_catalog | leave_music_out_first_pass.combined_track_all | 10/10 | 19 | 11.2% | WA0 |
| aist_catalog | leave_music_out_first_pass.motion_top1_accepted | 10/10 | 17 | 22.6% | gPO_sBM_cAll_d12_mPO4_ch02 |
| aist_catalog | leave_music_out_first_pass.genre_accepted | 10/10 | 7 | 53.3% | PO |

A review flag means >=20% song-balanced share and presence in >=5 songs (or every song in a smaller group). It is a descriptive flag, not statistical significance or proof of a model defect.
Inspect selection_frequencies.csv and per_song.csv, then compare diagnostic embedding tracks, combined tracks, motion recommendations and completed plays. Nonuniform catalog sizes, style/tempo compatibility, loop boundaries and transition feasibility can all affect concentration.
Runtime uses one fixed startup seed and the normal file-input path; it is not a live-microphone or hardware trial. A held startup with no switches is not evidence that ONNX selected that motion.
