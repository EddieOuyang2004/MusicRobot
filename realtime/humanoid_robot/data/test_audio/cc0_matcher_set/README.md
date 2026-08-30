# CC0 matcher test set

This is a compact out-of-catalog test set for `realtime_music_humanoid_matcher.py`.
It spans regular and syncopated beats, roughly 89--188 BPM, non-4/4 material,
different timbres, and one deliberately weak-beat negative control.

All tracks came from the [FreePD music library](https://en.freepd.cn/music),
whose library page identifies the files as CC0/public-domain music. The local
`manifest.json` records the original title/path, diagnostic tempo estimate, and
SHA-256 checksum for each file. The tempo values are estimates rather than
ground-truth labels; in particular, beat trackers can invent a tempo for the
ambient negative-control track.

Run one track from the repository root:

```powershell
python realtime/humanoid_robot/src/realtime_music_humanoid_matcher.py `
  --audio-input "realtime/humanoid_robot/data/test_audio/cc0_matcher_set/01_electronic_backbeat.mp3" `
  --headless --realtime --max-seconds 30
```

Re-download the complete set with:

```powershell
& realtime/humanoid_robot/data/test_audio/download_cc0_matcher_set.ps1
```

Suggested test order:

1. `01`, `05`, and `09`: normal steady/mid-tempo cases.
2. `02`, `03`, and `06`: syncopated or fast stress cases.
3. `04` and `07`: timbre and dynamics outside the usual dance-music range.
4. `08`: triple-meter challenge for a four-beat switching policy.
5. `10`: negative control; motion switching should ideally be conservative when
   beat confidence/onset density is low.
