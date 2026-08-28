# Humanoid matcher full-entrypoint smoke result

Generated: `2026-08-12`

## Configuration

- Entrypoint: `realtime_music_humanoid_matcher.py`
- Original WAV: `gBR_sBM_cAll_d04_mBR0_ch01.wav`
- Input mode: `--audio-input` realtime virtual microphone
- Runtime: headless MuJoCo with `--realtime`
- Match window/interval: production defaults, 6 s / 1 s
- Model: `discogs-effnet-bsdynamic-1.onnx`, explicitly supplied for both
  embedding and tag outputs
- Model SHA-256: `a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c`

## Runs

The entrypoint was run in fresh processes with `--max-seconds` set to 10, 15,
and 30 seconds.

All three runs:

- opened the original WAV through `MatcherFileMicrophoneSource`;
- completed the one-second virtual microphone startup/noise-reset delay;
- loaded the initial AIST++ motion and headless MuJoCo player;
- produced active music status, BPM, RMS/band features, detected beats, and
  accepted-beat controller updates;
- exited normally without an exception.

However, none of these bounded runs printed a `match | ...` result or a
`Pending motion ...` message before exit. Therefore this smoke test passes the
virtual microphone, realtime analyzer, controller, and MuJoCo input path, but
does **not** pass retrieval-result delivery or automatic motion switching in the
complete production loop.

The dedicated virtual-microphone benchmark separately confirms that the same
source, ONNX extractor, matcher, and asynchronous retrieval worker complete
queries correctly. The difference indicates a production-loop timing/shutdown
issue that should be diagnosed before treating full automatic switching as
validated.

## Reproduction command

Run from `realtime/humanoid_robot/data/music_catalog` so the current catalog
motion paths resolve correctly:

```powershell
& 'C:\Users\Eddie\Desktop\MusicRobot\.venv\Scripts\python.exe' `
  'C:\Users\Eddie\Desktop\MusicRobot\realtime\humanoid_robot\src\realtime_music_humanoid_matcher.py' `
  --catalog 'C:\Users\Eddie\Desktop\MusicRobot\realtime\humanoid_robot\data\music_catalog\catalog.json' `
  --embedding-model 'C:\Users\Eddie\Desktop\MusicRobot\realtime\humanoid_robot\models\discogs-effnet-bsdynamic-1.onnx' `
  --tag-model 'C:\Users\Eddie\Desktop\MusicRobot\realtime\humanoid_robot\models\discogs-effnet-bsdynamic-1.onnx' `
  --audio-input 'C:\Users\Eddie\Desktop\MusicRobot\realtime\humanoid_robot\data\aistpp\audio\gBR_sBM_cAll_d04_mBR0_ch01.wav' `
  --headless --realtime --max-seconds 30 --status-interval 5
```
