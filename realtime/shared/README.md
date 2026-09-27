# Shared realtime runtime

`music_runtime.py` owns audio feature extraction, beat timing, `MusicFrame`,
`RealtimeMusicAnalyzer`, and `AdaptiveMotionController`. The humanoid runtime and
archived arm player both import this module; it has no PyBullet dependency.

Numba caches default to `tmp/numba_cache/` at the repository root. An existing
`NUMBA_CACHE_DIR` environment setting still takes precedence.
