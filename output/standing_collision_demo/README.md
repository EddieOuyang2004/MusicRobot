# Standing motion: collision hold annotations

Purple border and COLLISION HOLD indicate a nonzero proposed joint step was suppressed by collision filtering and all 29 final joint speeds are below 1e-6 rad/s. Root motion is excluded and may continue. Purple is not a marker of the specific colliding body pair.

Orange markers indicate saved joint speeds at 9.42478 rad/s (tolerance 1e-5). Orange and purple timelines describe all original 60 Hz intervals; video is sampled at 30 Hz. The status at frame f describes the interval ending at f.

This diagnostic video uses a fresh instrumented run, with trace and trajectory SHA-256 matched. Original dataset files were not overwritten. Compared with the previous artifact, maximum joint difference is 0.001477 rad; 236 intervals in this run are confirmed collision holds. The original artifact had no per-frame collision trace, so new events were not attached to its video.

Normal video: 12 s. Half-speed video: 24 s. Original-time labels are retained in the half-speed version.
