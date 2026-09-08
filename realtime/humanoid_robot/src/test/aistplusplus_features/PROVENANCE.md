# AIST++ feature extractor provenance

The files in this directory are a local, pinned copy of the kinetic and manual
feature implementation used by the official
[`google/aistplusplus_api`](https://github.com/google/aistplusplus_api)
evaluation code.

- Upstream commit: `2dd7b3e946b794fd0081c98e2e2433545abf8b87`
- Upstream paths: `aist_plusplus/features/{kinetic,manual,utils}.py`
- Retrieved: 2026-09-03
- Local changes: imports and formatting only; feature equations, constants,
  thresholds, joint order, frame rates, and the upstream acceleration formula
  are unchanged.

The upstream repository is Apache-2.0 licensed. `LICENSE_BSD` retains the full
embedded BSD licence notice for the Fairmotion-derived implementation.
