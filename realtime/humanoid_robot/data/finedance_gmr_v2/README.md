# FineDance GMR v2 motions

This directory contains 2,522 converted Unitree G1 motion files (60 FPS),
about 1.15 GB of binary data. The `.pkl` files are tracked with Git LFS.
`manifest.json` records artifact SHA-256 checksums and conversion metadata;
`train.txt`, `val.txt`, and `test.txt` retain the dataset split lists.
Eight conversion failures are recorded in `failures.json` and have no motion artifact.
Logs, lock files, temporary files, and corrupt checkpoint backups are local only.

## Download after cloning or pulling

Install Git LFS, then run from the repository root:

```sh
git lfs install
git pull
git lfs pull --include="realtime/humanoid_robot/data/finedance_gmr_v2/*.pkl"
```

For a fresh checkout, run `git lfs install` before `git clone` to download
motion files automatically. Git LFS must be available to obtain the binary
contents instead of small pointer files.

This directory contains converted motions, not the source FineDance audio,
SMPL data, or generated combined retrieval catalog. Those assets are still
required separately for building and using the complete music matcher catalog.
Absolute paths in the manifest describe the original conversion environment.
See [conversion documentation](../../../../docs/finedance_gmr_conversion.md)
and [catalog integration](../../../../docs/finedance_matcher_integration.md).
