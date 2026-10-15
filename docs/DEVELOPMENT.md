# Development and release

The initial repository commit is a source-only snapshot of Blender add-on 0.3.1 and the reviewed worker. [source-snapshot.json](../provenance/source-snapshot.json) checks the exact copied runtime/helper files. Original developer services, installed add-ons and study outputs were not replaced during packaging.

## Build and inspect

```bash
python scripts/check_snapshot.py
python scripts/build_packages.py
```

The build script reads an explicit file list. It never gathers models, footage, environments or a whole shared directory. It produces the artist ZIP, a worker-source ZIP with the required relative layout, and SHA256SUMS under ignored `dist/`.

## Basic validation without models

Use a normal Python environment with NumPy/SciPy to generate the synthetic moving-camera fixture, then Blender's bundled Python for import checks:

```bash
python production_bridge/tests/make_fixture.py
/path/to/blender -b -t 2 --factory-startup --python-exit-code 1   --python production_bridge/tests/blender_validate.py --   production_bridge/test_results/synthetic/result.mocap.zip   production_bridge/test_results/synthetic/review
/path/to/blender -b -t 2 --factory-startup --python-exit-code 1   --python production_bridge/tests/blender_reject_invalid.py
```

The screen-splitting test requires a graphical Blender process, **not** `-b`:

```bash
/path/to/blender -t 2 --factory-startup --python-exit-code 1   --python production_bridge/tests/blender_log_view.py
```

It opens a separate factory window and quits on completion. Inspect the generated JSON/error output. Do not run screen split/close operators in background mode. This preserves the workstation's low-concurrency approach: one Blender/inference process at a time, with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2` where appropriate.

For a worker smoke test, use `production_bridge/tests/run_api_job.py` with explicit endpoint, test video, output directory and a short frame count. The production footage/results are deliberately not in this repository.

## Future changes

Use a feature branch and update only the relevant layer. Keep Blender operations on its main thread. Preserve job-specific cancellation, checksum validation, frame mapping, body/world coordinate conventions and native rig correctness. Do not silently change inference/model revisions while changing UI.

This baseline's source-hash check must be deliberately refreshed when runtime source changes; it is not a requirement to keep code frozen forever. Update the version, source manifest, package contents and validation notes together. Runtime and bridge/portability changes need discussion with the project owner before implementation. The proposed facial-control feature has not been started in this snapshot.

For deployment, the engine team mirrors/pins this repository and the reviewed external sources, stages assets via DVC, rebuilds the dedicated environment and validates the chosen feature modes. The current development services still reference their original source paths; changing their deployment pin or paths is a separate reviewed step.

Use the `leob03` GitHub identity for this repository. Authentication belongs in GitHub CLI/SSH credential storage, never in a committed remote URL, config example, log or token file. Do not commit model weights, capture media, generated bundles, personal configs or developer environment binaries.
