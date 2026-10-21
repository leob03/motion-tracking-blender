# Motion Tracking for Blender through ComfyUI

Select a local video in Blender, process one actor on a ComfyUI GPU worker, and import an editable body/finger armature, the fitted mesh, cameras, and the original movie background. **Artist machines need Blender and network access; no CUDA, PyTorch or model files.**

Version **0.6.0** reworks the workflow: image-sequence (EXR/DPX/TIFF/PNG) plates with OCIO-tone-mapped upload proxies, one-time uploads, an actor preview (**Detect Actors**), presets, a job list for queuing several actors, and **Export to Nuke**. A **solved scene camera** (for example from the ViPE Remote Tracking add-on) can drive the job: GVHMR uses its rotations and the body is placed in that camera's world with a depth-preserving plate lock. On the worker, a near-lossless intermediate video, GPU actor detection, a focal-length fix for camera odometry and optional warm models shorten jobs. The 0.5.0 face controls are unchanged. The original reviewed handoff remains at tag `v0.3.1`.

| Mode | Body and world placement | Hands / face |
| --- | --- | --- |
| GVHMR | SMPL-X body, inferred world position and orientation | Manually editable finger bones; no predicted fingers or facial stage |
| SAM + GVHMR | SAM/MHR articulation with GVHMR hip XYZ and root orientation | Optional SAM hand refinement; face off, SAM native, or RealDenseFace + joined FLAME head |

The hybrid does not give SAM independent world recovery. Its Video Fit Camera preserves the selected actor's projection; it is not a shared environment matchmove.

## Start here

- **Artist:** [Blender installation, use and editing](docs/BLENDER.md)
- **Worker developer:** [Current worker setup and external dependencies](docs/WORKER_SETUP.md)
- **Geronimo / DCC bridge:** [Existing API, adapter boundaries and integration discussion](docs/BRIDGE_HANDOFF.md)
- **Models / DVC:** [Exact assets, download links and required paths](docs/MODELS.md)
- **Offline / container build:** [Environment, CUDA builds and remaining validation](docs/OFFLINE.md)
- **Data conventions:** [Result bundle](docs/BUNDLE.md)
- **Checks:** [Validation and limitations](docs/VALIDATION.md)
- **Maintainer:** [Build, change and release workflow](docs/DEVELOPMENT.md)

Build the current artist add-on and worker packages using ordinary Python 3:

```bash
python scripts/build_packages.py
```

Install `dist/mocap_comfy-0.6.0.zip` in Blender. The separate `mocap_worker_snapshot-0.6.0.zip` preserves the worker/helper directory layout; it is **not** a standalone custom-node ZIP. Follow the worker setup guide to install its custom-node symlink.

## What this repository preserves

The Blender add-on, ComfyUI output node, worker/fusion/rig/face code, and five study helper modules are included. Current source hashes are recorded in [source-snapshot.json](provenance/source-snapshot.json). Third-party repositories, weights, Python environments, generated captures and production footage are excluded.

**The server is not yet a self-contained production distribution.** It currently requires two different ComfyUI source trees, the reviewed GVHMR node port, study-relative assets, fixed development paths, and an inherited Python environment. The handoff documents these rather than silently changing the tested runtime. A clean rebuild/container, Windows artist session and integration with the team's new generic bridge have not been validated.

RealDenseFace + FLAME results include 100 signed expression shape keys and editable jaw/eye bones, animated from the retained predictions. The joined neck is protected from facial corrections. See [facial editing](docs/FLAME_EDITING.md). SAM-native expressions remain cached. One actor is processed per job. Interactive click/mask selection, multiple actors in one job, production shot lookup and Maya are not part of this adapter yet.

See [third-party provenance](THIRD_PARTY.md). An overall redistribution license for the project's original code has not been selected. Model files are obtained separately under their applicable terms.
