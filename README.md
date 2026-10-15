# Motion Tracking for Blender through ComfyUI

Select a local video in Blender, process one actor on a ComfyUI GPU worker, and import an editable body/finger armature, the fitted mesh, cameras, and the original movie background. **Artist machines need Blender and network access; no CUDA, PyTorch or model files.**

This is the reviewed **0.3.1 prototype**, preserved before the next facial-editing and generic DCC-bridge integration work.

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

Download the artist add-on and worker source ZIPs from the [0.3.1 prototype release](https://github.com/leob03/motion-tracking-blender/releases/tag/v0.3.1), or build them using ordinary Python 3:

```bash
python scripts/build_packages.py
```

Install `dist/mocap_comfy-0.3.1.zip` in Blender. The separate `mocap_worker_snapshot-0.3.1.zip` preserves the worker/helper directory layout; it is **not** a standalone custom-node ZIP. Follow the worker setup guide to install its custom-node symlink.

## What this repository preserves

The Blender add-on, ComfyUI output node, worker/fusion/rig/face code, and five study helper modules are included. Runtime source is byte-for-byte copied from the tested prototype; hashes are recorded in [source-snapshot.json](provenance/source-snapshot.json). Third-party repositories, weights, Python environments, generated captures and production footage are excluded.

**The server is not yet a self-contained production distribution.** It currently requires two different ComfyUI source trees, the reviewed GVHMR node port, study-relative assets, fixed development paths, and an inherited Python environment. The handoff documents these rather than silently changing the tested runtime. A clean rebuild/container, Windows artist session and integration with the team's new generic bridge have not been validated.

No new facial control rig is included: captured FLAME expressions remain cached, while body/head/neck and finger bones are editable. One actor is processed per job. Interactive click/mask selection, multiple actors in one job, production shot lookup and Maya are not part of this adapter yet.

See [third-party provenance](THIRD_PARTY.md). An overall redistribution license for the project's original code has not been selected. Model files are obtained separately under their applicable terms.
