# Validation evidence and limits

## Reviewed prototype

Blender **4.4.0 on Linux**, with a Linux/CUDA worker on a 16 GB Quadro RTX 5000. Earlier adapter checks also ran on Blender 4.2.3 LTS; the latest rig/UI release was checked on 4.4.

Version 0.3.0 completed fresh hybrid/FLAME and GVHMR jobs through ComfyUI. The installed Blender Submit operator streamed logs/progress, displayed a recent-run ETA and automatically imported a 12-frame hybrid/FLAME result. Scoped cancellation, worker errors and checksum rejection were checked in the original development environment.

Version 0.3.1 reused those captured results for the UI/import update:

- Hybrid/FLAME: 127-bone native MHR rig; maximum checked mesh-coordinate error about 0.00000060 m.
- GVHMR: 55-bone native SMPL-X rig; maximum checked mesh-coordinate error about 0.00000072 m.
- Camera reprojection errors below 0.001 pixel at first/middle/last checked frames.
- Elbow/finger keyframe edits moved weighted regions and preserved unaffected regions.
- Cameras visible, two child collections, no separate reference-marker empties.
- Actual graphical split/reuse/close behavior preserves the viewport and other editors; new logs reuse the pane. ZIP enable/disable/re-enable passed.

Sanitized measured reports are in [validation/](../validation/). These are integration/transfer checks, not ground-truth motion accuracy measurements.

## Packaging validation

The handoff preserves runtime source bytes and records external pins plus the scoped RealDenseFace patch. Seventeen model/support assets were hashed one at a time. The three converted GVHMR checkpoints and converted SAM checkpoint match their pinned upstream Hugging Face SHA256 records. Public source commits were checked for availability. See the machine-readable provenance files.

Synthetic geometry/import, malformed bundle/rig rejection, package composition and source-byte checks are rerun from this repository during packaging. This snapshot does not rerun GPU inference or certify a fresh worker/container rebuild. Latest packaging outcomes are recorded in `validation/packaging.json`.

## Not yet validated

- Windows Blender sessions and production gateway/authentication.
- Clean environment/container rebuild, enforced offline execution, target GPU extension builds.
- Integration with the team's generic DCC bridge or Maya.
- Arbitrary long shots, memory budgets, cache streaming or results above the 2 GiB client cap.
- Shared-world consistency across independently processed actors, camera calibration/ground truth, severe occlusion robustness.
- Editable FLAME expression/jaw/eye controls; current facial animation is cached.

The measured 137-frame development job took about 269 seconds with hybrid body/hands and face disabled, including startup/preparation/export. This is one workstation result, not a production SLA. The process is sequential and currently reloads workers/models for each job.
