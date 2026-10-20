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
- Semantic FACS/ARKit facial controls and large facial edits/collision handling. Native FLAME PCA/jaw/eye corrections are validated below.

The measured 137-frame development job took about 269 seconds with hybrid body/hands and face disabled, including startup/preparation/export. This is one workstation result, not a production SLA. The process is sequential and currently reloads workers/models for each job.

## FLAME editing 0.4.0

A retained 12-frame RealDenseFace capture was upgraded without rerunning inference. Its canonical FLAME model replay differed from the original CUDA solver by at most 3.22e-7 m. All 12 unedited Blender frames were checked (maximum 1.79e-6 m). Expression 001 and 8-degree jaw/left-eye/right-eye corrections at first/middle/last frames agreed with an independent CPU skinning reference within 4.00e-7 m. Protected neck and body changes were exactly zero in these checks.

Key/restore, malformed facial bundle rejection before scene changes, and save/reopen with the add-on unregistered passed; reloaded corrected geometry was unchanged. These are deformation-transfer checks, not facial tracking accuracy or aesthetic-quality measurements. The reviewed fitter has pose blendshapes disabled; editing matches that convention.

A fresh eight-frame hand-signs job also completed through the local ComfyUI API (about 139 seconds including body, hands, face and export). All eight imported frames passed; facial edits agreed with the independent reference within 3.82e-7 m. A manual MHR head rotation carried the FLAME face within 5.58e-7 m. The body and protected lower neck remained unchanged under facial-only edits. See [measured results](../validation/flame-editing.json).

The final export was rebuilt from the retained predictions to align facial/body rest frames and smooth the collar transition. The tested neck-transition triangles did not invert (minimum normal cosine above 0.992 on both captures). Legacy GVHMR-only geometry/reprojection and malformed bundle checks also passed.
