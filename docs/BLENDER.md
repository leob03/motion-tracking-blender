# Blender installation and artist workflow

## Install

Build `mocap_comfy-0.6.0.zip` with `python scripts/build_packages.py` (output in `dist/`). Install it through **Edit → Preferences → Add-ons → Install from Disk**, enable **Mocap ComfyUI**, then restart Blender. Blender 4.4.0 on Linux is tested; Windows remains unvalidated.

The worker URL, optional bearer token and download folder are **add-on preferences** (shared by all scenes; also shown in the **Worker** sub-panel). For a local worker use `http://127.0.0.1:8188`. A remote worker can use the team's approved HTTP(S) endpoint or an SSH tunnel:

```bash
ssh -N -L 8188:127.0.0.1:8188 USER@GPU_WORKER
```

The plate stays on the artist machine and is uploaded over HTTP once per content (a SHA-256 name lets later jobs reuse it). No shared filesystem is required. Tokens are never saved in the scene. The production gateway has not been validated.

## Panel layout (3D View sidebar → Mocap)

The header shows the worker state (refresh button = **Check Connection**), the **Submit to ComfyUI** button and the selected job's stage checklist, progress, elapsed time and estimate. Sub-panels:

1. **Worker**: URL, token, download folder, *Keep models warm on worker*.
2. **Plate and actor**: a movie, or any frame of an EXR/DPX/TIFF/PNG/JPEG sequence. Size, frame count and fps are read from headers. **First/Last** are 1-based for movies and file frame numbers for sequences (↔ selects the whole plate); the line below shows the frame count and timeline range. Sequences ask for their colour space: the add-on tone-maps and downsizes the selection to a JPEG proxy before upload (ViPE's proxy format) and keeps the file frame numbers on the timeline. **Detect Actors** uploads the plate (reused by the job) and shows the first frame with numbered people; click a number to pick the actor. The job then selects that person by overlap. Without detection, actors are numbered 0, 1, 2 left to right; −1 picks the largest.
3. **Camera**:
   - *Estimate from footage*: as before; focal assumed from the image diagonal.
   - *Known lens*: focal length (mm) and filmback; camera motion still inferred.
   - *Solved scene camera*: pick a tracked camera in this scene (ViPE, 3DE, …). See below.
4. **Processing**: preset (*Quick body (GVHMR)*, *Body + hands*, *Body + hands + face*) with the underlying method, hand refinement, face and smoothing settings; processing size and FPS override.
5. **Import**: mesh/armature/auto-import options and **Import Saved Bundle…**.
6. **Result** (when a result is active): summary (frames, camera, plate error, world-fit residual, worker timings), review cameras, **Edit Body / Finger Bones**, **Export to Nuke…**.
7. **Face corrections**: unchanged; see [facial editing](FLAME_EDITING.md).
8. **Jobs**: every submitted job of this scene (state icons), **Resume**, **Import**, **Show Log**, remove from list. Several jobs (actors, ranges) can be queued at once; the worker runs them in order.
9. **Logs**: live log pane beside the viewport.

## Solved scene camera (ViPE hand-off)

1. Track the plate with the **ViPE Remote Tracking** add-on (or import another solved camera). Set its floor/origin/scale with ViPE's scene alignment tools if you need a ground-aligned scene; this can also be done afterwards.
2. In **Camera**, choose *Solved scene camera* and pick the camera (the camera button uses the scene's active camera). For a ViPE camera the timing is taken from its job; for any other camera, video frame **First** lands on **Blender start frame**. Warnings appear if the camera is not keyed over the job range, if the ViPE plate size differs, or if ViPE estimated lens distortion (the body is fitted on the distorted plate).
3. Submit. The camera is sampled on every job frame and uploaded with the plate.

The worker keeps GVHMR's **world motion** (it feeds the solved camera rotations to GVHMR instead of its own odometry) and places it into the scene with one rigid transform for the whole shot. A small sideways **plate lock**, never a change in depth, then keeps the body on the plate, and a **rotation lock** turns the body's root toward the orientation seen in the plate. *Plate lock* 0 keeps GVHMR's world placement (and its foot contacts); 1 follows the plate. *Rotation lock* defaults to 0.5; use 1 on plate-critical shots. *Lock smoothing* (frames, default 3) trades plate fit against jitter. When GVHMR's world drifts relative to the solved camera, following the plate costs some foot sliding: choose per shot. The imported character is parented to the camera's parent, so it follows later ViPE alignment edits, and the ViPE camera stays the active camera. GVHMR's own camera is added hidden for comparison. The Result panel reports the plate error (pixels), the world-fit residual and the tilt between GVHMR's gravity and the scene's up axis (large values mean the scene is not ground-aligned yet).

## Review and edit

New imports create a `Mocap | shot | actor` collection containing **01 Character** and **02 Cameras**. Existing objects and earlier imports remain untouched. Timeline, frame rate, plate resolution and units are set from the result; curves are written in bulk (import up to about 2× faster).

- **Video Fit Camera** (estimated cameras) preserves this hybrid actor's SAM projection after replacing its world/root transform. It cannot be assumed to fit other actors or the background.
- **GVHMR World Camera** is GVHMR's native inferred camera; the hybrid mesh can differ from its original video fit.

Press numpad 0 to review the plate background (movie clip, or the proxy frames for sequences). Click **Edit Body / Finger Bones**, adjust bones in Pose Mode, then insert keys or use Auto Keying. Capture is keyed on every frame: unkeyed edits disappear when scrubbing. The rig is FK, without animator IK controls. Hybrid rigs have 127 MHR bones, or 131 with editable FLAME; GVHMR rigs have 55 SMPL-X bones. GVHMR does not predict finger motion.

## Nuke

**Export to Nuke…** writes, for the active result: `<shot>_mocap.nk` (plate Read, Camera2, ReadGeo2 of the body Alembic, Scene and ScanlineRender check), `<shot>_camera.chan` and `<shot>_body.abc` (skinned body and camera, world space, metres), plus a README. Conventions match the ViPE exporter (Camera2 `rot_order ZXY`, filmback from the Blender camera). The `.nk` wiring was generated without Nuke and is unverified.

## Jobs, cancellation and resume

**Cancel Job** uses the fork's job-scoped cancellation route. **Resume** reconnects to a submitted job (job ID kept in the scene) without submitting it again. A server restart may remove queue/history, so keep downloaded results. Full received logs are saved with each download.
