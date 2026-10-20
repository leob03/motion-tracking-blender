# Blender installation and artist workflow

## Install

Build `mocap_comfy-0.4.0.zip` with `python scripts/build_packages.py` (output in `dist/`). Install it through **Edit → Preferences → Add-ons → Install from Disk**, enable **Mocap ComfyUI**, then restart Blender. Blender 4.4.0 on Linux is tested; Windows remains unvalidated.

Open the 3D View sidebar with **N**, then **Mocap**. Enter the worker's ComfyUI URL. For a local worker use `http://127.0.0.1:8188`. A remote worker can use the team's approved HTTP(S) endpoint or an SSH tunnel:

```bash
ssh -N -L 8188:127.0.0.1:8188 USER@GPU_WORKER
```

Use `http://127.0.0.1:8188` in Blender on the tunnel's client machine. The video stays selectable from that artist machine and is uploaded through HTTP. No shared filesystem is required. Bearer-token support exists in advanced settings; tokens are not saved in the scene. The production gateway has not been validated.

## Submit

1. Choose a local MP4/MOV/MKV/AVI/WebM/M4V file. Pick a zero-based source start and frame count. **0 means all remaining frames**; start with 24 frames for a new worker. Temporal inference needs at least eight decodable frames.
2. Choose an actor: indices 0, 1, 2 run left to right in the first selected frame; -1 chooses the largest. The chosen boxes are tracked through the range. This is not the study portal's click/mask selector.
3. Choose **SAM + GVHMR world/root**, or **GVHMR body + world (faster)**. Hybrid offers hand refinement, SAM smoothing, and three face choices: off, SAM native, RealDenseFace + FLAME head.
4. Choose moving/static camera, timeline start, mesh and armature import. Default Blender start is 1001; **Video fit** is the default review camera.
5. Press **Submit to ComfyUI**. Networking runs in a background thread; scene import runs on Blender's main thread. Models run sequentially on the worker.

The six numbered sections collapse independently. **Progress and logs** shows available stage/frame counts, elapsed time, last worker activity and approximate remaining time. **Open Live Log** splits the viewport and places a Text Editor on its right. **Close Log** in its header or the Mocap sidebar closes the pane. Repeated opens reuse it. Restore a maximized viewport with Ctrl Space first. Full received logs are saved alongside downloads; the visible text retains the last 100,000 characters.

**Cancel This Job** requires the fork's job-specific cancellation route. **Resume Job** reconnects using a saved prompt ID; it does not submit a duplicate. A server restart may remove queue/history, so retain downloaded results. **Import Downloaded Bundle** imports an existing `.mocap.zip` without inference.

## Review and edit

New imports create a `Mocap | shot | actor` collection containing **01 Character** and **02 Cameras**. Both cameras are visible; separate reference-marker empties are no longer created. Existing objects and earlier imports remain untouched. Timeline, frame rate, resolution, units and active camera are set from the result.

- **Video Fit Camera** preserves this hybrid actor's SAM projection after replacing its world/root transform. It cannot be assumed to fit other actors or the background.
- **GVHMR World Camera** is GVHMR's native inferred camera; the hybrid mesh can differ from its original video fit. GVHMR-only jobs use this camera for either review choice.

Press numpad 0 to review the movie background. Click **Edit Body / Finger Bones**, adjust bones in Pose Mode, then insert transform keys or use Auto Keying. Capture is keyed on every frame: unkeyed edits disappear when scrubbing. The rig is FK, without animator IK controls. The unedited fitted surface is preserved through a cache before skinning; learned correctives are not recomputed for large manual pose changes.

Hybrid rigs have 127 MHR bones, or 131 with editable FLAME; GVHMR rigs have 55 SMPL-X bones. GVHMR does not predict finger motion. For RealDenseFace + FLAME, open **6. Face corrections** to edit expression shape keys and jaw/eye bones, key corrections, or restore a captured frame. See [facial editing](FLAME_EDITING.md). SAM-native facial animation remains cached.

Frame-count defaults apply to new settings. Saved scenes retain their stored values. Reimport old downloaded bundles to obtain the new names/organization; bundles predating native rig export remain cache-only.
