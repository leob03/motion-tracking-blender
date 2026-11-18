# SAM 3D Body on camera (0.7.0)

The default body method since 0.7.0. SAM 3D Body gives the best pose and fingers, including on close-ups and
upper-body shots where GVHMR falls apart, but on its own it guesses the person's distance from the camera on
every frame, so in a scene the body jitters back and forth. With a known camera the worker fixes that:

1. **One body shape for the shot.** The actor's MHR shape and skeleton scale are the per-frame medians; the shot is
   re-run through MHR so mesh, joints and rig agree. Bone lengths are constant in the imported rig.
2. **Back on the plate.** Locking the proportions changes limb lengths, so each frame's joint angles are re-fitted
   (150 Adam steps on the GPU, about 0.12 s per frame) to put the locked body's 70 keypoints (body, hands, feet)
   where SAM's own body was on the plate, with a prior keeping the pose close to SAM's.
3. **Depth placement.** One number per frame moves the body along its viewing rays (its position on the plate does
   not change), chosen so that in the scene the root moves smoothly in depth (robustly: real fast moves are kept), a
   planted, visible foot does not slide, and the apparent size stays what SAM saw. A global factor maps SAM metres to
   scene units; it is fitted only when the camera travels and the feet are planted (otherwise 1).

No GVHMR stage runs. GVHMR and SAM + GVHMR stay available as fallbacks, for moving cameras that are not tracked.

Code: `production_bridge/sam_camera.py` (lock, refit, placement), `comparison_study/run_model.py --place`,
`production_bridge/worker.py` (`export_sam_camera`, `locked_off_camera`), `production_bridge/lens_estimate.py`.

## Cameras

- **Tracked camera (Camera Track):** as in the 0.6 hand-off, the add-on samples the scene camera on every job frame.
  The body is parented to the camera's parent (the Camera Track alignment).
- **Locked-off camera:** the worker builds a static camera. Lens, in order:
  1. The plate's lens metadata (image sequences): focal length and filmback from the EXR header, e.g. ARRI ALEXA 35
     `FocalLength` 32.77 mm and `SensorPixelPitch` 6.075 µm × 4608 px = 5394 px at plate width. Correct for EXRs scaled
     from the acquisition size; a crop is reported.
  2. The focal length typed in the Camera panel (with the filmback preset).
  3. An estimate from the image: GeoCalib (Apache-2.0, the calibration ViPE itself uses to start its lens) on 8 frames
     with one shared focal length. On the study clips it was within 0.90–1.11× of ViPE's solved focal lengths.
  GeoCalib also gives the camera's tilt and roll, so the scene comes in level, and the floor is put at Z = 0 under the
  planted feet when they are visible.
- Moving camera, not tracked: track it with Camera Track first, or use a GVHMR method.

The focal length now travels as a fraction of the plate width (`focal_width_ratio`), so it stays correct for image
sequences uploaded as downscaled proxies (the 0.6 *Known lens* mode sent plate pixels, which was wrong for proxies).

## Level from Actor

Result panel, for SAM-on-camera results on a Camera Track camera. The planted feet over the shot describe the
ground; Camera Track points near the feet (native cloud, dense environment) refine it. The fitted ground becomes
Z = 0 by moving the Camera Track alignment, like Camera Track's own Set Floor (its alignment reset still works), so
camera, points, environment, splats and the body move together and the plate match is unchanged. *Origin under the
actor* (default on) puts the world origin where the actor first stands. Choice of ground: *feet + points* (point
plane under the feet within 10° of the feet), *feet* (feet span an area), *points*, or *posture* (standing posture
gives up; height from the feet). The Result panel reports the method, the tilt corrected and how close the feet are.

## Results

End-to-end worker runs (processing size; plate error on the body joints inside the frame, against SAM's own projection):

| Clip | Camera | Plate error mean / p95 (px) | Depth jitter (cm) | Planted feet (frames L/R) | Worker time |
|---|---|---|---|---|---|
| barnab (walk, camera travels 2.4 m; 960 px) | Camera Track hand-off | 1.3 / 3.4 | 6.0 → 4.2 | 49 / 29 | 174 s |
| karoshi090 (upper body; 1280 px) | ViPE camera | 1.3 / 4.8 | 1.3 → 1.0 | 0 / 0 | 99 s |
| hand_signs (close-up, plain backdrop; 1120 px) | locked-off, GeoCalib lens 2202 px | 2.2 / 5.3 | 1.3 → 1.0 | 0 / 0 | 375 s (323 frames) |

The pose re-fit is what keeps a constant-bone rig on the plate: on hand_signs, locking the proportions alone left the
fingers 12.5 px off (median, 34 px p95); after the re-fit 1.5 px (5.9 px p95). Depth moves are anchored at the centre
of the keypoints inside the frame and bounded by a plate budget (0.4 % of the width), so close-ups stay on the plate and
distant bodies get the smoothing. barnab end to end: Level from Actor used the feet plus 9 089 Camera Track points,
corrected an 11.5° tilt, and left the planted ankles 9 cm above the ground (2.8 cm RMS from the expected ankle height).
Offline on six study clips: jitter halves on omitb (6.5 → 3.3 cm) and planted-foot sliding drops ~70 %; little gain on
fast action (unstoppable).

## Validation

```bash
python production_bridge/tests/sam_camera_placement.py          # synthetic placement: jitter, plate budget, feet, scale
blender -b --factory-startup --python-exit-code 1 --python production_bridge/tests/blender_panel_smoke.py
python production_bridge/tests/scene_camera_known_answer.py production_bridge result.mocap.zip
blender -b --factory-startup --python-exit-code 1 --python production_bridge/tests/blender_vipe_handoff.py -- \
  vipe_result.zip plate.mp4 OUT http://127.0.0.1:8188 [existing.mocap.zip]
```

The hand-off test accepts SAM-on-camera bundles (placement report instead of GVHMR registration). On barnab: Blender vs
worker projection 0.0002 px, mesh vs bundle 3 µm, projection unchanged after an alignment edit, Nuke export written.
Also run on the worker: GVHMR fallback with a scene camera (unchanged), SAM native face, and RealDenseFace/FLAME on SAM
on camera (editable v2 bundle, 131 bones, imports with expression keys).

## Limits

- Depth on close-ups without visible feet relies on body size and smoothness only; the scene scale stays 1.
- Locked proportions trade a few pixels of plate error at the extremities for a clean rig (constant bones);
  the re-fit keeps fingers and limbs on the plate (hand_signs fingers: 12.5 → 1.5 px median).
- One actor per job (as before). Lens distortion is not modelled.
- The pose re-fit and placement add about 0.15 s per frame on a Quadro RTX 5000 (6.3 GB peak with SAM); SAM inference dominates.
- Run with `MOCAP_DEBUG_PLACEMENT=1` to keep `sam/placement_input.pkl` (SAM pose, clip, camera) for offline tuning.
