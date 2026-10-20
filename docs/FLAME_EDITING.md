# Facial corrections in Blender 0.5.0

Choose **SAM + GVHMR world/root** and **RealDenseFace + FLAME head**, submit, then import with **Mesh** and **Editable armature** enabled. Use artist add-on 0.5.0 and the editable-FLAME worker from 0.4.0 or later. Existing editable captures work without re-export or inference. No additional model download or artist Python dependency is required.

Open **Mocap → 4. Face corrections** with the desired character or its review camera selected:

1. Scrub to the frame you want to correct. Press **Face Close-up** to frame the face and show the controls. **Previous View** restores your earlier view and guide visibility. You can also enable **Show Visual Face Controls** directly while looking through the video-fit camera.
2. Drag a **green** handle on a mouth corner, upper/lower lip, brow, or eyelid. The handle follows the evaluated mesh; its label identifies the region. Left/right refer to the actor. Motion is constrained to the current view plane, so a camera-view drag changes the visible fit without asking you to guess a depth value.
3. Drag the **orange jaw** handle down to open or up to close. Drag either **purple eye-aim** handle to adjust that eye's gaze. These operate the existing FLAME bones. Eyelid handles change expression, not gaze.
4. Hold **Shift** for smaller movements. **Esc** or right-click cancels the current drag. **Ctrl Z** undoes a completed drag. **Key on release** is enabled by default, so a completed edit survives scrubbing. With it disabled, press **Key Face Corrections** before changing frames. Capture is keyed every frame: changes affect the edited frame, without an automatic temporal falloff.
5. **Restore Captured Face at This Frame** restores and keys the original expressions/jaw/eyes at that frame, preserving corrections elsewhere. **Control labels** hides the callouts when you prefer an uncluttered view.

The previous controls remain available under **Advanced expression sliders** and **Edit Jaw / Eye Bones**. The latter switches off the visual handles and shows the native bones in Pose Mode. Pages 0–9 expose the 100 signed FLAME PCA components; they are not named FACS/ARKit controls.

## What the visual handles solve

Expression handles fit a combination of the existing FLAME expression keys to your drag. Other facial landmarks and sampled surface points act as soft anchors, and a penalty discourages large coefficient changes. This is an interactive NumPy solve in Blender; there is no ComfyUI round trip or model inference. Body pose, identity, jaw/eye pose and the protected collar are excluded from the expression solve.

The solver uses the current skinned pose and object transforms, so existing head corrections are respected. Nearby facial regions can still move: a FLAME component affects multiple vertices, and anchors are soft constraints. This is suitable for small corrections, not arbitrary sculpting or perfectly independent FACS actions. A drag is limited to 15 mm at the character's scale and three coefficient units per component; jaw/eye changes are limited to 0.35 radians per drag. There is no collision solve or guarantee of complete eyelid/lip closure.

Visual handles require the standard FLAME2023 5023-vertex head and the original linear armature modifier. Disable additional deformation/topology modifiers when using the solver. Shape keys and bone animation remain native Blender data; only the editing widgets need the add-on. Saved corrections play without it. **Face Close-up** changes viewport framing and guide visibility, not the shot camera or character transform.

Rare coefficients outside Blender's slider range are represented by a displayed scale (for example ×2); the captured motion is not clipped. Facial controls work on the existing joined body mesh. An existing MHR jaw/eye bone is not the new FLAME control: use the dedicated `FLAME face` bone collection.

## How it works

The worker loads the same FLAME2023 identity, expression basis, joint pivots and skin weights as RealDenseFace. Four bones are added under the MHR head: an attachment helper, jaw, and two eyes. The recorded FLAME rotations are baked onto them. Expression shape keys are animated from the fitted coefficients.

The fitted surface is preserved by keeping a residual cache before skinning. Predicted expression deformation is subtracted from that cache and added back through the animated keys. Small changes therefore modify the captured result without refitting the video. Facial bones share the body head’s rest frame. Influence fades smoothly by distance along the mesh from the collar: the upper neck follows the jaw, while the seam and body stay fixed. Large corrections can still cause local distortion or self-intersection; there is no collision solve.

The reviewed RDF solver disables pose-corrective blendshapes (`USE_POSE_BS=false`). This export matches it. A model/decoder replay check rejects incompatible assets or fitter conventions. Model identity is fixed for this workflow; changing identity is not an artist control.

The mesh, shape keys, bones and animation are native Blender data. Saved scenes evaluate without the add-on, external model files, drivers or Python handlers. The add-on provides convenient selection, keying and restoration UI. The official [FLAME Blender add-on](https://github.com/TimoBolkart/FLAME-Blender-Add-on) informed the control conventions; it is not required or bundled.

SAM-native face capture remains cached. These controls currently apply to the joined RealDenseFace/FLAME head on the hybrid body, not to a new FLAME-to-SMPL-X facial retargeter.

## Upgrade a retained capture

Old files do not gain controls just by updating Blender. On the worker, use the existing inference environment:

```bash
/path/to/worker/python scripts/upgrade_flame_bundle.py \
  --study /path/to/comparison_study \
  --input old_result.mocap.zip --output editable_result.mocap.zip
```

This requires the retained `face_parameters.npz`, native body rig and the worker's existing FLAME model. It runs no neural inference and preserves the original file. Import the new bundle into a new collection. Version 2 intentionally prevents old add-ons from importing the residual mesh without its expression keys.

## Focused validation

```bash
/path/to/worker/python production_bridge/tests/make_flame_reference.py \
  --study /path/to/comparison_study --bundle editable_result.mocap.zip \
  --output reference.npz
/path/to/blender -b -t 2 --factory-startup --python-exit-code 1 \
  --python production_bridge/tests/blender_flame.py -- \
  editable_result.mocap.zip reference.npz validation/face_review
```

Run one Blender or inference process at a time on the development workstation. The validation covers original geometry, expression/jaw/eyes against an independent skinning implementation, body/neck isolation, key/restore, malformed input, and native save/reopen.
