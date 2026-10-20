# Facial corrections in Blender 0.4.0

Choose **SAM + GVHMR world/root** and **RealDenseFace + FLAME head**, submit, then import with **Mesh** and **Editable armature** enabled. Both worker and artist add-on must be updated to 0.4.0. No additional model download or artist Python dependency is required.

Open **Mocap → 6. Face corrections** with the desired character or its review camera selected:

1. Scrub to a frame and adjust the signed **Expression** sliders. Pages 0–9 expose the 100 FLAME PCA components. These are the model's mathematical components, not named smile/blink/FACS controls. They also appear as normal shape keys on the body mesh.
2. Press **Edit Jaw / Eye Bones**. Rotate `FLAME_jaw`, `FLAME_eye.L` or `FLAME_eye.R` in Pose Mode. Their local axes match the fitted FLAME model. Head/neck placement remains on the body rig.
3. Press **Key Face Corrections** before changing frame. Standard Blender keyframe insertion and Graph Editor workflows also work. Capture is keyed every frame, so an edit at one frame does not automatically spread over neighboring frames.
4. **Restore Captured Face at This Frame** restores and keys the original expressions/jaw/eyes at that frame, preserving corrections elsewhere. Blender Undo also works for these operators.

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
