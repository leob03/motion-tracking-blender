# Source and asset provenance

The original integration code has no newly selected blanket redistribution license. External models/code retain their own terms. This packaging task does not grant permission to redistribute weights or clear production use.

| Component | Source / role |
| --- | --- |
| GVHMR | https://github.com/zju3dv/GVHMR — underlying body/world method |
| Reviewed GVHMR Comfy port | https://github.com/PozzettiAndrea/ComfyUI-MotionCapture — actual inference implementation used here |
| User's earlier fork | https://github.com/leob03/ComfyUI-MotionCapture — historical reference, not substituted for the measured source pin |
| SAM / Comfy port | https://github.com/Comfy-Org/ComfyUI and https://github.com/facebookresearch/sam-3d-body |
| MHR | https://github.com/facebookresearch/MHR — rig and topology correspondence assets |
| RealDenseFace | https://github.com/LinzhouLi/RealDenseFace — optional face fitting and CUDA solver |
| SMIRK | https://github.com/georgeretsi/smirk — landmark embedding only; no SMIRK inference in this adapter |
| SMPL-X | https://smpl-x.is.tue.mpg.de/ — neutral body model, obtained separately |
| FLAME | https://flame.is.tue.mpg.de/ — FLAME2023 head, obtained separately |
| YOLO | https://github.com/ultralytics/ultralytics — initial actor detection/tracking boxes |

[Source pins](provenance/source-pins.json) record the actual commits and selected runtime file hashes. The RealDenseFace patch preserves the reviewed BF16 capability check and lazy visibility-rasterizer initialization. The pinned checkout predates some current upstream files; use its own applicable notices and the approved model terms rather than assuming the latest repository state matches it.

`mhr_bone_names.json` preserves the native MHR bone naming used by the skinning adapter. Third-party libraries/checkouts are referenced rather than vendored. Weights, original shot media and fitted production results are excluded. Review model/code terms with the studio before rollout, as with the earlier camera-tracking handoff.

The [FLAME Blender add-on](https://github.com/TimoBolkart/FLAME-Blender-Add-on) was reviewed as a reference for native shape-key/joint controls. It is not a dependency and none of its add-on code or downloadable model assets are bundled in this source repository. FLAME editing result bundles contain model-derived deformation data and should be handled under the applicable model terms.
