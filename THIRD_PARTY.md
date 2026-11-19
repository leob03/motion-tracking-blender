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
| SMPL-X | https://smpl-x.is.tue.mpg.de/ — neutral body model for the GVHMR fallback only, obtained separately |
| FLAME | https://flame.is.tue.mpg.de/ — FLAME 2023 Open head (CC-BY-4.0), obtained separately. Credit: T. Li, T. Bolkart, M. J. Black, H. Li, J. Romero, *Learning a model of facial shape and expression from 4D scans*, SIGGRAPH Asia 2017 |
| D-FINE | https://github.com/Peterande/D-FINE, weights https://huggingface.co/ustc-community/dfine-large-coco (Apache-2.0) — actor detection boxes, run through Hugging Face Transformers |
| FLAME landmark file | `production_bridge/assets/flame_open_mhr_landmarks.npz`, made by `production_bridge/tools/make_flame_landmarks.py` from FLAME 2023 Open, MHR and SAM 3D Body's face-keypoint map |

Licence status per pipeline path: [docs/LICENSING.md](docs/LICENSING.md). Until 0.7.1 the worker also used Ultralytics YOLOv8 (AGPL-3.0) for detection, and SMPL-X plus SMIRK's landmark embedding to place the FLAME 2023 head; 0.8.0 removed all three from the SAM 3D Body and FLAME paths.

[Source pins](provenance/source-pins.json) record the actual commits and selected runtime file hashes. The RealDenseFace patch preserves the reviewed BF16 capability check and lazy visibility-rasterizer initialization. The pinned checkout predates some current upstream files; use its own applicable notices and the approved model terms rather than assuming the latest repository state matches it.

`mhr_bone_names.json` preserves the native MHR bone naming used by the skinning adapter. Third-party libraries/checkouts are referenced rather than vendored. Weights, original shot media and fitted production results are excluded. Review model/code terms with the studio before rollout, as with the earlier camera-tracking handoff.

The [FLAME Blender add-on](https://github.com/TimoBolkart/FLAME-Blender-Add-on) was reviewed as a reference for native shape-key/joint controls. It is not a dependency and none of its add-on code or downloadable model assets are bundled in this source repository. FLAME editing result bundles contain model-derived deformation data and should be handled under the applicable model terms.
