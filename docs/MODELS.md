# Models and DVC handoff

All listed files belong on the GPU worker. **The artist machine needs none of these weights.** The [machine-readable manifest](../provenance/model-manifest.json) contains exact filenames, root-relative destinations, SHA256 values, byte sizes and source URLs. No weights are committed.

## Feature sets

| Enabled mode | Unique files | GiB |
| --- | ---: | ---: |
| GVHMR body/world | 11 | 5.268 |
| SAM 3D Body (on camera or hybrid), including optional SAM native face | 13 | 8.013 |
| SAM 3D Body + RealDenseFace/FLAME | 18 | 8.909 |

The complete set is approximately **9.57 GB / 8.91 GiB** (0.7.0 adds the 116 MB GeoCalib weights; 0.8.0 replaces the YOLO detector with the 126 MB D-FINE-L weights and FLAME 2023 with FLAME 2023 Open). Sizes exclude source repositories, Python/CUDA environments, compiler caches and job scratch. Some small entries already ship in pinned source repositories; warehouse them with their source or as DVC assets, but preserve their expected locations.

## Download links

| Asset | MiB | Required for | Download kind |
| --- | ---: | --- | --- |
| [gvhmr](https://huggingface.co/apozz/motion-capture-safetensors/resolve/4d1ff1105aad31ef01d2aa2c1151a0317d267925/gvhmr.safetensors) | 155.866 | All modes | download |
| [vitpose](https://huggingface.co/apozz/motion-capture-safetensors/resolve/4d1ff1105aad31ef01d2aa2c1151a0317d267925/vitpose.safetensors) | 2430.803 | All modes | download |
| [hmr2](https://huggingface.co/apozz/motion-capture-safetensors/resolve/4d1ff1105aad31ef01d2aa2c1151a0317d267925/hmr2.safetensors) | 2583.821 | All modes | download |
| [person_detector](https://huggingface.co/ustc-community/dfine-large-coco/tree/aa8f8211545e4cff5adf52ea512f9ac74a9824ec) (D-FINE-L: `model.safetensors`, `config.json`, `preprocessor_config.json`) | 119.806 | All modes | download (Apache-2.0) |
| [smplx_neutral](https://smpl-x.is.tue.mpg.de/) | 103.714 | All modes | registration download |
| [sam_body](https://huggingface.co/Comfy-Org/sam-3d-body/resolve/7c974aea58e062252a6cb5b6f9b7f639d545b729/detection/sam_3d_body_dinov3_bf16.safetensors) | 2699.602 | Hybrid | download |
| [smplx2smpl_sparse.npz](https://raw.githubusercontent.com/PozzettiAndrea/ComfyUI-MotionCapture/35c8dbc4de14448d4626a0f99e32c869371fea12/nodes/body_model/smplx2smpl_sparse.npz) | 0.046 | All modes | source asset |
| [smpl_coco17_J_regressor.npy](https://raw.githubusercontent.com/PozzettiAndrea/ComfyUI-MotionCapture/35c8dbc4de14448d4626a0f99e32c869371fea12/nodes/body_model/smpl_coco17_J_regressor.npy) | 0.447 | All modes | source asset |
| [smplx_verts437.npy](https://raw.githubusercontent.com/PozzettiAndrea/ComfyUI-MotionCapture/35c8dbc4de14448d4626a0f99e32c869371fea12/nodes/body_model/smplx_verts437.npy) | 0.003 | All modes | source asset |
| [realdenseface_vitb](https://drive.google.com/file/d/1eLlN_BCFMn0jSo0lNwBFqtI6igTh7Nc3/view) | 862.598 | RealDenseFace/FLAME | release archive |
| [facebox](https://drive.google.com/file/d/1eLlN_BCFMn0jSo0lNwBFqtI6igTh7Nc3/view) | 3.983 | RealDenseFace/FLAME | release archive |
| [flame_assets](https://drive.google.com/file/d/1eLlN_BCFMn0jSo0lNwBFqtI6igTh7Nc3/view) | 0.362 | RealDenseFace/FLAME | release archive |
| [flame2023_open](https://flame.is.tue.mpg.de/) (FLAME 2023 Open) | 50.567 | RealDenseFace/FLAME | registration download (CC-BY-4.0) |
| [head_hand_mask.npz](https://raw.githubusercontent.com/facebookresearch/MHR/d96fafa33bbf018647c70c3525e91f53e79d2a14/tools/mhr_smpl_conversion/assets/head_hand_mask.npz) | 0.212 | RealDenseFace/FLAME | source asset |
| [smpl_mean_params](https://raw.githubusercontent.com/PozzettiAndrea/ComfyUI-MotionCapture/35c8dbc4de14448d4626a0f99e32c869371fea12/nodes/hmr2/smpl_mean_params.npz) | 0.001 | All modes | source asset |
| [geocalib_pinhole](https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar) | 110.697 | SAM 3D Body on camera, locked-off without lens metadata | download (Apache-2.0) |

The three GVHMR files are the **converted safetensors** consumed by the reviewed Comfy port; upstream GVHMR `.ckpt` downloads are not interchangeable inputs to this wrapper. The manifest pins the Hugging Face conversion revision and verifies its hashes against the local files. The SAM conversion is similarly revision/hash pinned.

The RealDenseFace link is its author-published release archive: retain only `vitb.pth`, `face_box.pth` and `flame_assets.npz` from that archive. ViT-S is not used. FLAME 2023 Open (the CC-BY-4.0 release, not the non-commercial FLAME 2023) and SMPL-X neutral (GVHMR fallback only) are separate official registration/download flows. Preserve the versions/checksums in the manifest; do not commit those files to this repository.

## Existing loader paths

The manifest root keys map to these **current** locations:

| Root key | Existing loader meaning |
| --- | --- |
| `nightly_models` | `/home/innovation/dev-lbringer/ComfyUI_nightly/models` |
| `gvhmr_node` | `/home/innovation/dev-lbringer/ComfyUI_nightly/custom_nodes/ComfyUI-MotionCapture` |
| `study` | `MOCAP_STUDY_ROOT` |

Append each entry’s `path` to that root:

```text
nightly_models/motion_capture/gvhmr.safetensors
nightly_models/motion_capture/vitpose.safetensors
nightly_models/motion_capture/hmr2.safetensors
nightly_models/motion_capture/body_models/smplx/SMPLX_NEUTRAL.npz
study/models/sam_3d_body_dinov3_bf16.safetensors
study/models/person_detector/dfine-large-coco/model.safetensors
study/models/person_detector/dfine-large-coco/config.json
study/models/person_detector/dfine-large-coco/preprocessor_config.json
gvhmr_node/nodes/body_model/smplx2smpl_sparse.npz
gvhmr_node/nodes/body_model/smpl_coco17_J_regressor.npy
gvhmr_node/nodes/body_model/smplx_verts437.npy
study/vendor/RealDenseFace/weights/realdenseface/vitb.pth
study/vendor/RealDenseFace/weights/facebox/face_box.pth
study/vendor/RealDenseFace/weights/flame/flame_assets.npz
study/models/face/flame_downloads/FLAME2023_Open/flame2023_Open.pkl
study/vendor/MHR/tools/mhr_smpl_conversion/assets/head_hand_mask.npz
gvhmr_node/nodes/hmr2/smpl_mean_params.npz
study/models/geocalib/pinhole.tar
```

For example, the face model is read from `<study>/models/face/flame_downloads/FLAME2023_Open/flame2023_Open.pkl`; the adapter supplies that path to the fitter. The conventional upstream `weights/flame/flame2023.pkl` location is not a second required copy for this adapter. The RealDenseFace network and support assets retain their vendor-relative `weights/` locations because the face subprocess changes working directory to that checkout.

The manifest’s root keys describe inventory locations; they are **not newly implemented config variables**. Only the study root is configurable today. Supporting arbitrary model mount paths requires the deferred worker-portability change. Production can reproduce the fixed layout with deliberate mounts/symlinks until then.

## Assets that are already inside the selected checkpoint

The SAM converted safetensors includes the DINOv3 backbone, MHR rig data, and its pure-PyTorch MediaPipe face components. No separate MHR FBX/LOD package, MediaPipe task or DINOv3 weight download is required for this integration. SAM native face uses the same checkpoint. Hands are handled by SAM hand refinement; no separate hand model is downloaded here.

RealDenseFace sets `dino_pretrained=false` and loads its complete ViT-B checkpoint. Its FaceBox weights remain required because the inference class instantiates that detector even when the adapter supplies actor-specific face boxes. The MHR head mask is a small support array, not an extra inference model. The FLAME landmark file that places the head on SAM's body ships in this repository (`production_bridge/assets/flame_open_mhr_landmarks.npz`, made by `production_bridge/tools/make_flame_landmarks.py`); SAM's face-keypoint map is read from the SAM checkpoint at run time.

The current GVHMR path uses SimpleVO. DPVO, DROID, Pixel3DMM and SMIRK are outside this prototype; since 0.8.0 no SMIRK or SMPL-X file is read by the SAM 3D Body or FLAME paths.

## DVC preparation

1. Retrieve the exact listed files and verify SHA256 before adding them to the studio warehouse. Public direct URLs are provided where the source supports them; registration/archive entries require their respective preparation steps.
2. Preserve filenames and resolve DVC/source symlinks inside the deployed container. A symlink into an unmounted developer cache is insufficient.
3. Pin the DVC descriptors alongside the reviewed source pins and environment recipe. Build compiled dependencies separately as described in [OFFLINE.md](OFFLINE.md).
4. Test all enabled modes with external networking blocked. This audit did not run a fresh offline container.

Checksums were computed sequentially with 8 MiB reads of explicitly identified files, without recursive model/storage scans. The complete manifest has 18 entries. The four large Hugging Face checkpoints also match upstream SHA256 metadata.

To verify staged files without traversing the model directories:

```bash
python scripts/verify_models.py \
  --study-root /path/to/study \
  --nightly-models /path/to/ComfyUI_nightly/models \
  --gvhmr-node /path/to/ComfyUI-MotionCapture \
  --mode face --sha256
```

Use `--mode gvhmr` or `--mode hybrid` for smaller feature sets. Without `--sha256`, the utility checks existence and byte counts only. Hashing reads one selected file at a time.
