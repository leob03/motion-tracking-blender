# Models and DVC handoff

All listed files belong on the GPU worker. **The artist machine needs none of these weights.** The [machine-readable manifest](../provenance/model-manifest.json) contains exact filenames, root-relative destinations, SHA256 values, byte sizes and source URLs. No weights are committed.

## Feature sets

| Enabled mode | Unique files | GiB |
| --- | ---: | ---: |
| GVHMR body/world | 9 | 5.202 |
| Hybrid body/hands, including optional SAM native face | 10 | 7.839 |
| Hybrid + RealDenseFace/FLAME | 17 | 8.735 |

The complete set is approximately **9.38 GB / 8.74 GiB**. Sizes exclude source repositories, Python/CUDA environments, compiler caches and job scratch. Some small entries already ship in pinned source repositories; warehouse them with their source or as DVC assets, but preserve their expected locations.

## Download links

| Asset | MiB | Required for | Download kind |
| --- | ---: | --- | --- |
| [gvhmr](https://huggingface.co/apozz/motion-capture-safetensors/resolve/4d1ff1105aad31ef01d2aa2c1151a0317d267925/gvhmr.safetensors) | 155.866 | All modes | download |
| [vitpose](https://huggingface.co/apozz/motion-capture-safetensors/resolve/4d1ff1105aad31ef01d2aa2c1151a0317d267925/vitpose.safetensors) | 2430.803 | All modes | download |
| [hmr2](https://huggingface.co/apozz/motion-capture-safetensors/resolve/4d1ff1105aad31ef01d2aa2c1151a0317d267925/hmr2.safetensors) | 2583.821 | All modes | download |
| [actor_detector](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt) | 52.357 | All modes | download |
| [smplx_neutral](https://smpl-x.is.tue.mpg.de/) | 103.714 | All modes | registration download |
| [sam_body](https://huggingface.co/Comfy-Org/sam-3d-body/resolve/7c974aea58e062252a6cb5b6f9b7f639d545b729/detection/sam_3d_body_dinov3_bf16.safetensors) | 2699.602 | Hybrid | download |
| [smplx2smpl_sparse.npz](https://raw.githubusercontent.com/PozzettiAndrea/ComfyUI-MotionCapture/35c8dbc4de14448d4626a0f99e32c869371fea12/nodes/body_model/smplx2smpl_sparse.npz) | 0.046 | All modes | source asset |
| [smpl_coco17_J_regressor.npy](https://raw.githubusercontent.com/PozzettiAndrea/ComfyUI-MotionCapture/35c8dbc4de14448d4626a0f99e32c869371fea12/nodes/body_model/smpl_coco17_J_regressor.npy) | 0.447 | All modes | source asset |
| [smplx_verts437.npy](https://raw.githubusercontent.com/PozzettiAndrea/ComfyUI-MotionCapture/35c8dbc4de14448d4626a0f99e32c869371fea12/nodes/body_model/smplx_verts437.npy) | 0.003 | All modes | source asset |
| [realdenseface_vitb](https://drive.google.com/file/d/1eLlN_BCFMn0jSo0lNwBFqtI6igTh7Nc3/view) | 862.598 | RealDenseFace/FLAME | release archive |
| [facebox](https://drive.google.com/file/d/1eLlN_BCFMn0jSo0lNwBFqtI6igTh7Nc3/view) | 3.983 | RealDenseFace/FLAME | release archive |
| [flame_assets](https://drive.google.com/file/d/1eLlN_BCFMn0jSo0lNwBFqtI6igTh7Nc3/view) | 0.362 | RealDenseFace/FLAME | release archive |
| [flame2023](https://flame.is.tue.mpg.de/) | 50.567 | RealDenseFace/FLAME | registration download |
| [mhr2smplx_mapping.npz](https://raw.githubusercontent.com/facebookresearch/MHR/d96fafa33bbf018647c70c3525e91f53e79d2a14/tools/mhr_smpl_conversion/assets/mhr2smplx_mapping.npz) | 0.238 | RealDenseFace/FLAME | source asset |
| [head_hand_mask.npz](https://raw.githubusercontent.com/facebookresearch/MHR/d96fafa33bbf018647c70c3525e91f53e79d2a14/tools/mhr_smpl_conversion/assets/head_hand_mask.npz) | 0.212 | RealDenseFace/FLAME | source asset |
| [flame_landmark_embedding](https://raw.githubusercontent.com/georgeretsi/smirk/c7de404c4389f073906a6db1adabf62efcea3f35/assets/landmark_embedding.npy) | 0.030 | RealDenseFace/FLAME | source asset |
| [smpl_mean_params](https://raw.githubusercontent.com/PozzettiAndrea/ComfyUI-MotionCapture/35c8dbc4de14448d4626a0f99e32c869371fea12/nodes/hmr2/smpl_mean_params.npz) | 0.001 | All modes | source asset |

The three GVHMR files are the **converted safetensors** consumed by the reviewed Comfy port; upstream GVHMR `.ckpt` downloads are not interchangeable inputs to this wrapper. The manifest pins the Hugging Face conversion revision and verifies its hashes against the local files. The SAM conversion is similarly revision/hash pinned.

The RealDenseFace link is its author-published release archive: retain only `vitb.pth`, `face_box.pth` and `flame_assets.npz` from that archive. ViT-S is not used. FLAME2023 and SMPL-X neutral are separate official registration/download flows. Preserve the versions/checksums in the manifest; do not commit those files to this repository.

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
nightly_models/ultralytics/segm/yolov8m-seg.pt
nightly_models/motion_capture/body_models/smplx/SMPLX_NEUTRAL.npz
study/models/sam_3d_body_dinov3_bf16.safetensors
gvhmr_node/nodes/body_model/smplx2smpl_sparse.npz
gvhmr_node/nodes/body_model/smpl_coco17_J_regressor.npy
gvhmr_node/nodes/body_model/smplx_verts437.npy
study/vendor/RealDenseFace/weights/realdenseface/vitb.pth
study/vendor/RealDenseFace/weights/facebox/face_box.pth
study/vendor/RealDenseFace/weights/flame/flame_assets.npz
study/models/face/flame_downloads/FLAME2023/flame2023.pkl
study/vendor/MHR/tools/mhr_smpl_conversion/assets/mhr2smplx_mapping.npz
study/vendor/MHR/tools/mhr_smpl_conversion/assets/head_hand_mask.npz
study/vendor/smirk/assets/landmark_embedding.npy
gvhmr_node/nodes/hmr2/smpl_mean_params.npz
```

For example, the face model is read from `<study>/models/face/flame_downloads/FLAME2023/flame2023.pkl`; the adapter supplies that path to the fitter. The conventional upstream `weights/flame/flame2023.pkl` location is not a second required copy for this adapter. The RealDenseFace network and support assets retain their vendor-relative `weights/` locations because the face subprocess changes working directory to that checkout.

The manifest’s root keys describe inventory locations; they are **not newly implemented config variables**. Only the study root is configurable today. Supporting arbitrary model mount paths requires the deferred worker-portability change. Production can reproduce the fixed layout with deliberate mounts/symlinks until then.

## Assets that are already inside the selected checkpoint

The SAM converted safetensors includes the DINOv3 backbone, MHR rig data, and its pure-PyTorch MediaPipe face components. No separate MHR FBX/LOD package, MediaPipe task or DINOv3 weight download is required for this integration. SAM native face uses the same checkpoint. Hands are handled by SAM hand refinement; no separate hand model is downloaded here.

RealDenseFace sets `dino_pretrained=false` and loads its complete ViT-B checkpoint. Its FaceBox weights remain required because the inference class instantiates that detector even when the adapter supplies actor-specific face boxes. The MHR topology mapping and SMIRK embedding are small support arrays, not extra inference models.

The current GVHMR path uses SimpleVO. DPVO, DROID, Pixel3DMM weights and SMIRK network weights are outside this prototype.

## DVC preparation

1. Retrieve the exact listed files and verify SHA256 before adding them to the studio warehouse. Public direct URLs are provided where the source supports them; registration/archive entries require their respective preparation steps.
2. Preserve filenames and resolve DVC/source symlinks inside the deployed container. A symlink into an unmounted developer cache is insufficient.
3. Pin the DVC descriptors alongside the reviewed source pins and environment recipe. Build compiled dependencies separately as described in [OFFLINE.md](OFFLINE.md).
4. Test all enabled modes with external networking blocked. This audit did not run a fresh offline container.

Checksums were computed sequentially with 8 MiB reads of explicitly identified files, without recursive model/storage scans. The complete manifest has 17 entries. The four large Hugging Face checkpoints also match upstream SHA256 metadata.

To verify staged files without traversing the model directories:

```bash
python scripts/verify_models.py \
  --study-root /path/to/study \
  --nightly-models /path/to/ComfyUI_nightly/models \
  --gvhmr-node /path/to/ComfyUI-MotionCapture \
  --mode face --sha256
```

Use `--mode gvhmr` or `--mode hybrid` for smaller feature sets. Without `--sha256`, the utility checks existence and byte counts only. Hashing reads one selected file at a time.
