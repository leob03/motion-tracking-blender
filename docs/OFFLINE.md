# Environment and offline deployment notes

All model inference and compiled dependencies belong on the **Linux GPU worker**, not on Windows/Blender artist machines.

## What was measured

[environment.json](../provenance/environment.json) records the working Python/CUDA and important package versions. [environment-observed.txt](../provenance/environment-observed.txt) records 561 visible distributions as a troubleshooting reference. It is an observed broad development environment, **not a tested minimal installation lock**. Do not blindly install all those unrelated packages into the coordinator.

The current `.venv` was created with `--system-site-packages` over the `comfyui_nightly` environment. Body processing sees timm 1.0.19; the face worker prepends a `face_deps` directory with timm 1.0.30 for the DINOv3 architecture. This overlay is an actual dependency, not a spare study directory.

The baseline is Python 3.11.11, torch 2.9.1+cu128, torchvision 0.24.1+cu128, NumPy 2.3.5 and SciPy 1.15.0. The full package record includes Ultralytics, smplx, OpenCV, PyAV, safetensors, einops, chumpy, roma, trimesh, nvdiffrast and related Comfy dependencies. The inference scripts force FP32 because automatic half precision failed in the reviewed Turing hand-refinement path. Re-evaluate precision changes on the target hardware separately.

## Rebuild outline for the face option

Use the pinned RealDenseFace source plus the included patch, then follow its source build instructions. With CUDA-enabled PyTorch and a compatible CUDA 12.8 toolkit/compiler installed, the relevant steps are:

```bash
# In a dedicated worker environment, not Blender's Python:
python -m pip install --no-deps --target /path/to/study/face_deps timm==1.0.30
# Install remaining reviewed Python dependencies in that environment.
# Build nvdiffrast 0.4.0 for the target environment rather than copying its .so.
cd /path/to/study/vendor/RealDenseFace
TORCH_CUDA_ARCH_LIST='8.6;8.9'   python -m pip install -e ./cuda_extensions/flame_solver --no-build-isolation
```

The architecture list above is an **Ampere/Ada example**, not appropriate for every GPU. Use the studio's deployment GPU targets. The measured workstation was a 16 GB Quadro RTX 5000/Turing. Include the pinned source's `cuda_extensions/third_party/glm` headers, CUDA cuBLAS/cuSOLVER libraries and a build toolchain. Keep an editable solver's source accessible in the image, or build/install a wheel; do not leave an editable link to a developer's host path. Ensure `flame_solver` is importable from the built extension directory or its installed wheel.

`face_worker.py` forces PyTorch's native allocator for the face subprocess: the reviewed fitter queries CUDA graph pools, which failed under the coordinator's cudaMallocAsync allocator with torch 2.9. This is intentional. Disabling neural-model compilation does not remove the compiled FLAME optimizer dependency.

These steps describe the current requirements; a clean container build and equivalent output have **not** been demonstrated by this handoff. Agree a proper minimal recipe/pins with the engine team before deployment. Include working OpenCV video codecs and FFmpeg/FFprobe for the reviewed video stack.

## Model staging / DVC

Use [model-manifest.json](../provenance/model-manifest.json) as the asset list. Its paths are relative to three explicitly named roots. Preserve the checkpoint filenames expected by the existing code and source-resident auxiliary assets. Warehouse only the selected files, not a whole Hugging Face cache or shared model repository.

The 17 entries include source-shipped topology/regression assets as well as model weights. Checksums were computed sequentially from exact files on the working machine. SHA256 values identify the required bytes regardless of Git/DVC symlink layout. Resolve symlink targets inside the container; host-only targets will break. FLAME and SMPL-X come from their official download process, not this Git repository.

The selected SAM converted checkpoint contains DINOv3, MHR rig data and the face-landmarker components. No separate native MHR FBX/LOD model, DINOv3 checkpoint or MediaPipe task file is loaded by this port. RealDenseFace has `dino_pretrained=false` and loads its full ViT-B checkpoint. The loader still constructs FaceBox even though the adapter supplies a face crop, so include `face_box.pth`.

## Offline acceptance

The prototype was tested with existing staged assets, **not an enforced network-isolated container**. Before deployment, stage the complete source/model/build outputs, block external networking, and exercise all enabled modes on representative footage. Check for unexpected downloads, missing source assets, path assumptions and runtime compilation. Do not assume `HF_HUB_OFFLINE` alone proves no network attempts.

Use local scratch for `TMPDIR`, bounded CPU threads, and sequential inference. Inputs/results/history retention and container-wide GPU scheduling are deployment responsibilities. Long shots can use substantial host memory and produce large per-frame mesh caches; the current client caps downloads at 2 GiB.
