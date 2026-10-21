# Current ComfyUI worker setup

This guide reproduces the **existing development architecture**. It is not a claim that copying the node directory alone creates a portable worker. Runtime adaptations were deliberately deferred for discussion with the shared DCC-bridge team.

## Three source roles

1. **API coordinator:** the team's Crafty ComfyUI fork runs `MocapProcessVideo`. It queues the job, streams status and returns a result descriptor. The development pin is in `provenance/source-pins.json`; the studio obtains this private source through its normal GitLab access.
2. **GVHMR process:** imports the reviewed `ComfyUI-MotionCapture` node port inside a separate older `ComfyUI_nightly` source tree. The original upstream GVHMR repository is a reference, not the direct runtime imported here.
3. **SAM process:** imports SAM body/hand/face and skeletal-export code from the pinned `comparison_study/vendor/ComfyUI` checkout. Do not assume the coordinator's installed SAM nodes are equivalent to this pin.

The models and optional RealDenseFace process run sequentially through the configured inference Python. The coordinator is launched with `--cpu` in the local script; this **does not** force worker inference onto CPU. The child scripts initialize CUDA.

## Source closure

Included project files are under `production_bridge/` and `comparison_study/`. The helper imports are:

- `character_selection.py`: seeded Hungarian box tracking.
- `build_root_orientation.py`: root orientation from the SAM skeletal GLB; imports `build_hybrid_candidates.py` for shared utilities.
- `join_neck.py`: fixed-topology neck joining/fairing for the optional FLAME head.
- `run_model.py`: launches the actual reviewed GVHMR/SAM ports and exports intermediate geometry/rigs.

The original study portal, videos, Blender review scenes, Pixel3DMM and SMIRK fitting workers are not required. SMIRK contributes only one landmark-embedding asset to the FLAME attachment. MHR contributes two topology assets; the standalone MHR inference package/FBX releases are not needed by this adapter.

External source commits and selected checksums are in [source-pins.json](../provenance/source-pins.json). A scoped RealDenseFace patch is included. It preserves the reviewed Turing-compatible BF16 guard and lazy creation of the disabled visibility rasterizer. Apply it to the recorded commit before building/testing. These are source pins and selected-file checks, not a full audit of every external worktree.

## Obtain the pinned external sources

For a new developer setup, first choose the checkout/study variables below. These commands create directories and clone source only; models and environments are staged separately. Use existing studio mirrors when appropriate.

```bash
mocap_checkout=/path/to/motion-tracking-blender
mocap_study="$mocap_checkout/comparison_study"
legacy_host=/home/innovation/dev-lbringer/ComfyUI_nightly
git clone https://github.com/comfyanonymous/ComfyUI.git "$legacy_host"
git -C "$legacy_host" checkout --detach ddc541ffdae0fe626de5a33192001f31c6ab93c6
git clone https://github.com/PozzettiAndrea/ComfyUI-MotionCapture.git "$legacy_host/custom_nodes/ComfyUI-MotionCapture"
git -C "$legacy_host/custom_nodes/ComfyUI-MotionCapture" checkout --detach 35c8dbc4de14448d4626a0f99e32c869371fea12
git clone https://github.com/Comfy-Org/ComfyUI.git "$mocap_study/vendor/ComfyUI"
git -C "$mocap_study/vendor/ComfyUI" checkout --detach 88ab4a06566454ad89db8f0bedb970d6c08cd1b7
git clone https://github.com/LinzhouLi/RealDenseFace.git "$mocap_study/vendor/RealDenseFace"
git -C "$mocap_study/vendor/RealDenseFace" checkout --detach 2821455af2c5d8996aec14000b37abf5dacc24f1
git clone https://github.com/facebookresearch/MHR.git "$mocap_study/vendor/MHR"
git -C "$mocap_study/vendor/MHR" checkout --detach d96fafa33bbf018647c70c3525e91f53e79d2a14
git clone https://github.com/georgeretsi/smirk.git "$mocap_study/vendor/smirk"
git -C "$mocap_study/vendor/smirk" checkout --detach c7de404c4389f073906a6db1adabf62efcea3f35
git -C "$mocap_study/vendor/RealDenseFace" apply \
  "$mocap_checkout/provenance/patches/realdenseface.patch"
```

The last three source checkouts are only needed for the optional FLAME path; MHR and SMIRK may instead be staged as the exact asset files listed in the model manifest. Source clone commands above are a setup recipe, not operations performed on your installed development repositories during this handoff. Do not run `git clone` into existing nonempty paths.

## Existing fixed layout

`MOCAP_STUDY_ROOT` and `MOCAP_PYTHON` are configurable. These paths **are still fixed in the reviewed source**:

| Path | Used by |
| --- | --- |
| `/home/innovation/dev-lbringer/ComfyUI_nightly` | `comparison_study/run_model.py`: GVHMR host and models root |
| Above + `custom_nodes/ComfyUI-MotionCapture` | Reviewed inference/body-model port |
| Above + `models/ultralytics/segm/yolov8m-seg.pt` | `worker.py`: CPU actor detection |
| Above + `models/motion_capture/body_models/smplx/SMPLX_NEUTRAL.npz` | `face_worker.py`: semantic landmark lookup |

Preserving those mount/symlink paths reproduces the current layout. Replacing them with configurable paths is a proposed follow-up, not a feature of this snapshot.

Relative to `MOCAP_STUDY_ROOT`, supply:

```text
run_model.py, character_selection.py, build_root_orientation.py,
build_hybrid_candidates.py, join_neck.py       # included
vendor/ComfyUI/                              # pinned SAM host
models/sam_3d_body_dinov3_bf16.safetensors
vendor/RealDenseFace/                        # optional face source + patch + built solver
face_deps/                                  # optional timm 1.0.30 overlay
models/face/flame_downloads/FLAME2023/flame2023.pkl
vendor/MHR/tools/mhr_smpl_conversion/assets/ # two listed topology assets
vendor/smirk/assets/landmark_embedding.npy
```

Keep `production_bridge/` and `comparison_study/` siblings when using the included helpers: `run_model.py` imports progress/rig export from its parent's `production_bridge` directory. The study root needs write access for `inference.lock`; this POSIX lock serializes this runtime's inference. It does not coordinate unrelated containers unless they share the same lock.

## Python, builds and models

Use Python **3.11.11**, PyTorch **2.9.1+cu128**, torchvision **0.24.1+cu128** as the measured baseline. See [offline/environment setup](OFFLINE.md) before attempting a fresh build. The development venv inherits packages from another environment and adds a face overlay; copying that venv is not a rebuild recipe. A clean minimal lock/container remains unvalidated.

Stage the exact [model inventory](MODELS.md). GVHMR uses SimpleVO for moving cameras in this adapter; there is no DPVO or DROID checkpoint requirement for this selected path. RealDenseFace additionally requires nvdiffrast and its local CUDA Gauss–Newton solver. Rebuild for deployment GPUs rather than copying this workstation's compiled binaries.

## Register the node in the coordinator

Keep the complete source checkout accessible to the worker process:

```bash
mocap_checkout=/path/to/motion-tracking-blender
comfy_checkout=/path/to/studio/ComfyUI
ln -s "$mocap_checkout/production_bridge/comfy_nodes"   "$comfy_checkout/custom_nodes/ComfyUI-MocapProduction"
export MOCAP_STUDY_ROOT="$mocap_checkout/comparison_study"
export MOCAP_PYTHON=/path/to/prepared/mocap/python
```

Do not overwrite an existing custom-node installation; update its pin through the team's normal deployment flow. The node resolves its own real path to find `worker.py`, so copying only its two Python files elsewhere breaks this relationship.

For the existing workstation, the reviewed runtime can still be used directly:

```bash
export MOCAP_COMFY_ROOT=/home/innovation/dev-lbringer/ComfyUI
export MOCAP_STUDY_ROOT=/mnt/share/dev-lbringer/markerless_mocap/comparison_study
export MOCAP_PYTHON="$MOCAP_STUDY_ROOT/.venv/bin/python"
export MOCAP_RUNTIME_DIR=/tmp/mocap-comfy
export MOCAP_PORT=8188
bash production_bridge/run_local_server.sh
```

A server may already own that port; do not start a duplicate. The launcher enables only `ComfyUI-MocapProduction`, binds loopback, caps CPU threads, and permits uploads up to 2048 MiB. In production, register the node and variables with the existing engine rather than blindly adopting these development launch flags.

A coordinator may use its own Python; the node's `MOCAP_PYTHON` is the executable for inference. The provided development launcher uses that executable for the coordinator too. The native 8188 server and all developer environments were left unchanged by this packaging task.

## Verify the fork integration

```bash
curl --fail http://127.0.0.1:8188/object_info/MocapProcessVideo
python production_bridge/tests/run_api_job.py   --endpoint http://127.0.0.1:8188 --video /path/to/test.mp4   --frames 24 --method gvhmr --out /tmp/mocap-smoke-gvhmr
```

Then run hybrid with face off, hybrid with SAM native face, and hybrid with RealDenseFace if that feature is deployed. Import each downloaded bundle in Blender. Check video alignment, visible cameras, pose edits, logs, cancellation and resume. The CLI test requires no Blender imports or model dependencies on the submitting machine.

The tested fork must accept movie upload through `/upload/image`, expose standard prompt/queue/history/view routes, and implement **job-scoped** `POST /api/jobs/{id}/cancel`. `/mocap/status/{id}` is supplied by this node. No core fork modifications were made here. Production auth, proxy limits and user/job ownership need coordination with the shared bridge team.

## FLAME editing update (0.4.0)

Deploy `production_bridge/flame_export.py` alongside the updated `worker.py`, and update the artist add-on to 0.4.0. The existing RealDenseFace/FLAME model assets and Python environment are reused. Each facial export verifies its model replay against the retained fit. New editable facial results use bundle v2; an older artist add-on will reject them. Non-FLAME jobs still use v1. See [editing and retained-capture upgrades](FLAME_EDITING.md).

## Worker update 0.6.0

Deploy `worker.py`, the new `media.py` and `scene_camera.py`, both `comfy_nodes/` files and `comparison_study/run_model.py` together, then restart ComfyUI. Models and the Python environment are unchanged.

- **Intermediate video:** the prepared clip is H.264 4:4:4 CRF 10 through PyAV (PSNR 45.9 dB vs 39.3 dB for the former OpenCV `mp4v`); the actor mask video is lossless. Movies are read sequentially from frame 0, so the start frame is exact for every codec.
- **Actor detection** runs on the GPU in half precision when CUDA is visible (CPU otherwise); its memory is released before inference.
- **Image sequences:** the add-on uploads a ZIP of display-referred JPEG frames plus `plate.json` (the ViPE add-on's proxy format); `media.Plate` reads movies and such archives alike.
- **Upload reuse:** uploads are named `<sha256>.<ext>`; `GET /mocap/upload/{name}` tells the add-on whether the worker already has that content.
- **Actor preview:** `POST /mocap/detect {video, frame, max_side, fps_override}` returns numbered person boxes (normalized) and a JPEG preview without using the prompt queue. The job input `actor_box` selects that actor by overlap instead of by left-to-right index.
- **Scene camera:** optional input `scene_camera` names an uploaded camera JSON (version 1; OpenCV camera-to-parent matrices in metres, Z-up, plus `[fx, fy, cx, cy]` per frame in plate pixels). GVHMR then uses the camera's rotations instead of SimpleVO, the camera intrinsics replace the estimate, and the body is placed in that camera's world (see the bundle doc). `plate_lock` (0–1), `orientation_lock` (0–1, default 0.5) and `plate_lock_smoothing` (frames, default 3) control the plate corrections. `MOCAP_DEBUG_KEEP=DIR` keeps each job's intermediates for offline tuning.
- **Focal fix:** SimpleVO now receives the job focal (it previously always assumed the image-diagonal focal, so a user focal reached GVHMR but not the odometry).

### Optional warm models

With the job input `warm_models=true`, `worker.py` hands GVHMR and SAM inference to long-lived `run_model.py serve` processes (one per model) over Unix sockets in `MOCAP_WARM_DIR` (default `/tmp/mocap-warm-<uid>`). They keep models in host RAM (about 6–7 GB per process) and release GPU memory after every job except the CUDA context (about 0.25 GB each), so they coexist with other GPU work. They exit after `MOCAP_WARM_IDLE_SECONDS` (default 900) without a job. Inference stays serialized by the same `inference.lock`; a cancelled job kills the server it was using. RealDenseFace still runs per job. Leave it off on shared pools where host RAM is scarce.
