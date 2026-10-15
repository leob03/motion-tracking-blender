# Generic DCC bridge handoff

This snapshot preserves the functioning Blender adapter and current ComfyUI transport. It does not implement or replace Geronimo's generic DCC bridge. The next interface changes should be agreed with the project owner first.

## Existing boundaries

| Component | Responsibility |
| --- | --- |
| `production_bridge/mocap_comfy/client.py` | Standard-library HTTP; check, upload, submit, monitor, cancel, download |
| `mocap_comfy/__init__.py` | Artist settings, background HTTP thread, main-thread event processing and import |
| `mocap_comfy/importer.py` | Bundle validation, movie background, camera/mesh creation and coordinate conversion |
| `mocap_comfy/rig.py` | Native bones, animation and skinning |
| `mocap_comfy/log_view.py` | Split/reuse/close the live log pane |
| `production_bridge/comfy_nodes/` | Queued job node, subprocess lifecycle, progress/log route, output checksum |
| `production_bridge/worker.py` | Detection/tracking, sequential backends, world/root fusion, portable export |
| `production_bridge/face_worker.py` | Optional RealDenseFace fit and FLAME attachment |

`client.py` has no Blender imports, but importing the **package** executes its Blender-dependent `__init__.py`. A non-Blender adapter must load this module directly or agree a package extraction. Bundle validation currently lives in the Blender-dependent importer; it has not been split into a DCC-neutral library.

The easiest integration seam is the existing `ComfyClient` operations. Replace that transport with the shared bridge while retaining validated bundle semantics and Blender import. Do not assume the camera-tracking repo's client/status/receipt formats match this implementation.

## Actual API contract

| Operation | Endpoint / payload |
| --- | --- |
| Check node/input support | `GET /object_info/MocapProcessVideo` |
| Upload video | `POST /upload/image`, multipart fields `image`, `type=input`, `subfolder=mocap` |
| Submit | `POST /prompt`, graph containing one `MocapProcessVideo` node |
| Queue/history | `GET /queue`, `GET /history/{prompt_id}` |
| Progress/log | `GET /mocap/status/{progress_id}?offset=N` |
| Cancel one prompt | `POST /api/jobs/{prompt_id}/cancel` with `{}`; fork-specific |
| Download | `GET /view?filename=result.mocap.zip&subfolder=mocap/{progress_id}&type=output` |

The upload route's name says image, but the tested fork accepts these movie bytes. Confirm that behavior, upload limits and any gateway body-size/time limits in production. Upload returns a relative path, not a worker executable path. The node rejects input paths outside its input directory.

There are **two IDs**: ComfyUI returns `prompt_id` for queue/history/cancellation; the client supplies a different UUID `progress_id` for the worker's output directory and log route. Preserve both. Resume recovers `progress_id` from the saved graph in queue/history.

The output node returns `ui.mocap_bundle` entries with `filename`, `subfolder`, `type`, `sha256`, `progress_id`. History exposes these under `outputs[...].mocap_bundle`. Downloads stream to a partial file, validate SHA256, then rename. The prototype limits bundles to 2 GiB.

The [example API graph](../production_bridge/workflow_api.json) lists the actual settings. This is API-format JSON, not a serialized visual workflow. Artist frame count defaults to 0; the node/example graph retain the earlier 120-frame default. Blender sends its explicit value. Import-only settings (timeline start, mesh/rig import, review camera, local movie path) stay in Blender.

## Progress and cancellation

The progress route returns state/stage, available `done`/`total`, elapsed time, heartbeat, optional ETA, `log_chunk`, `log_offset`, `log_size`, and time since the last log write. Offset is a byte cursor; reads are capped at 65,536 bytes. This version decodes each chunk with replacement on invalid UTF-8 boundaries. It has no log-generation/restart protocol. Queue/history polling is approximately every two seconds. The log/status route is custom-node-owned; it does not require core ComfyUI edits.

Cancellation must target the submitted prompt atomically. The worker creates a process group and terminates it, including child inference processes, on interruption/error. Never substitute a global `/interrupt` call on a shared server. This worker implementation uses POSIX process groups and `fcntl`; the **GPU worker is Linux**, even when Blender is on Windows.

Keep Blender responsive, modify scene data only on its main thread, preserve job-scoped cancellation, retain explicit frame/coordinate conventions and verify result checksums. Stopping monitoring is not the same as cancelling inference.

## Discuss before adapting

1. Shared bridge submission/capability/receipt interfaces; mapping the two IDs; ownership of uploads/downloads and cancellation.
2. Configurable worker source/model paths and a self-contained source package, replacing the documented development layout.
3. A dedicated reproducible worker environment/container, rebuilt CUDA extensions, and offline model staging via DVC.
4. Production authentication, job ownership, retention and source identity. The current node adds no per-user authorization; upload/background media have no source-SHA binding. A bundle checksum only verifies downloaded result bytes.
5. Windows testing, long-shot memory/cache size, reliable error/reconnect UX, and then production shot lookup / interactive actor selection.
6. Editable FLAME expression/jaw/eye controls, after preserving this baseline.

These are identified next steps, not implemented changes or a production-readiness claim. No messages were sent to coworkers automatically.
