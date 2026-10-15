"""Server-side adapter for the study's tested inference and fusion paths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from ultralytics import YOLO
from progress import report
from rig_export import read_sam_rig, attach_flame_weights, preserve_fitted_surface


def status(root, stage):
    keys = {'Detecting and tracking the selected actor': 'prepare',
            'Running GVHMR inference': 'gvhmr', 'Running SAM inference': 'sam',
            'Fitting RealDenseFace / FLAME and joining the neck': 'face',
            'Building portable Blender bundle': 'export', 'Complete': 'complete'}
    report(keys[stage], stage)


def prepare(root, source, options, study):
    from character_selection import track
    status(root, 'Detecting and tracking the selected actor')
    model = YOLO('/home/innovation/dev-lbringer/ComfyUI_nightly/models/ultralytics/segm/yolov8m-seg.pt')
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Cannot decode the uploaded video')
    fps = options['fps_override'] or cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Video has no usable frame rate; set the FPS override')
    out = root / 'inputs' / 'shot'
    out.mkdir(parents=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, options['start_frame'])
    available = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-options['start_frame'])
    expected = min(available, options['max_frames']) if options['max_frames'] else available
    detections = []
    writer = None
    try:
        while not options['max_frames'] or len(detections) < options['max_frames']:
            ok, frame = cap.read()
            if not ok:
                break
            source_h, source_w = frame.shape[:2]
            scale = min(1., options['max_side'] / max(source_h, source_w))
            w, h = int(source_w * scale) // 2 * 2, int(source_h * scale) // 2 * 2
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(str(out / 'input.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('Cannot write the prepared video')
                cv2.imwrite(str(root / 'first_frame.jpg'), frame)
            result = model.predict(frame, classes=[0], verbose=False, device='cpu')[0]
            detections.append(result.boxes.xyxy.cpu().numpy().tolist())
            writer.write(frame)
            report('prepare', 'Detecting and tracking the selected actor', len(detections), expected or None, frames=expected or None)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if len(detections) < 8:
        raise ValueError('Select at least 8 decodable frames for temporal body inference')
    seeds = sorted(detections[0], key=lambda b: (b[0] + b[2]) / 2)
    if not seeds:
        raise ValueError('No person detected in the first selected frame; change the video start')
    index = options['actor_index']
    if index == -1:
        index = int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in seeds]))
    if index < 0 or index >= len(seeds):
        raise ValueError(f'Actor index {index} is unavailable; first frame contains {len(seeds)} people')
    all_boxes, observed = track(seeds, detections)
    boxes = all_boxes[:, index]
    gap = 0
    for i, found in enumerate(observed[:, index]):
        gap = 0 if found else gap + 1
        if gap > 12:
            raise ValueError(f'Actor lost near source frame {options["start_frame"]+i}; shorten the range or change the initial actor')
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError('Actor tracking produced an empty box')
    mask_writer = cv2.VideoWriter(str(out / 'boxes.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not mask_writer.isOpened():
        raise RuntimeError('Cannot write actor masks')
    try:
        for b in boxes:
            mask = np.zeros((h, w, 3), np.uint8)
            x1, y1 = np.floor(b[:2]).astype(int)
            x2, y2 = np.ceil(b[2:]).astype(int)
            mask[y1:y2+1, x1:x2+1] = 255
            mask_writer.write(mask)
    finally:
        mask_writer.release()
    focal = options['focal_px'] * w / source_w if options['focal_px'] else float(np.hypot(w, h))
    meta = dict(name='shot', source=source.name, start_frame=options['start_frame'], frames=len(boxes), fps=fps,
                width=w, height=h, original_width=source_w, original_height=source_h,
                moving_camera=options['moving_camera'], boxes_xyxy=boxes.tolist(),
                K=[[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]],
                track_id=f'person_{index+1:03}', actor_index=index,
                intrinsics_source='user focal in original-width pixels' if options['focal_px'] else 'assumed image diagonal focal; not calibrated',
                tracking='First-frame seeds sorted left to right; Hungarian IoU/velocity association',
                inferred_box_frames=np.flatnonzero(~observed[:, index]).tolist())
    (out / 'clip.json').write_text(json.dumps(meta, indent=2))
    preview = cv2.imread(str(root / 'first_frame.jpg'))
    for i, b in enumerate(seeds):
        x1, y1, x2, y2 = np.asarray(b, dtype=int)
        color = (60, 220, 80) if i == index else (200, 180, 80)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
        cv2.putText(preview, f'Actor {i}', (x1, max(y1-8, 25)), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    cv2.imwrite(str(root / 'actor_preview.jpg'), preview)
    return meta


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def export_bundle(root, meta, options, destination, timings=None, started=None):
    from build_root_orientation import root_rotation
    status(root, 'Building portable Blender bundle')
    export_start = time.perf_counter()
    timings = dict(timings or {})
    base = root / 'results' / 'shot'
    gv = read_npz(base / 'gvhmr/native.npz')
    params = read_npz(base / 'gvhmr/smpl_params_0001.npz')
    R = Rotation.from_rotvec(params['global_orient']).as_matrix() @ Rotation.from_rotvec(params['global_orient_incam']).as_matrix().transpose(0, 2, 1)
    t = gv['vertices_world'].astype(np.float64).mean(1) - np.einsum('fij,fj->fi', R, gv['vertices_camera'].astype(np.float64).mean(1))
    count = meta['frames']
    camera = np.tile(np.eye(4), (count, 1, 1))
    camera[:, :3, :3], camera[:, :3, 3] = R, t
    native_error = float(np.max(np.abs(np.einsum('fij,fvj->fvi', R, gv['vertices_camera']) + t[:, None] - gv['vertices_world'])))
    if native_error > 5e-5:
        raise ValueError(f'GVHMR camera/world consistency check failed ({native_error} meters)')
    arrays = dict(camera_to_world=camera, K=np.broadcast_to(gv['K'], (count, 3, 3)),
                  source_frames=np.arange(meta['start_frame'], meta['start_frame']+count),
                  joints_world=gv['joints_world'], vertices_world=gv['vertices_world'], faces=gv['faces'])
    rig = read_npz(base/'gvhmr/rig_native.npz') if options['method'] == 'gvhmr' else None
    if options['method'] == 'hybrid':
        variant = 'face' if options.get('face_capture') == 'sam_native' else 'smooth'
        sam = read_npz(base / f'sam/{variant}.npz')
        src = root_rotation(base / 'sam/smooth.glb', count)
        target = Rotation.from_rotvec(params['global_orient']).as_matrix()
        parent_R = target @ src.transpose(0, 2, 1)
        parent_t = gv['joints_world'][:, [6, 7]].mean(1) - np.einsum('fij,fj->fi', parent_R, sam['joints_camera'][:, [6, 7]].mean(1))
        for key in ('vertices', 'joints'):
            arrays[key+'_world'] = (np.einsum('fij,fvj->fvi', parent_R, sam[key+'_camera']) + parent_t[:, None]).astype(np.float32)
        arrays['faces'] = sam['faces']
        fitcam = np.tile(np.eye(4), (count, 1, 1))
        fitcam[:, :3, :3], fitcam[:, :3, 3] = parent_R, parent_t
        arrays['actor_camera_to_world'] = fitcam
        rig = read_sam_rig(base/f'sam/{variant}.glb', count, fitcam)
        if options.get('face_capture') == 'realdenseface_flame':
            composite = read_npz(base/'face/composite.npz')
            arrays['vertices_world'] = (np.einsum('fij,fvj->fvi', parent_R, composite['vertices_camera'])+parent_t[:, None]).astype(np.float32)
            arrays['faces'] = composite['faces']
            attach_flame_weights(rig, sam['vertices_camera'][0], composite['vertices_camera'][0])
        actor_jc = np.einsum('fji,fvj->fvi', parent_R, arrays['joints_world']-parent_t[:, None])
        actor_p = np.einsum('fij,fvj->fvi', arrays['K'], actor_jc)
        arrays['actor_joints_projected_pixels'] = actor_p[..., :2]/actor_p[..., 2:3]
    jc = np.einsum('fji,fvj->fvi', R, arrays['joints_world']-t[:, None])
    p = np.einsum('fij,fvj->fvi', arrays['K'], jc)
    arrays['joints_projected_pixels'] = p[..., :2]/p[..., 2:3]
    rig_error = preserve_fitted_surface(rig, arrays['vertices_world'],
                lambda done, total: report('export', 'Preserving fitted surface for editable bones', done, total, frames=count))
    arrays.update(rig)
    arrays_path = root / 'animation.npz'
    np.savez_compressed(arrays_path, **arrays)
    manifest = dict(schema='mocap-blender-bundle', version=1, method=options['method'],
                    frames=count, fps=meta['fps'], width=meta['width'], height=meta['height'],
                    source_start_frame=meta['start_frame'], track_id=meta['track_id'],
                    coordinates='right-handed Y-up world; meters; camera extrinsics map OpenCV camera to world',
                    joint_names=gv['joint_names'].tolist(), animation='animation.npz', source=meta['source'],
                    intrinsics_source=meta['intrinsics_source'], options=options,
                    camera_note='GVHMR inferred camera, not calibrated environment matchmove. Hybrid actor-fit camera preserves only this actor projection; it is not a shared scene camera.',
                    body_note='Native animated armature with skin weights and per-frame pre-skin surface cache; body/finger bones are editable. Facial surface remains cached.',
                    rig=dict(type='MHR' if options['method']=='hybrid' else 'SMPL-X', bones=len(rig['rig_names']),
                             surface_reconstruction_error_m=rig_error,
                             note='Learned surface/face correctives are retained at captured frames, not recomputed for edited poses.'),
                    face_capture=options.get('face_capture', 'none'),
                    face_details=json.loads((base/'face/metrics.json').read_text()) if (base/'face/metrics.json').exists() else None,
                    tracking=meta['tracking'], inferred_box_frames=meta['inferred_box_frames'],
                    validation=dict(gvhmr_camera_world_max_m=native_error),
                    inference={method: json.loads((base/method/'run.json').read_text()) for method in (['gvhmr', 'sam'] if options['method']=='hybrid' else ['gvhmr'])})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        bundle.write(arrays_path, 'animation.npz')
        bundle.write(root/'actor_preview.jpg', 'actor_preview.jpg')
        bundle.write(root/'inputs/shot/clip.json', 'tracking.json')
        if (base/'face/parameters.npz').exists():
            bundle.write(base/'face/parameters.npz', 'face_parameters.npz')
        elif options.get('face_capture') == 'sam_native':
            with (base/'sam/expressions.npy').open('rb') as f:
                expression = np.load(f, allow_pickle=False)
            np.savez_compressed(root/'face_parameters.npz', mhr_expression=expression)
            bundle.write(root/'face_parameters.npz', 'face_parameters.npz')
        timings['export_seconds'] = time.perf_counter()-export_start
        if started is not None:
            timings['worker_total_seconds'] = time.perf_counter()-started
        manifest['timings'] = timings
        bundle.writestr('manifest.json', json.dumps(manifest, indent=2))
    status(root, 'Complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-root', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--options', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('MOCAP_STATUS_FILE', str(args.job_root/'status.json'))
    sys.path.insert(0, str(args.study))
    cv2.setNumThreads(2)
    options = json.loads(args.options.read_text())
    options.setdefault('face_capture', 'none')
    started = time.perf_counter()
    timings = {}
    # Ultralytics CPU detection changes CUDA_VISIBLE_DEVICES in this process.
    # Inference children must inherit the server's original GPU selection.
    inference_env = os.environ.copy()
    meta = prepare(args.job_root, args.video, options, args.study)
    timings['prepare_seconds'] = time.perf_counter()-started
    methods = ['gvhmr', 'sam'] if options['method'] == 'hybrid' else ['gvhmr']
    for method in methods:
        stage_start = time.perf_counter()
        status(args.job_root, 'Running '+method.upper()+' inference')
        command = [sys.executable, str(args.study/'run_model.py'), method, 'shot', '--work-root', str(args.job_root), '--sam-smoothing', str(options['sam_smoothing'])]
        if not options['hand_refinement']:
            command.append('--no-hand-refinement')
        if method == 'sam' and options['face_capture'] == 'sam_native':
            command.append('--face')
        subprocess.run(command, check=True, env=inference_env)
        timings[method+'_seconds'] = time.perf_counter()-stage_start
    if options['face_capture'] == 'realdenseface_flame':
        stage_start = time.perf_counter()
        status(args.job_root, 'Fitting RealDenseFace / FLAME and joining the neck')
        # RealDenseFace's compiled fitter uses CUDA graph pool inspection, which
        # PyTorch 2.9 does not support with ComfyUI's cudaMallocAsync allocator.
        face_env = dict(inference_env, PYTORCH_ALLOC_CONF='backend:native', PYTORCH_CUDA_ALLOC_CONF='backend:native')
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('face_worker.py')), '--study', str(args.study), '--job-root', str(args.job_root)], check=True, env=face_env)
        timings['face_seconds'] = time.perf_counter()-stage_start
    export_bundle(args.job_root, meta, options, args.output, timings, started)


if __name__ == '__main__':
    main()
