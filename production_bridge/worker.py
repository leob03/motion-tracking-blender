"""Server-side adapter for the study's tested inference and fusion paths."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from progress import report
from media import Plate, VideoSink, processing_size
from rig_export import read_sam_rig, attach_flame_weights, preserve_fitted_surface
import scene_camera

DETECTOR = '/home/innovation/dev-lbringer/ComfyUI_nightly/models/ultralytics/segm/yolov8m-seg.pt'
# Bundle Y-up world -> Blender Z-up world (the importer applies the forward direction).
Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)


def status(root, stage):
    keys = {'Detecting and tracking the selected actor': 'prepare',
            'Running GVHMR inference': 'gvhmr', 'Running SAM inference': 'sam',
            'Fitting RealDenseFace / FLAME and joining the neck': 'face',
            'Building portable Blender bundle': 'export', 'Complete': 'complete'}
    report(keys[stage], stage)


def person_detector():
    """YOLO person boxes on the GPU when available (about 20x faster than the CPU path)."""
    import torch
    from ultralytics import YOLO
    model = YOLO(DETECTOR)
    gpu = torch.cuda.is_available()
    def detect(frame):
        result = model.predict(frame, classes=[0], verbose=False, device=0 if gpu else 'cpu', half=gpu)[0]
        return result.boxes.xyxy.cpu().numpy().astype(float).tolist()
    def close():
        nonlocal model
        model = None
        if gpu:
            torch.cuda.empty_cache()
    return detect, close


def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    inter = np.prod(np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])))
    return float(inter/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter, 1e-9))


def draw_actors(frame, seeds, chosen=None):
    preview = frame.copy()
    for i, b in enumerate(seeds):
        x1, y1, x2, y2 = np.asarray(b, dtype=int)
        color = (60, 220, 80) if i == chosen else (200, 180, 80)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 3)
        cv2.putText(preview, f'Actor {i}', (x1, max(y1-8, 25)), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    return preview


def actor_crop(frame, box, number, size=192):
    """Square close-up of one detected person (clean frame, small margin) with its number, as base64 JPEG."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    side = max(x2-x1, y2-y1)*1.08
    cx, cy = (x1+x2)/2, (y1+y2)/2
    left, top = int(max(0, cx-side/2)), int(max(0, cy-side/2))
    right, bottom = int(min(w, cx+side/2)), int(min(h, cy+side/2))
    crop = np.zeros((int(side)+1, int(side)+1, 3), np.uint8)
    ox, oy = int(left-(cx-side/2)), int(top-(cy-side/2))
    crop[oy:oy+bottom-top, ox:ox+right-left] = frame[top:bottom, left:right]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(crop, (0, 0), (44, 34), (40, 40, 40), -1)
    cv2.putText(crop, str(number), (6, 27), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2)
    ok, jpeg = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(jpeg.tobytes()).decode()


def detect_actors(source, frame_index, max_side, fps_override, output):
    """Preflight for the add-on: numbered person boxes on one frame, without inference."""
    plate = Plate(source, fps_override)
    frame = plate.frame(frame_index)
    if frame is None:
        raise ValueError(f'Frame {frame_index} is beyond the end of the video')
    source_h, source_w = frame.shape[:2]
    w, h = processing_size(source_w, source_h, max_side)
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    detect, close = person_detector()
    seeds = sorted(detect(frame), key=lambda b: (b[0]+b[2])/2)
    close()
    largest = int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in seeds])) if seeds else -1
    ok, jpeg = cv2.imencode('.jpg', draw_actors(frame, seeds, largest), [cv2.IMWRITE_JPEG_QUALITY, 85])
    result = dict(frame=frame_index, width=w, height=h, original_width=source_w, original_height=source_h,
                  frames=plate.count, fps=plate.fps, largest=largest,
                  boxes=[[x1/w, y1/h, x2/w, y2/h] for x1, y1, x2, y2 in seeds],
                  crops=[actor_crop(frame, box, i) for i, box in enumerate(seeds)],
                  preview_jpeg=base64.b64encode(jpeg.tobytes()).decode())
    Path(output).write_text(json.dumps(result))


def load_known_camera(path, frames, source_w, source_h, w, h):
    """Scene camera from the add-on: OpenCV camera-to-local (Z-up, metres) -> Y-up arrays at processing size."""
    camera = json.loads(Path(path).read_text())
    if camera.get('version') != 1:
        raise ValueError('Unsupported scene camera format')
    c2w = np.asarray(camera['c2w'], float).reshape(-1, 4, 4)
    intrinsics = np.asarray(camera['K'], float).reshape(-1, 4)
    if len(c2w) < frames or len(intrinsics) != len(c2w) or not np.isfinite(c2w).all() or not np.isfinite(intrinsics).all():
        raise ValueError(f'The scene camera covers {len(c2w)} frames; the job needs {frames}')
    if abs(camera['width']/camera['height']-source_w/source_h) > .01:
        raise ValueError(f"Scene camera aspect {camera['width']}x{camera['height']} does not match the video {source_w}x{source_h}")
    c2w, intrinsics = c2w[:frames], intrinsics[:frames]
    c2w = np.einsum('ij,fjk->fik', Y_UP_TO_Z_UP.T, c2w)
    c2w[:, :3, :3] = np.stack([np.linalg.svd(R)[0]@np.linalg.svd(R)[2] for R in c2w[:, :3, :3]])
    sx, sy = w/camera['width'], h/camera['height']
    K = np.zeros((frames, 3, 3))
    K[:, 0, 0], K[:, 1, 1] = intrinsics[:, 0]*sx, intrinsics[:, 1]*sy
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = intrinsics[:, 2]*sx, intrinsics[:, 3]*sy, 1
    up = Y_UP_TO_Z_UP[:3, :3].T@np.asarray(camera.get('scene_up_local', [0, 0, 1]), float)
    info = {key: camera.get(key) for key in ('name', 'parent', 'world_scale', 'timeline_start', 'source', 'distortion')}
    return c2w, K, dict(info, scene_up=up.tolist())


PREPARED_VERSION = 1
PREPARED_KEEP = 12


def prepared_plate(root, source, options, cache_dir=None):
    """Downscaled intermediate video and per-frame person boxes for the job range.

    Cached per plate range (the upload name is its content hash), so another actor of the same
    plate skips decoding, encoding and detection.
    """
    out = root / 'inputs' / 'shot'
    out.mkdir(parents=True)
    key = hashlib.sha256(json.dumps(dict(source=source.name, bytes=source.stat().st_size, start=options['start_frame'], count=options['max_frames'],
                                         side=options['max_side'], fps=options['fps_override'], version=PREPARED_VERSION), sort_keys=True).encode()).hexdigest()[:24]
    cached = Path(cache_dir)/key if cache_dir else None
    if cached is not None and (cached/'prepared.json').is_file():
        report('prepare', 'Reusing the prepared plate and person detections')
        shutil.copyfile(cached/'input.mp4', out/'input.mp4')
        os.utime(cached)
        return json.loads((cached/'prepared.json').read_text())
    plate = Plate(source, options['fps_override'])
    available = max(0, plate.count-options['start_frame'])
    expected = min(available, options['max_frames']) if options['max_frames'] else available
    detect, close = person_detector()
    detections = []
    writer = None
    try:
        for frame in plate.read(options['start_frame'], options['max_frames']):
            source_h, source_w = frame.shape[:2]
            w, h = processing_size(source_w, source_h, options['max_side'])
            if (w, h) != (source_w, source_h):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = VideoSink(out / 'input.mp4', plate.fps, w, h)
            detections.append(detect(frame))
            writer.write(frame)
            report('prepare', 'Detecting people on every frame', len(detections), expected or None, frames=expected or None)
    finally:
        close()
        if writer is not None:
            writer.close()
    if len(detections) < 8:
        raise ValueError('Select at least 8 decodable frames for temporal body inference')
    info = dict(fps=plate.fps, width=w, height=h, source_width=source_w, source_height=source_h, plate=plate.plate,
                name=plate.plate['pattern'] if plate.plate else source.name, detections=detections)
    if cached is not None:
        staging = cached.with_name(key+'.part')
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        shutil.copyfile(out/'input.mp4', staging/'input.mp4')
        (staging/'prepared.json').write_text(json.dumps(info))
        shutil.rmtree(cached, ignore_errors=True)
        staging.rename(cached)
        for old in sorted((d for d in cached.parent.iterdir() if d.is_dir() and not d.name.endswith('.part')), key=lambda d: d.stat().st_mtime)[:-PREPARED_KEEP]:
            shutil.rmtree(old, ignore_errors=True)
    return info


def video_frame(path, index):
    capture = cv2.VideoCapture(str(path))
    try:
        for _ in range(index):
            capture.grab()
        ok, frame = capture.read()
    finally:
        capture.release()
    return frame if ok else None


def prepare(root, source, options, study, camera_file=None, cache_dir=None):
    from character_selection import track
    status(root, 'Detecting and tracking the selected actor')
    info = prepared_plate(root, source, options, cache_dir)
    out = root / 'inputs' / 'shot'
    fps, w, h, source_w, source_h = info['fps'], info['width'], info['height'], info['source_width'], info['source_height']
    detections = info['detections']
    first_number = (info['plate']['first_frame'] if info['plate'] else 1)+options['start_frame']
    # The actor is picked on any frame of the range and tracked forwards and backwards from it.
    seed = min(max(int(options.get('seed_frame', 0)), 0), len(detections)-1)
    seeds = sorted(detections[seed], key=lambda b: (b[0] + b[2]) / 2)
    if not seeds:
        raise ValueError(f'No person detected on plate frame {first_number+seed}; pick another frame')
    index = options['actor_index']
    if options.get('actor_box'):
        # The add-on's preflight box (normalized) identifies the actor independently of ordering.
        chosen = np.asarray(options['actor_box'], float)*[w, h, w, h]
        overlaps = [iou(chosen, b) for b in seeds]
        if max(overlaps) < .3:
            raise ValueError(f'The actor picked in Detect Actors is not found on plate frame {first_number+seed}; detect again')
        index = int(np.argmax(overlaps))
    if index == -1:
        index = int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in seeds]))
    if index < 0 or index >= len(seeds):
        raise ValueError(f'Actor index {index} is unavailable; plate frame {first_number+seed} contains {len(seeds)} people')
    forward, forward_seen = track(seeds, detections[seed:])
    backward, backward_seen = track(seeds, detections[seed::-1])
    all_boxes = np.concatenate([backward[::-1][:-1], forward])
    observed = np.concatenate([backward_seen[::-1][:-1], forward_seen])
    boxes = all_boxes[:, index]
    seen = observed[:, index]
    gap = 0
    for i, found in enumerate(seen):
        gap = 0 if found else gap + 1
        if gap > 12:
            start = i-gap+1
            end = next((j for j in range(i, len(seen)) if seen[j]), len(seen))-1
            raise ValueError(f'Actor {index} not found on plate frames {first_number+start}-{first_number+end}: occlusion, leaving the frame '
                             'or fast motion. Shorten the range, or pick the actor on a frame near there with Detect Actors.')
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError('Actor tracking produced an empty box')
    mask_writer = VideoSink(out / 'boxes.mp4', fps, w, h, lossless=True)
    try:
        for b in boxes:
            mask = np.zeros((h, w, 3), np.uint8)
            x1, y1 = np.floor(b[:2]).astype(int)
            x2, y2 = np.ceil(b[2:]).astype(int)
            mask[y1:y2+1, x1:x2+1] = 255
            mask_writer.write(mask)
    finally:
        mask_writer.close()
    known_info = None
    if camera_file:
        c2w, K_frames, known_info = load_known_camera(camera_file, len(boxes), source_w, source_h, w, h)
        np.savez(out / 'known_camera.npz', c2w=c2w, K=K_frames)
        K = np.median(K_frames, axis=0)
        intrinsics_source = f"scene camera '{known_info['name']}'"
    else:
        focal = options['focal_px'] * w / source_w if options['focal_px'] else float(np.hypot(w, h))
        K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]])
        intrinsics_source = 'user focal in original-width pixels' if options['focal_px'] else 'assumed image diagonal focal; not calibrated'
    meta = dict(name='shot', source=info['name'], start_frame=options['start_frame'], frames=len(boxes), fps=fps,
                width=w, height=h, original_width=source_w, original_height=source_h,
                moving_camera=options['moving_camera'], boxes_xyxy=boxes.tolist(), K=K.tolist(),
                track_id=f'person_{index+1:03}', actor_index=index, intrinsics_source=intrinsics_source,
                tracking=f'Seeds on range frame {seed} sorted left to right; Hungarian IoU/velocity association forwards and backwards',
                seed_frame=seed, inferred_box_frames=np.flatnonzero(~observed[:, index]).tolist(), known_camera=known_info,
                plate=info['plate'])
    (out / 'clip.json').write_text(json.dumps(meta, indent=2))
    cv2.imwrite(str(root / 'actor_preview.jpg'), draw_actors(video_frame(out/'input.mp4', seed), seeds, index))
    return meta


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def rigid(R, t):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def place_in_scene(gv, options, known, reference_camera_joints):
    """Register GVHMR's world body into the scene camera world, then plate-lock it.

    Returns per-frame world transforms (F, 4, 4) applied on top of GVHMR's world, plus a report.
    """
    c2w, K = known['c2w'], known['K']
    R0, t0, registration = scene_camera.register(gv['joints_world'].astype(float), gv['joints_camera'].astype(float), c2w)
    registered = gv['joints_world']@R0.T+t0
    strength, sigma = options.get('plate_lock', 1.), options.get('plate_lock_smoothing', 3.)
    offset = scene_camera.plate_lock(registered[:, [6, 7]].mean(1), reference_camera_joints[:, [6, 7]].mean(1), c2w, strength, sigma)
    transforms = np.tile(rigid(R0, t0), (len(c2w), 1, 1))
    transforms[:, :3, 3] += offset
    distance = np.linalg.norm(offset, axis=1)
    return transforms, dict(registration=registration, gravity_tilt_degrees=scene_camera.tilt_degrees(R0, known['info']['scene_up']),
                            plate_error_before_lock_px=scene_camera.plate_error(K, c2w, registered, reference_camera_joints),
                            plate_lock=dict(strength=strength, smoothing_frames=sigma, max_offset_m=float(distance.max()), mean_offset_m=float(distance.mean())))


def export_bundle(root, meta, options, destination, timings=None, started=None, study=None):
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
    known = None
    if meta.get('known_camera'):
        stored = read_npz(root/'inputs/shot/known_camera.npz')
        known = dict(c2w=stored['c2w'], K=stored['K'], info=meta['known_camera'])
    arrays = dict(camera_to_world=camera, K=known['K'] if known else np.broadcast_to(gv['K'], (count, 3, 3)),
                  source_frames=np.arange(meta['start_frame'], meta['start_frame']+count),
                  joints_world=gv['joints_world'], vertices_world=gv['vertices_world'], faces=gv['faces'])
    scene = None
    if known:
        # The plate reference is the body the result shows: SAM's camera-space joints for the hybrid.
        reference = read_npz(base/'sam/smooth.npz')['joints_camera'] if options['method'] == 'hybrid' else gv['joints_camera']
        world_fix, scene = place_in_scene(gv, options, known, reference)
        scene['orientation_lock'] = options.get('orientation_lock', 0.)
        if options['method'] == 'gvhmr' and options.get('orientation_lock', 0.) > 0:
            # Turn the body about its placed hip toward GVHMR's own camera-space orientation.
            world_orient = world_fix[:, :3, :3]@Rotation.from_rotvec(params['global_orient']).as_matrix()
            turn = scene_camera.orientation_lock(world_orient, Rotation.from_rotvec(params['global_orient_incam']).as_matrix(), known['c2w'],
                                                 options['orientation_lock'], options.get('plate_lock_smoothing', 3.))
            hip = np.einsum('fij,fj->fi', world_fix[:, :3, :3], gv['joints_world'][:, [6, 7]].mean(1))+world_fix[:, :3, 3]
            world_fix[:, :3, :3] = turn@world_fix[:, :3, :3]
            world_fix[:, :3, 3] = np.einsum('fij,fj->fi', turn, world_fix[:, :3, 3]-hip)+hip
        arrays['camera_to_world'] = known['c2w']
        arrays['inferred_camera_to_world'] = world_fix[0]@camera
        for key in ('vertices_world', 'joints_world'):
            arrays[key] = (np.einsum('fij,fvj->fvi', world_fix[:, :3, :3], gv[key])+world_fix[:, None, :3, 3]).astype(np.float32)
    rig = None
    if options['method'] == 'gvhmr':
        rig = read_npz(base/'gvhmr/rig_native.npz')
        if known:
            rig['rig_pose'] = world_fix[:, None]@rig['rig_pose']
    if options['method'] == 'hybrid':
        variant = 'face' if options.get('face_capture') == 'sam_native' else 'smooth'
        sam = read_npz(base / f'sam/{variant}.npz')
        src = root_rotation(base / 'sam/smooth.glb', count)
        target = Rotation.from_rotvec(params['global_orient']).as_matrix()
        hip_world = gv['joints_world'][:, [6, 7]].mean(1)
        if known:
            target = world_fix[:, :3, :3]@target
            hip_world = np.einsum('fij,fj->fi', world_fix[:, :3, :3], hip_world)+world_fix[:, :3, 3]
            if options.get('orientation_lock', 0.) > 0:
                turn = scene_camera.orientation_lock(target, src, known['c2w'], options['orientation_lock'], options.get('plate_lock_smoothing', 3.))
                target = turn@target
        parent_R = target @ src.transpose(0, 2, 1)
        parent_t = hip_world - np.einsum('fij,fj->fi', parent_R, sam['joints_camera'][:, [6, 7]].mean(1))
        for key in ('vertices', 'joints'):
            arrays[key+'_world'] = (np.einsum('fij,fvj->fvi', parent_R, sam[key+'_camera']) + parent_t[:, None]).astype(np.float32)
        arrays['faces'] = sam['faces']
        fitcam = np.tile(np.eye(4), (count, 1, 1))
        fitcam[:, :3, :3], fitcam[:, :3, 3] = parent_R, parent_t
        if not known:
            arrays['actor_camera_to_world'] = fitcam
        rig = read_sam_rig(base/f'sam/{variant}.glb', count, fitcam)
        if options.get('face_capture') == 'realdenseface_flame':
            composite = read_npz(base/'face/composite.npz')
            arrays['vertices_world'] = (np.einsum('fij,fvj->fvi', parent_R, composite['vertices_camera'])+parent_t[:, None]).astype(np.float32)
            arrays['faces'] = composite['faces']
            attach_flame_weights(rig, sam['vertices_camera'][0], composite['vertices_camera'][0])
        if not known:
            actor_jc = np.einsum('fji,fvj->fvi', parent_R, arrays['joints_world']-parent_t[:, None])
            actor_p = np.einsum('fij,fvj->fvi', arrays['K'], actor_jc)
            arrays['actor_joints_projected_pixels'] = actor_p[..., :2]/actor_p[..., 2:3]
    C = arrays['camera_to_world']
    jc = np.einsum('fji,fvj->fvi', C[:, :3, :3], arrays['joints_world']-C[:, None, :3, 3])
    p = np.einsum('fij,fvj->fvi', arrays['K'], jc)
    arrays['joints_projected_pixels'] = p[..., :2]/p[..., 2:3]
    if known:
        scene['plate_error_px'] = scene_camera.plate_error(arrays['K'], C, arrays['joints_world'], reference)
    face_editing = None
    if options.get('face_capture') == 'realdenseface_flame':
        from flame_export import add_flame_controls
        if study is None:
            raise ValueError('The FLAME editing export requires the study model path')
        report('export', 'Building editable FLAME expressions, jaw and eyes')
        face_editing = add_flame_controls(arrays, rig, read_npz(base/'face/parameters.npz'), study)
        rig_error = face_editing['surface_reconstruction_error_m']
    else:
        rig_error = preserve_fitted_surface(rig, arrays['vertices_world'],
                    lambda done, total: report('export', 'Preserving fitted surface for editable bones', done, total, frames=count))
    arrays.update(rig)
    arrays_path = root / 'animation.npz'
    np.savez_compressed(arrays_path, **arrays)
    camera_note = ('Scene camera supplied by the artist; the body is GVHMR world motion placed with one rigid transform, plus a smoothed sideways plate lock.'
                   if known else 'GVHMR inferred camera, not calibrated environment matchmove. Hybrid actor-fit camera preserves only this actor projection; it is not a shared scene camera.')
    manifest = dict(schema='mocap-blender-bundle', version=2 if face_editing else 1, method=options['method'],
                    frames=count, fps=meta['fps'], width=meta['width'], height=meta['height'],
                    original_width=meta['original_width'], original_height=meta['original_height'],
                    source_start_frame=meta['start_frame'], track_id=meta['track_id'],
                    coordinates='right-handed Y-up world; meters; camera extrinsics map OpenCV camera to world',
                    joint_names=gv['joint_names'].tolist(), animation='animation.npz', source=meta['source'],
                    plate=meta.get('plate'), intrinsics_source=meta['intrinsics_source'], options=options,
                    camera_mode='scene' if known else 'inferred', scene_camera=dict(known['info'], **scene) if known else None,
                    camera_note=camera_note,
                    body_note='Native animated armature with skin weights and retained surface residuals. FLAME results include expression keys and jaw/eye bones.' if face_editing else 'Native body/finger rig with captured surface correctives; facial surface remains cached.',
                    face_editing=face_editing,
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


class WarmModel:
    """Client for run_model.py's warm server; starts it on demand and relays its log."""

    def __init__(self, method, study, env):
        self.method, self.study, self.env = method, study, env
        folder = Path(os.environ.get('MOCAP_WARM_DIR', f'/tmp/mocap-warm-{os.getuid()}'))
        folder.mkdir(mode=0o700, exist_ok=True)
        self.socket_path = folder/f'{method}.sock'
        self.pid_path = folder/f'{method}.pid'
        self.server_log = folder/f'{method}.log'

    def connect(self, timeout=None):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        client.connect(str(self.socket_path))
        return client

    def ensure(self):
        try:
            with self.connect(5) as client:
                client.sendall(b'{"ping": true}\n')
                if json.loads(client.makefile('r').readline()).get('ok'):
                    return False
        except (OSError, ValueError):
            pass
        idle = os.environ.get('MOCAP_WARM_IDLE_SECONDS', '900')
        with self.server_log.open('a') as log:
            server = subprocess.Popen([sys.executable, '-u', str(self.study/'run_model.py'), 'serve', self.method, '--socket', str(self.socket_path), '--idle-timeout', idle],
                                      stdout=log, stderr=subprocess.STDOUT, env=self.env, start_new_session=True)
        self.pid_path.write_text(str(server.pid))
        deadline = time.time()+600
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f'The warm {self.method} server exited; see {self.server_log}')
            try:
                with self.connect(5) as client:
                    client.sendall(b'{"ping": true}\n')
                    if json.loads(client.makefile('r').readline()).get('ok'):
                        return True
            except (OSError, ValueError):
                time.sleep(.5)
        raise RuntimeError(f'The warm {self.method} server did not start')

    def run(self, argv, job_root):
        started = self.ensure()
        print(f'[Mocap] {"Started" if started else "Reusing"} warm {self.method.upper()} server', flush=True)
        log = job_root/f'warm_{self.method}.log'
        log.touch()
        done = False
        try:
            with self.connect() as client, log.open('r') as tail:
                client.sendall((json.dumps(dict(argv=argv, status_file=os.environ['MOCAP_STATUS_FILE'], log_file=str(log)))+'\n').encode())
                client.settimeout(1)
                reply = b''
                while not reply.endswith(b'\n'):
                    sys.stdout.write(tail.read())
                    sys.stdout.flush()
                    try:
                        chunk = client.recv(65536)
                    except socket.timeout:
                        continue
                    if not chunk:
                        raise RuntimeError(f'The warm {self.method} server stopped during the job; see {self.server_log}')
                    reply += chunk
                sys.stdout.write(tail.read())
            result = json.loads(reply)
            done = True
            if not result.get('ok'):
                raise RuntimeError(f'{self.method.upper()} inference failed: {result.get("error")}')
        finally:
            if not done:
                # Cancelled or broken: do not leave the server computing a job nobody waits for.
                try:
                    os.kill(int(self.pid_path.read_text()), signal.SIGKILL)
                except (OSError, ValueError):
                    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-root', type=Path)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--options', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--camera', type=Path)
    parser.add_argument('--prepared-cache', type=Path, help='Reuse/keep prepared plates (intermediate video + person boxes)')
    parser.add_argument('--detect', type=int, metavar='FRAME', help='Only detect actors on this frame and write JSON to --output')
    parser.add_argument('--max-side', type=int, default=1280)
    parser.add_argument('--fps-override', type=float, default=0.)
    args = parser.parse_args()
    sys.path.insert(0, str(args.study))
    cv2.setNumThreads(2)
    if args.detect is not None:
        detect_actors(args.video, args.detect, args.max_side, args.fps_override, args.output)
        return
    os.environ.setdefault('MOCAP_STATUS_FILE', str(args.job_root/'status.json'))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    options = json.loads(args.options.read_text())
    options.setdefault('face_capture', 'none')
    started = time.perf_counter()
    timings = {}
    # Inference children inherit the server's original GPU selection.
    inference_env = os.environ.copy()
    meta = prepare(args.job_root, args.video, options, args.study, args.camera, args.prepared_cache)
    timings['prepare_seconds'] = time.perf_counter()-started
    methods = ['gvhmr', 'sam'] if options['method'] == 'hybrid' else ['gvhmr']
    for method in methods:
        stage_start = time.perf_counter()
        status(args.job_root, 'Running '+method.upper()+' inference')
        argv = [method, 'shot', '--work-root', str(args.job_root), '--sam-smoothing', str(options['sam_smoothing'])]
        if not options['hand_refinement']:
            argv.append('--no-hand-refinement')
        if method == 'sam' and options['face_capture'] == 'sam_native':
            argv.append('--face')
        if options.get('warm_models'):
            WarmModel(method, args.study, inference_env).run(argv, args.job_root)
        else:
            subprocess.run([sys.executable, str(args.study/'run_model.py')]+argv, check=True, env=inference_env)
        timings[method+'_seconds'] = time.perf_counter()-stage_start
    if options['face_capture'] == 'realdenseface_flame':
        stage_start = time.perf_counter()
        status(args.job_root, 'Fitting RealDenseFace / FLAME and joining the neck')
        # RealDenseFace's compiled fitter uses CUDA graph pool inspection, which
        # PyTorch 2.9 does not support with ComfyUI's cudaMallocAsync allocator.
        face_env = dict(inference_env, PYTORCH_ALLOC_CONF='backend:native', PYTORCH_CUDA_ALLOC_CONF='backend:native')
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('face_worker.py')), '--study', str(args.study), '--job-root', str(args.job_root)], check=True, env=face_env)
        timings['face_seconds'] = time.perf_counter()-stage_start
    export_bundle(args.job_root, meta, options, args.output, timings, started, study=args.study)
    keep = os.environ.get('MOCAP_DEBUG_KEEP')
    if keep:
        # Developer aid: keep inputs and intermediate results to re-run export_bundle offline.
        import shutil
        target = Path(keep)/args.output.parent.name
        shutil.copytree(args.job_root, target, ignore=shutil.ignore_patterns('*.pkl'))
        (target/'options.json').write_text(json.dumps(options))
        print('[Mocap] Kept job intermediates in', target, flush=True)


if __name__ == '__main__':
    main()
