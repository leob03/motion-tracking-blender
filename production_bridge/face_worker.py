"""Fit RealDenseFace/FLAME to the selected actor and attach a joined FLAME head.
Reuses the study's semantic alignment and lower-neck fairing, preserving the
SAM body/hands and taking head orientation from SAM rather than the face fit.
The head model is FLAME 2023 Open (CC-BY-4.0); it is placed with SAM 3D Body's
own face keypoints and their FLAME Open embedding (see face_assets.py).
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from progress import report
import face_assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--job-root', type=Path, required=True)
    parser.add_argument('--sam-variant', default='smooth', choices=['smooth', 'placed'],
                        help='SAM body the FLAME head is joined to; placed = SAM 3D Body on camera (locked proportions)')
    args = parser.parse_args()
    study, root = args.study.resolve(), args.job_root.resolve()
    lock = (study/'inference.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    started = time.perf_counter()
    sys.path.insert(0, str(study/'face_deps'))
    repo = study/'vendor/RealDenseFace'
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(study))
    os.chdir(repo)
    import common.fix_chumpy
    from tracker import RealDenseFaceInferencer, RealDenseFaceInferenceConfig, FaceBoxConfig
    from tracker import FLAMEConfig, GNFlameOptimizer, GNFlameOptimizerConfig, load_fitting_config
    from common.types import InferenceOutput, FittingOutput
    from join_neck import join

    base = root/'results/shot'
    out = base/'face'
    out.mkdir(exist_ok=True)
    meta = json.loads((root/'inputs/shot/clip.json').read_text())
    with (base/'sam/pose_raw.pkl').open('rb') as stream:
        pose = pickle.load(stream)  # Internal output of this job, never an uploaded pickle.
    people = [frame[0] for frame in pose['frames']]
    report('face', 'Loading RealDenseFace')
    estimator = RealDenseFaceInferencer(RealDenseFaceInferenceConfig(compile_model=False), FaceBoxConfig(half=False))
    cap = cv2.VideoCapture(str(root/'inputs/shot/input.mp4'))
    fields = {key: [] for key in ('vertex_coord', 'vertex_coord_log_var', 'vertex_depth', 'vertex_depth_log_var', 'face_bbox')}
    try:
        report('face', 'RealDenseFace facial prediction', 0, len(people), frames=len(people))
        for i, person in enumerate(people):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Missing facial source frame {i}')
            points = np.asarray(person['pred_face_keypoints_2d'])
            low, high = points.min(0), points.max(0)
            bbox = np.r_[low, high].astype(np.float32)
            if not np.isfinite(bbox).all() or min(high-low) < 4:
                raise ValueError(f'No usable selected-actor face crop at frame {i}')
            output = estimator.infer_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), face_bbox=bbox)
            for key in fields:
                value = getattr(output, key)
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                fields[key].append(value)
            report('face', 'RealDenseFace facial prediction', i+1, len(people), frames=len(people))
    finally:
        cap.release()
    inferred = time.perf_counter()
    del estimator, pose, people
    gc.collect()
    torch.cuda.empty_cache()
    prior = InferenceOutput(**{key: np.stack(value) for key, value in fields.items()}, image_width=meta['width'], image_height=meta['height'])
    flame_path = face_assets.flame_model_path(study)
    config = load_fitting_config('configs/fitting/video_offline.yaml', GNFlameOptimizerConfig,
                                 flame=FLAMEConfig(flame_model_path=str(flame_path)), device='cuda')
    optimizer = GNFlameOptimizer(config)
    report('face', 'Fitting FLAME expressions, jaw and eyes')
    with torch.no_grad():
        fit = optimizer.fit_sequence(prior, camera=None)
        x = fit.x.clone()
        expressions = optimizer.num_expressions
        x[:, expressions:expressions+6] = 0
        x[:, expressions+15:] = 0
        canonical, _ = optimizer.decode(FittingOutput(identity=fit.identity, x=x))
        neutral, _ = optimizer.decode(FittingOutput(identity=fit.identity, x=torch.zeros_like(x[:1])))
    canonical = canonical.cpu().numpy()
    neutral = neutral[0].cpu().numpy()
    head_faces = optimizer.faces.cpu().numpy()
    params = dict(identity=fit.identity.cpu().numpy(), x=fit.x.cpu().numpy(),
                  camera_fov_y=np.array(fit.camera_fov_y if fit.camera_fov_y is not None else np.nan),
                  canonical_vertices=canonical, expression_count=np.array(expressions),
                  parameter_format=np.array('RealDenseFace '+face_assets.FLAME_NAME+'; original fitted x, not MHR coefficients'),
                  flame_model=np.array(face_assets.FLAME_NAME))
    fitted = time.perf_counter()
    if not np.isfinite(canonical).all():
        raise ValueError('Nonfinite FLAME face fit')
    del optimizer, fit
    gc.collect()
    torch.cuda.empty_cache()

    with np.load(base/f'sam/{args.sam_variant}.npz', allow_pickle=False) as data:
        body = data['vertices_camera']
        body_faces = data['faces']
    report('face', 'Attaching FLAME head and joining the neck')
    # The same SAM face keypoints on both meshes: on SAM's body through SAM's own keypoint map, on the FLAME
    # neutral through their FLAME Open embedding. Only the stable ones (eye corners, nose) place the head.
    embed = face_assets.landmarks()
    stable = embed['stable']
    landmarks = np.einsum('kv,fvd->fkd', face_assets.face_keypoint_map(study, stable), body.astype(np.float32))
    tri = head_faces[embed['lmk_faces_idx'][stable]]
    A = (neutral[tri]*embed['lmk_bary_coords'][stable][:, :, None]).sum(1)
    anchor = A.mean(0)
    A = A-anchor
    B = landmarks
    centers = B.mean(1)
    B = B-centers[:, None]
    u, _, vt = np.linalg.svd(A.T[None]@B)
    fix = np.tile(np.eye(3), (len(body), 1, 1))
    fix[:, -1, -1] = np.linalg.det(u@vt)
    rotation = u@fix@vt
    scale = float(np.median(np.sqrt((B*B).sum((1, 2))/(A*A).sum())))
    head = (canonical-anchor)@rotation*scale+centers[:, None]
    with np.load(study/'vendor/MHR/tools/mhr_smpl_conversion/assets/head_hand_mask.npz') as masks:
        mask = masks['head_mask'] > .5
    cut_faces = body_faces[~mask[body_faces].any(1)]
    # Kept exactly as fitted during neck smoothing: the faces carrying SAM face keypoints and the jaw-skinned chin edge.
    support = embed['protected_vertices'].astype(np.int64)
    vertices, faces, neck = join(body, cut_faces, head, head_faces, rotation, support)
    assert np.array_equal(vertices[:, :len(body[0])], body)
    assert np.array_equal(vertices[:, len(body[0])+support], head[:, support])
    np.savez_compressed(out/'composite.npz', vertices_camera=vertices.astype(np.float32), faces=faces)
    params.update(flame_to_camera_rotation=rotation, flame_to_camera_scale=scale,
                  flame_neutral_anchor=anchor, mhr_head_anchor_camera=centers)
    np.savez_compressed(out/'parameters.npz', **params)
    residual = np.linalg.norm(A@rotation*scale-B, axis=-1)
    metrics = dict(method='RealDenseFace '+face_assets.FLAME_NAME+' joined head', frames=len(body),
                   flame_model=face_assets.FLAME_NAME, credit=face_assets.FLAME_CREDIT,
                   placement='SAM 3D Body face keypoints (eye corners, nose) matched to their FLAME Open embedding',
                   placement_residual_mm_median=float(np.median(residual)*1000),
                   dense_inference_seconds=inferred-started, fitting_seconds=fitted-inferred,
                   total_seconds=time.perf_counter()-started, scale=scale, neck_join=neck,
                   body_vertex_count=len(body[0]), head_vertex_count=len(head[0]),
                   protected_face_vertices=support.tolist(), body_max_change_m=0.,
                   crop_source='Selected SAM actor face keypoints on every frame',
                   attachment='SAM head orientation; fixed per-clip identity scale; FLAME expression/jaw/eye motion',
                   limitations='Composite mesh cache, not native MHR expressions. Face fit and identity can still differ from footage; no collision solve.')
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != 'protected_face_vertices'}), flush=True)


if __name__ == '__main__':
    main()
