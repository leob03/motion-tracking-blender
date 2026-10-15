"""Controlled translation-only hybrids; reuse predictions, do not rerun inference.

SAM articulation, fingers, scale and per-frame orientation are unchanged.
Export a per-frame parent transform usable with the original MHR animation.
These candidates borrow GVHMR estimates and are not independent SAM world solves.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

R = Path(__file__).resolve().parent

def load(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}

def rms_acceleration(points, fps):
    return float(np.sqrt(np.mean(np.diff(points, n=2, axis=0) ** 2)) * fps ** 2)

def project(points, K):
    homogeneous = points @ K.T
    return homogeneous[..., :2] / homogeneous[..., 2:3]

def build(clip):
    base = R / 'results' / clip
    g = load(base / 'gvhmr/native.npz')
    p = load(base / 'gvhmr/smpl_params_0001.npz')
    s = load(base / 'sam/smooth.npz')
    rot = Rotation.from_rotvec(p['global_orient']).as_matrix() @ Rotation.from_rotvec(p['global_orient_incam']).as_matrix().transpose(0, 2, 1)
    t = g['vertices_world'].astype(np.float64).mean(1) - np.einsum('fij,fj->fi', rot, g['vertices_camera'].astype(np.float64).mean(1))
    transform = lambda value: np.einsum('fij,fvj->fvi', rot, value) + t[:, None]
    inverse = lambda value: np.einsum('fji,fvj->fvi', rot, value - t[:, None])
    native_error = float(np.max(np.abs(transform(g['vertices_camera']) - g['vertices_world'])))
    assert native_error < 2e-5, native_error
    sam_world = transform(s['vertices_camera'])
    sam_joints_world = transform(s['joints_camera'])
    sam_hip_camera = s['joints_camera'][:, [6, 7]].mean(1)
    gv_hip_world = g['joints_world'][:, [6, 7]].mean(1)
    sam_hip_world = sam_joints_world[:, [6, 7]].mean(1)
    gv_hip_camera = g['joints_camera'][:, [6, 7]].mean(1)
    # Preserve the camera-space hip ray while replacing its estimated depth.
    assert np.all(sam_hip_camera[:, 2] > 0) and np.all(gv_hip_camera[:, 2] > 0)
    depth_target = sam_hip_camera * (gv_hip_camera[:, 2:3] / sam_hip_camera[:, 2:3])
    depth_delta = np.einsum('fij,fj->fi', rot, depth_target - sam_hip_camera)
    out = base / 'hybrid_candidates'
    out.mkdir(exist_ok=True)
    fps = float(g['fps'])
    diagnostics = []
    for key, delta in [('camera_only', np.zeros_like(t)), ('gvhmr_xyz', gv_hip_world - sam_hip_world), ('gvhmr_depth', depth_delta)]:
        verts = sam_world + delta[:, None]
        joints = sam_joints_world + delta[:, None]
        vc = inverse(verts)
        jc = inverse(joints)
        matrices = np.tile(np.eye(4), (len(verts), 1, 1))
        matrices[:, :3, :3] = rot
        matrices[:, :3, 3] = t + delta
        hip = joints[:, [6, 7]].mean(1)
        local_error = float(np.max(np.abs((verts - hip[:, None]) - (sam_world - sam_hip_world[:, None]))))
        assert local_error < 1e-6
        if key == 'gvhmr_xyz':
            assert np.max(np.abs(hip - gv_hip_world)) < 1e-6
        if key == 'gvhmr_depth':
            assert np.max(np.abs(project(jc[:, [6, 7]].mean(1), g['K']) - project(sam_hip_camera, g['K']))) < .01
        np.savez_compressed(out / f'{key}.npz', vertices_world=verts.astype(np.float32), joints_world=joints.astype(np.float32), faces=s['faces'], fps=fps)
        # For the existing MHR camera-space mesh/rig in meters; not its native cm basis.
        np.savez_compressed(out / f'{key}_placement.npz', world_from_camera=matrices, original_camera_to_world_R=rot, original_camera_to_world_t=t, additional_world_translation=delta, fps=fps,
            input_coordinates='Existing exported MHR camera-space animation in meters, OpenCV axes; includes pred_cam_t.',
            output_coordinates='GVHMR inferred Y-up world in meters',
            clip=clip, mode=key, body_source='SAM smooth', world_source='GVHMR; estimated, not ground truth')
        diagnostics.append(dict(mode=key, root_acceleration_rms_m_s2=rms_acceleration(hip, fps),
            initial_camera_depth_acceleration_rms_m_s2=rms_acceleration(hip @ rot[0, :, 2], fps),
            joint_reprojection_change_from_sam_px=float(np.linalg.norm(project(jc, g['K']) - project(s['joints_camera'], g['K']), axis=-1).mean()),
            maximum_root_relative_geometry_change_m=local_error,
            root_distance_from_gvhmr_mean_m=float(np.linalg.norm(hip-gv_hip_world, axis=-1).mean())))
    metrics = dict(clip=clip,frames=len(sam_world), fps=fps, native_camera_to_world_validation_max_m=native_error,
        gvhmr_root_acceleration_rms_m_s2=rms_acceleration(gv_hip_world,fps),candidates=diagnostics,
        limitation='No ground truth. Acceleration measures smoothness, and reprojection change measures deviation from SAM, not image accuracy. XYZ inherits GVHMR root failures; depth-only preserves the SAM hip ray, not the full mesh projection. No foot contact, common multi-actor solve, scale fitting or orientation replacement.')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('clips',nargs='+');options=parser.parse_args()
    for name in options.clips: build(name)
