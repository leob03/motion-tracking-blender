"""Synthetic check of sam_camera.solve_placement (no models, no GPU).

A 12-joint body with locked proportions walks at 30 fps, seen by a camera that dollies sideways. SAM's per-frame guess
scales size and distance together (invisible in 2D) and adds a 2 % size-estimate error, so the locked body's implied
depth jitters. The placement must remove most of that jitter, stay within the plate budget, find the planted feet and
recover the scene scale (1) from feet plus camera travel.

python production_bridge/tests/sam_camera_placement.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sam_camera  # noqa: E402

rng = np.random.default_rng(1)
F, fps, W, H = 120, 30., 1280, 720
f = 1100.
K = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1.]])
# Body (camera axes: x right, y down, z forward) relative to the hip centre: shoulders, elbows, wrists, hips, knees, ankles.
template = np.array([[-.18, -.5, 0], [.18, -.5, 0], [-.22, -.25, 0], [.22, -.25, 0], [-.24, 0, 0], [.24, 0, 0],
                     [-.1, 0, 0], [.1, 0, 0], [-.1, .45, 0], [.1, .45, 0], [-.1, .9, 0], [.1, .9, 0]])
# World: Y up, camera at height 1.0 (feet in frame) looking along +Z (camera y down = world -Y), dollying along +X at 0.5 m/s.
c2w = np.tile(np.eye(4), (F, 1, 1))
c2w[:, :3, :3] = np.diag([1., -1., 1.])
c2w[:, 0, 3] = np.arange(F)/fps*.5
c2w[:, 1, 3] = 1.0
# Hip walks along +X at 0.8 m/s, 4 m away; feet planted alternately (stance phases keep an ankle fixed in the world).
t = np.arange(F)/fps
hip_world = np.stack([t*.8-.5, np.full(F, .9), np.full(F, 4.)], 1)
joints_world = hip_world[:, None]+template*[1, -1, 1]
step = (t*1.6).astype(int)  # 0.625 s per step
for i in range(F):
    side = 10+step[i] % 2
    start = np.flatnonzero(step == step[i])[0]
    joints_world[i, side, 0] = joints_world[start, side, 0]  # planted ankle stays put during its stance phase
R, T = c2w[:, :3, :3], c2w[:, :3, 3]
truth = np.einsum('fji,fvj->fvi', R, joints_world-T[:, None])  # true camera-space joints
hip = truth[:, 6:8].mean(1)
body = truth-hip[:, None]  # the locked body: one size, true proportions
# SAM's per-frame guess: size and distance scaled together (n, invisible in 2D) plus a 2 % size-estimate error (m).
n = np.exp(rng.normal(0, .04, F))
m = np.exp(rng.normal(0, .02, F))
translation = hip*n[:, None]
sam_joints = body*(n*m)[:, None, None]+translation[:, None]
sam_size = sam_camera.body_size(sam_joints)
pix = lambda x: (x@K.T)[..., :2]/(x@K.T)[..., 2:3]
ankles_2d = pix(sam_joints)[:, 10:12]
extent = np.percentile(np.linalg.norm(pix(sam_joints)-pix(translation[:, None]), axis=-1), 90, axis=1)

q, s, contacts, report = sam_camera.solve_placement(body, translation, sam_size, c2w, fps, ankles_2d, W, H, scale_free=True, extent_px=extent)
placed = body+(q[:, None]*translation)[:, None]
depth_error_before = np.abs(translation[:, 2]/(n*m)-hip[:, 2])  # SAM-implied depth for the locked body
depth_error_after = np.abs(s*q*translation[:, 2]-hip[:, 2])
plate = np.linalg.norm(pix(placed)-pix(sam_joints), axis=-1)
print(f"depth error (cm): before {depth_error_before.mean()*100:.1f}, after {depth_error_after.mean()*100:.1f}; "
      f"jitter {report['jitter_cm_before']:.1f} -> {report['jitter_cm_after']:.1f} cm; plate shift p95 {np.percentile(plate, 95):.2f} px; "
      f"contacts {report['contact_frames']}; scale {s:.3f}")
assert abs(np.log(q)).max() > 1e-3, 'the solver did not move from the identity start'
assert report['jitter_cm_after'] < .6*report['jitter_cm_before'], 'depth jitter not reduced'
assert depth_error_after.mean() < .7*depth_error_before.mean(), 'depth not closer to the truth'
assert np.percentile(plate, 95) < .01*W, 'plate shift beyond 1 % of the width'
assert sum(report['contact_frames']) > F//3, 'planted feet not found'
# run_model.py starts from the refit depth (log factors exactly 0) and anchors at the visible keypoints: the solver
# must still move (SciPy's first trust region is |x0|; it once stalled there).
rest_hip = translation/(n*m)[:, None]
rest = body+rest_hip[:, None]
anchor = rest.mean(1)
r, s2, _, report2 = sam_camera.solve_placement(rest-anchor[:, None], anchor, sam_camera.body_size(rest), c2w, fps, ankles_2d, W, H,
                                               scale_free=True, extent_px=extent)
print(f"identity start: jitter {report2['jitter_cm_before']:.1f} -> {report2['jitter_cm_after']:.1f} cm, scale {s2:.3f}")
assert abs(np.log(r)).max() > 1e-3 and report2['jitter_cm_after'] < .6*report2['jitter_cm_before'], 'stalled at the identity start'
print('SAM-on-camera placement: OK')
