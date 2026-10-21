"""Run: python scene_camera_known_answer.py PRODUCTION_BRIDGE_DIR BUNDLE.mocap.zip
Known-answer test: a scene camera that is GVHMR's own camera moved by a rigid transform G."""
import sys, io, zipfile, numpy as np
sys.path.insert(0, sys.argv[1]); import scene_camera as sc
from scipy.spatial.transform import Rotation
z = zipfile.ZipFile(sys.argv[2]); a = np.load(io.BytesIO(z.read('animation.npz')))
cam, world = a['camera_to_world'], a['joints_world'].astype(float)
cam_joints = sc.to_camera(cam, world)                       # exact camera-space joints
G = np.eye(4); G[:3,:3] = Rotation.from_euler('xyz', [80, -30, 120], degrees=True).as_matrix(); G[:3,3] = [3, -1, 7]
known = G @ cam
rng = np.random.default_rng(0)
noisy = cam_joints * (1 + rng.normal(0, .03, (len(cam), 1, 1)))   # 3% per-frame depth jitter, like HMR depth
noisy[40:45] *= 1.4                                                   # five bad frames
R0, t0, info = sc.register(world, noisy, known)
print('rotation error deg', np.degrees(Rotation.from_matrix(R0 @ G[:3,:3].T).magnitude()).round(3), 'translation error m', np.linalg.norm(t0 - G[:3,3]).round(4), info)
reg = world @ R0.T + t0
K = np.broadcast_to(a['K'], (len(cam), 3, 3))
print('plate error before lock', sc.plate_error(K, known, reg, noisy))
off = sc.plate_lock(reg[:, [6,7]].mean(1), noisy[:, [6,7]].mean(1), known, 1., 0.)
print('plate error after lock ', sc.plate_error(K, known, reg + off[:, None], noisy))
depth_change = (sc.to_camera(known, (reg + off[:, None])[:, [6]])[..., 2] - sc.to_camera(known, reg[:, [6]])[..., 2])
print('max camera-depth change from lock (m)', float(np.abs(depth_change).max()))
print('tilt', sc.tilt_degrees(R0, G[:3,1]))
