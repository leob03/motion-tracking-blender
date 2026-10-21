"""Run with the worker Python: offline_export.py KEPT_JOB_DIR '{"name": {option overrides}}'
Keep job intermediates with MOCAP_DEBUG_KEEP=DIR on the worker.
Re-run export_bundle on kept job intermediates with option variants; score plate fit and motion."""
import io, json, os, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
BRIDGE = Path(__file__).resolve().parents[1]
STUDY = Path(os.environ.get('MOCAP_STUDY_ROOT', str(BRIDGE.parent/'comparison_study')))
sys.path[:0] = [str(BRIDGE), str(STUDY)]
os.environ['MOCAP_STATUS_FILE'] = tempfile.mktemp()
import worker
root = Path(sys.argv[1]); variants = json.loads(sys.argv[2])
meta = json.loads((root/'inputs/shot/clip.json').read_text())
base = json.loads((root/'options.json').read_text())
for name, overrides in variants.items():
    out = Path(tempfile.mkdtemp())/'r.zip'
    worker.export_bundle(root, meta, dict(base, **overrides), out, study=STUDY)
    z = zipfile.ZipFile(out); m = json.loads(z.read('manifest.json')); a = np.load(io.BytesIO(z.read('animation.npz')))
    j = a['joints_world'].astype(float); fps = m['fps']; hip = j[:, [6, 7]].mean(1)
    acc = np.linalg.norm(hip[2:]-2*hip[1:-1]+hip[:-2], axis=1)*fps**2
    C = a['camera_to_world']; depth = np.einsum('fi,fi->f', hip-C[:, :3, 3], C[:, :3, 2])
    dacc = np.abs(depth[2:]-2*depth[1:-1]+depth[:-2])*fps**2
    up = np.array(m['scene_camera']['scene_up']); feet = j[:, [10, 11]]; h = feet@up
    speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1)*fps; contact = h[1:] < np.percentile(h, 20, axis=0)+0.03
    sc = m['scene_camera']
    from scipy.spatial.transform import Rotation
    root_rot = Rotation.from_matrix(a['rig_pose'][:, 0, :3, :3]/np.linalg.norm(a['rig_pose'][:, 0, :3, :3], axis=1, keepdims=True))
    omega = (root_rot[1:]*root_rot[:-1].inv()).as_rotvec()*fps
    ang_acc = np.degrees(np.linalg.norm(np.diff(omega, axis=0), axis=1)*fps)
    print(json.dumps(dict(variant=name, plate_mean_px=round(sc['plate_error_px']['mean_px'], 2), plate_p95_px=round(sc['plate_error_px']['p95_px'], 2),
        hip_accel_rms=round(float(np.sqrt((acc**2).mean())), 2), depth_accel_rms=round(float(np.sqrt((dacc**2).mean())), 2),
        foot_slide=round(float(np.median(speed[contact])), 3), root_ang_acc_rms_deg_s2=round(float(np.sqrt((ang_acc**2).mean())), 0), max_offset_m=round(sc['plate_lock']['max_offset_m'], 3))))
