"""World-motion quality proxies (no ground truth): hip acceleration and foot sliding during contact."""
import io, sys, zipfile, json, numpy as np
def load(p):
    z = zipfile.ZipFile(p); return json.loads(z.read('manifest.json')), np.load(io.BytesIO(z.read('animation.npz')))
for path in sys.argv[1:]:
    m, a = load(path); j = a['joints_world'].astype(float); fps = m['fps']
    hip = j[:, [6, 7]].mean(1)
    acc = np.linalg.norm(hip[2:]-2*hip[1:-1]+hip[:-2], axis=1)*fps**2
    feet = j[:, [10, 11]]                                # ankles, Y-up world
    # Up axis: GVHMR world is Y-up; in scene mode use the axis along which ankles vary least relative to hips.
    up = np.array(m.get('scene_camera', {}) and m['scene_camera'].get('scene_up') or [0, 1, 0], float)
    height = feet@up
    speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1)*fps
    contact = (height[1:] < np.percentile(height, 20, axis=0)+0.03)
    print(json.dumps(dict(bundle=path.split('/')[-2], camera=m.get('camera_mode', 'inferred'),
        hip_accel_rms_m_s2=round(float(np.sqrt((acc**2).mean())), 2), hip_accel_p95=round(float(np.percentile(acc, 95)), 2),
        foot_slide_in_contact_m_s=round(float(np.median(speed[contact])), 3))))
