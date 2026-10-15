import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation

out = Path(__file__).resolve().parents[1]/'test_results/synthetic/result.mocap.zip'
out.parent.mkdir(parents=True, exist_ok=True)
n = 8
cam = np.tile(np.eye(4), (n, 1, 1))
cam[:, :3, :3] = Rotation.from_euler('y', np.linspace(0, .15, n)).as_matrix()
cam[:, :3, 3] = np.stack([np.linspace(0, .2, n), np.ones(n)*1.5, np.zeros(n)], -1)
K = np.tile([[800., 0., 301.], [0., 700., 213.], [0., 0., 1.]], (n, 1, 1))
local = np.tile([[-.2, -.3, 4.], [.3, -.2, 4.2], [.1, .2, 3.8]], (n, 1, 1))
local[:, :, 0] += np.linspace(0, .1, n)[:, None]
world = np.einsum('fij,fvj->fvi', cam[:, :3, :3], local)+cam[:, None, :3, 3]
p = np.einsum('fij,fvj->fvi', K, local)
arr = io.BytesIO()
np.savez_compressed(arr, camera_to_world=cam, K=K, joints_world=world, vertices_world=world,
                    faces=np.array([[0, 1, 2]]), joints_projected_pixels=p[..., :2]/p[..., 2:3])
manifest = dict(schema='mocap-blender-bundle', version=1, method='fixture', frames=n, fps=30000/1001,
                width=640, height=480, source_start_frame=3, track_id='fixture_001',
                joint_names=['a', 'b', 'c'], camera_note='Synthetic camera for projection validation')
with zipfile.ZipFile(out, 'w') as z:
    z.writestr('manifest.json', json.dumps(manifest))
    z.writestr('animation.npz', arr.getvalue())
print(out)
