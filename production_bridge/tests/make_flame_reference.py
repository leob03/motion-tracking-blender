"""Independent canonical FLAME forward reference for Blender edit validation."""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rigid_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flame_export import model_assets

p = argparse.ArgumentParser()
p.add_argument('--bundle', type=Path, required=True)
p.add_argument('--study', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
with zipfile.ZipFile(a.bundle) as z:
    with np.load(io.BytesIO(z.read('face_parameters.npz')), allow_pickle=False) as d:
        params = {k: d[k] for k in d.files}
    with np.load(io.BytesIO(z.read('animation.npz')), allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
E = int(params['expression_count'])
neutral, joints, dirs, weights = model_assets(a.study, params['identity'], E)
def forward(expression, rotations):
    rest = torch.from_numpy(neutral+np.einsum('e,evc->vc', expression, dirs))
    _, T = batch_rigid_transform(torch.from_numpy(rotations[None]), torch.from_numpy(joints[None]), torch.tensor([-1, 0, 1, 1, 1]), dtype=torch.float64)
    blend = torch.einsum('vj,jab->vab', torch.from_numpy(weights), T[0])
    return (torch.einsum('vab,vb->va', blend[:, :3, :3], rest)+blend[:, :3, 3]).numpy()
world = arrays['actor_camera_to_world'][:, :3, :3] @ params['flame_to_camera_rotation'].transpose(0, 2, 1)*float(params['flame_to_camera_scale'])
frames = sorted(set([0, len(params['x'])//2, len(params['x'])-1]))
references = []
for f in frames:
    expression = params['x'][f, :E].copy()
    rotations = np.tile(np.eye(3), (5, 1, 1))
    rotations[2:] = Rotation.from_rotvec(params['x'][f, E+6:E+15].reshape(3, 3)).as_matrix()
    baseline = forward(expression, rotations)
    cases=[]
    e = expression.copy(); e[0] += .75
    cases.append(forward(e, rotations)-baseline)
    for joint, axis in [(2, 'x'), (3, 'y'), (4, 'y')]:
        r = rotations.copy();r[joint] = r[joint]@Rotation.from_euler(axis, 8, degrees=True).as_matrix()
        cases.append(forward(expression, r)-baseline)
    references.append(np.stack(cases)@world[f].T)
a.output.parent.mkdir(parents=True, exist_ok=True)
camera=(params['canonical_vertices']-params['flame_neutral_anchor'])@params['flame_to_camera_rotation']*float(params['flame_to_camera_scale'])+params['mhr_head_anchor_camera'][:,None]
raw=np.einsum('fij,fvj->fvi',arrays['actor_camera_to_world'][:,:3,:3],camera)+arrays['actor_camera_to_world'][:,None,:3,3]
face_support=np.max(np.linalg.norm(raw-arrays['vertices_world'][:,-len(neutral):],axis=-1),axis=0)<2e-5
np.savez_compressed(a.output, frames=frames, deltas=np.stack(references), face_support=face_support)
print(json.dumps({'reference_frames':frames,'cases':['expression_001_plus_0.75','jaw_8deg','left_eye_8deg','right_eye_8deg']}))
