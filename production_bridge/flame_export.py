"""Export native FLAME expression keys and jaw/eye bones, preserving the fit.

Only the worker reads the licensed model. Blender receives numeric deformation
data; it needs neither the model pickle nor an inference runtime.
"""
import importlib.util
from pathlib import Path
import pickle

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation
from rig_export import compact_weights, preserve_fitted_surface

BONES = ['FLAME_root', 'FLAME_jaw', 'FLAME_eye.L', 'FLAME_eye.R']


def model_assets(study, identity, expressions):
    shim = Path(study)/'vendor/RealDenseFace/common/fix_chumpy.py'
    spec = importlib.util.spec_from_file_location('flame_compat', shim)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Trusted server asset only; never unpickle an uploaded result.
    with (Path(study)/'models/face/flame_downloads/FLAME2023/flame2023.pkl').open('rb') as f:
        data = pickle.load(f, encoding='latin1')
    directions = np.asarray(data['shapedirs'], dtype=np.float64)
    neutral = np.asarray(data['v_template']) + np.einsum('vce,e->vc', directions[..., :300], identity)
    joints = np.asarray(data['J_regressor'] @ neutral)
    return neutral, joints, directions[..., 300:300+expressions].transpose(2, 0, 1), np.asarray(data['weights'])


def add_flame_controls(arrays, rig, params, study):
    """Add four bones and expression directions, then rebuild the residual cache.

The RDF fitter fixes joints from identity (not expression), and its reviewed
CUDA build has USE_POSE_BS=false. Match those conventions exactly.
"""
    if any(k.startswith('flame_') for k in arrays):
        raise ValueError('This bundle already has editable FLAME data')
    frames = len(arrays['vertices_world'])
    expressions = int(params['expression_count'])
    x = params['x']
    if x.shape != (frames, expressions+18):
        raise ValueError('Unsupported FLAME parameter layout')
    neutral, joints, dirs, weights = model_assets(study, params['identity'], expressions)
    count = len(neutral)
    offset = arrays['vertices_world'].shape[1]-count
    if offset <= 0 or weights.shape != (count, 5):
        raise ValueError('Unsupported joined FLAME topology')
    rot = Rotation.from_rotvec(x[:, expressions+6:expressions+15].reshape(-1, 3)).as_matrix().reshape(frames, 3, 3, 3)
    # Independent replay against the retained solver output catches convention
    # changes, different models or a fitter that enables pose blendshapes.
    shaped = neutral + np.einsum('fe,evc->fvc', x[:, :expressions], dirs)
    decoded = shaped*(weights[:, 0]+weights[:, 1])[None, :, None]
    for j in range(3):
        posed = np.einsum('fij,fvj->fvi', rot[:, j], shaped-joints[j+2])+joints[j+2]
        decoded += weights[None, :, j+2, None]*posed
    replay_error = float(np.max(np.abs(decoded-params['canonical_vertices'])))
    if replay_error > 2e-5:
        raise ValueError(f'FLAME model/decoder differs from retained predictions ({replay_error:g} m)')

    scale = float(params['flame_to_camera_scale'])
    camera = np.tile(np.eye(4), (frames, 1, 1))
    camera[:, :3, :3] = params['flame_to_camera_rotation'].transpose(0, 2, 1)*scale
    camera[:, :3, 3] = params['mhr_head_anchor_camera']-np.einsum('fij,j->fi', camera[:, :3, :3], params['flame_neutral_anchor'])
    world = arrays['actor_camera_to_world'] @ camera
    names = rig['rig_names'].tolist()
    if 'c_head' not in names or any(n in names for n in BONES):
        raise ValueError('Expected an unmodified MHR head rig')
    head_index = names.index('c_head')
    # FLAME and MHR influences must share a rest frame at the collar. Using the
    # first world-space head transform as the rest frame mixes incompatible
    # skin rotations, hiding distortions in the residual until the jaw is edited.
    head_skin = rig['rig_pose'][0, head_index] @ np.linalg.inv(rig['rig_rest'][head_index])
    H = np.linalg.inv(head_skin) @ world[0]
    singular = np.linalg.svd(H[:3, :3], compute_uv=False)
    if singular.max()/singular.min() > 1.0001:
        raise ValueError('Nonuniform FLAME rest scaling requires a different bone basis')
    raw_head = np.einsum('fij,fvj->fvi', world[:, :3, :3], params['canonical_vertices'])+world[:, None, :3, 3]
    fitted_head = arrays['vertices_world'][:, offset:]
    # Protect the collar itself; the faired upper neck must still follow the
    # jaw. Freezing the entire fairing band creates a crease under the chin.
    head_faces = arrays['faces'][(arrays['faces'] >= offset).all(1)]-offset
    edges = np.concatenate([head_faces[:, [0, 1]], head_faces[:, [1, 2]], head_faces[:, [2, 0]]])
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    boundary = edges[counts == 1]
    neighbors = {}
    for a, b in boundary:
        neighbors.setdefault(int(a), set()).add(int(b)); neighbors.setdefault(int(b), set()).add(int(a))
    loops = []
    remaining = set(neighbors)
    while remaining:
        front = [min(remaining)]; component = set(front); remaining.difference_update(front)
        while front:
            fresh = set().union(*(neighbors[v] for v in front)) & remaining
            component.update(fresh); remaining.difference_update(fresh); front = list(fresh)
        loops.append(np.array(sorted(component)))
    if not loops:
        raise ValueError('FLAME neck boundary is missing')
    neck = min(loops, key=lambda ids: neutral[ids, 1].mean())
    lengths = np.linalg.norm(neutral[edges[:, 0]]-neutral[edges[:, 1]], axis=1)
    graph = sparse.coo_matrix((np.r_[lengths, lengths], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(count, count)).tocsr()
    distance = dijkstra(graph, directed=False, indices=neck, min_only=True)
    fade = np.clip((distance-.015)/.05, 0., 1.)
    fade = fade*fade*(3-2*fade)

    old_count = len(rig['rig_names'])
    rests, poses = [], []
    for j, pivot in enumerate([joints[1], joints[2], joints[3], joints[4]]):
        local = np.eye(4); local[:3, 3] = pivot
        rests.append(H @ local)
        pose = np.tile(local, (frames, 1, 1))
        if j:
            pose[:, :3, :3] = rot[:, j-1]
        poses.append(world @ pose)
    rig['rig_names'] = np.array(names+BONES)
    rig['rig_parents'] = np.r_[rig['rig_parents'], names.index('c_head'), old_count, old_count, old_count].astype(np.int32)
    rig['rig_rest'] = np.concatenate([rig['rig_rest'], rests])
    rig['rig_pose'] = np.concatenate([rig['rig_pose'], np.stack(poses, axis=1)], axis=1)
    dense = np.zeros((count, old_count+4))
    rows = np.arange(count)[:, None]
    np.add.at(dense, (rows, rig['rig_indices'][offset:]), rig['rig_weights'][offset:]*(1-fade[:, None]))
    dense[:, old_count] = fade*(weights[:, 0]+weights[:, 1])
    dense[:, old_count+1:] = fade[:, None]*weights[:, 2:]
    ids, values = compact_weights(dense)
    width = max(ids.shape[1], rig['rig_indices'].shape[1])
    def pad(a): return np.pad(a, ((0, 0), (0, width-a.shape[1])))
    rig['rig_indices'] = np.concatenate([pad(rig['rig_indices'][:offset]), pad(ids)])
    rig['rig_weights'] = np.concatenate([pad(rig['rig_weights'][:offset]), pad(values)])
    error = preserve_fitted_surface(rig, arrays['vertices_world'])
    rest_dirs = np.einsum('ij,evj->evi', H[:3, :3], dirs)*fade[None, :, None]
    rig['rig_vertices'][:, offset:] -= np.einsum('fe,evc->fvc', x[:, :expressions], rest_dirs).astype(np.float32)
    q = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_quat().reshape(frames, 3, 4)[..., [3, 0, 1, 2]]
    for f in range(1, frames):
        q[f] *= np.where((q[f-1]*q[f]).sum(-1) < 0, -1., 1.)[:, None]
    arrays.update(flame_expression_dirs=rest_dirs.astype(np.float32),
                  flame_expressions=x[:, :expressions].astype(np.float32),
                  flame_vertex_offset=np.array(offset, np.int32),
                  flame_neck_mask=fade.astype(np.float32),
                  flame_bone_quaternions=q.astype(np.float32))
    return dict(version=1, model='FLAME2023', expressions=expressions,
                bones=BONES[1:], solver_replay_max_m=replay_error,
                surface_reconstruction_error_m=error,
                protected_neck_vertices=int((fade==0).sum()),
                note='Native expression shape keys and jaw/eye FK bones. Captured residual surface retained; lower neck corrections fade to zero. RDF pose blendshapes are disabled, matching the reviewed fitter.')
