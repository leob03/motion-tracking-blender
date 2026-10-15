"""Native skeleton/weights plus per-frame pre-skin surfaces for exact fitted meshes."""
import json
from pathlib import Path
import struct

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree


def compact_weights(weights):
    count = max(1, int((weights > 1e-8).sum(1).max()))
    ids = np.argsort(-weights, axis=1)[:, :count]
    values = np.take_along_axis(weights, ids, axis=1)
    values /= values.sum(1, keepdims=True)
    return ids.astype(np.int32), values.astype(np.float32)


def read_sam_rig(path, frames, world_from_camera):
    with Path(path).open('rb') as stream:
        magic, version, _ = struct.unpack('<4sII', stream.read(12))
        assert magic == b'glTF' and version == 2
        size, _ = struct.unpack('<II', stream.read(8))
        gltf = json.loads(stream.read(size))
        size, _ = struct.unpack('<II', stream.read(8))
        binary = stream.read(size)

    def accessor(index):
        a = gltf['accessors'][index]
        view = gltf['bufferViews'][a['bufferView']]
        components = {'SCALAR': 1, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
        dtype = np.dtype({5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}[a['componentType']])
        stride = view.get('byteStride', components*dtype.itemsize)
        return np.ndarray((a['count'], components), dtype, buffer=binary,
                          offset=view.get('byteOffset', 0)+a.get('byteOffset', 0),
                          strides=(stride, dtype.itemsize)).copy()

    nodes, skin = gltf['nodes'], gltf['skins'][0]
    node_ids = skin['joints']
    parents = {child: i for i, node in enumerate(nodes) for child in node.get('children', [])}
    channels = {}
    for animation in gltf.get('animations', []):
        for channel in animation['channels']:
            target = channel['target']
            if target['path'] == 'weights':
                continue
            sampler = animation['samplers'][channel['sampler']]
            values = accessor(sampler['output'])
            if len(values) != frames:
                raise ValueError('Skeletal GLB frame count differs from mesh cache')
            channels[(target['node'], target['path'])] = values
    globals_ = {}

    def global_matrix(index):
        if index in globals_:
            return globals_[index]
        node = nodes[index]
        if 'matrix' in node:
            local = np.tile(np.array(node['matrix']).reshape(4, 4).T, (frames, 1, 1))
        else:
            values = {}
            for key, default in [('translation', [0, 0, 0]), ('rotation', [0, 0, 0, 1]), ('scale', [1, 1, 1])]:
                values[key] = channels.get((index, key), np.tile(node.get(key, default), (frames, 1)))
            local = np.tile(np.eye(4), (frames, 1, 1))
            local[:, :3, :3] = Rotation.from_quat(values['rotation']).as_matrix()*values['scale'][:, None, :]
            local[:, :3, 3] = values['translation']
        globals_[index] = global_matrix(parents[index])@local if index in parents else local
        return globals_[index]

    poses = np.stack([global_matrix(i) for i in node_ids], axis=1)
    poses = (world_from_camera@np.diag([1., -1., -1., 1.]))[:, None]@poses
    rest = np.linalg.inv(accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1))
    hierarchy = np.array([node_ids.index(parents[i]) if parents.get(i) in node_ids else -1 for i in node_ids], np.int32)
    attributes = gltf['meshes'][0]['primitives'][0]['attributes']
    ids = accessor(attributes['JOINTS_0']).astype(np.int32)
    weights = accessor(attributes['WEIGHTS_0'])
    if 'JOINTS_1' in attributes:
        ids = np.concatenate([ids, accessor(attributes['JOINTS_1']).astype(np.int32)], axis=1)
        weights = np.concatenate([weights, accessor(attributes['WEIGHTS_1'])], axis=1)
    weights /= weights.sum(1, keepdims=True)
    names_file = Path(__file__).with_name('mhr_bone_names.json')
    names = json.loads(names_file.read_text()) if names_file.exists() else [nodes[i]['name'] for i in node_ids]
    assert len(names) == len(node_ids)
    return dict(rig_names=np.array(names), rig_parents=hierarchy, rig_rest=rest,
                rig_pose=poses, rig_indices=ids, rig_weights=weights)


def attach_flame_weights(rig, body_camera, composite_camera):
    """Transfer nearby MHR head/neck skin weights to added FLAME vertices."""
    count = len(body_camera)
    added = composite_camera[count:]
    names = rig['rig_names'].tolist()
    head_joints = {i for i, name in enumerate(names) if name.startswith(('c_head', 'c_neck'))}
    for joint, parent in enumerate(rig['rig_parents']):
        if int(parent) in head_joints:
            head_joints.add(joint)
    support_weights = (np.isin(rig['rig_indices'], list(head_joints))*rig['rig_weights']).sum(1)
    support = np.flatnonzero(support_weights > .25)
    if len(support) < 4:
        raise ValueError('MHR head/neck skin support is missing')
    distances, neighbors = cKDTree(body_camera[support]).query(added, k=4)
    ids = support[neighbors]
    mix = 1/np.maximum(distances, .002)**2
    mix /= mix.sum(1, keepdims=True)
    dense = np.zeros((len(added), len(rig['rig_names'])), np.float64)
    rows = np.arange(len(added))[:, None]
    for neighbor in range(4):
        np.add.at(dense, (rows, rig['rig_indices'][ids[:, neighbor]]),
                  rig['rig_weights'][ids[:, neighbor]]*mix[:, neighbor, None])
    # Keep original native influences; appended head may need a wider sparse row.
    added_ids, added_weights = compact_weights(dense)
    width = max(added_ids.shape[1], rig['rig_indices'].shape[1])
    def pad(x):
        return np.pad(x, ((0, 0), (0, width-x.shape[1])))
    rig['rig_indices'] = np.concatenate([pad(rig['rig_indices']), pad(added_ids)])
    rig['rig_weights'] = np.concatenate([pad(rig['rig_weights']), pad(added_weights)])


def preserve_fitted_surface(rig, vertices, report=None):
    """Undo each frame's blended skin transform, retaining learned correctives/face.

    Blender reapplies the animated skeleton. At unedited poses this reconstructs
    the fitted surface; bone edits deform it through normal armature skinning.
    """
    inverse_rest = np.linalg.inv(rig['rig_rest'])
    ids, weights = rig['rig_indices'], rig['rig_weights']
    output = np.empty_like(vertices, dtype=np.float32)
    worst = 0.
    for frame, world in enumerate(vertices):
        matrices = rig['rig_pose'][frame]@inverse_rest
        blended = (matrices[ids]*weights[..., None, None]).sum(1)
        output[frame] = np.linalg.solve(blended[:, :3, :3], (world-blended[:, :3, 3])[..., None])[..., 0]
        reconstructed = np.einsum('vij,vj->vi', blended[:, :3, :3], output[frame])+blended[:, :3, 3]
        worst = max(worst, float(np.max(np.abs(reconstructed-world))))
        if report:
            report(frame+1, len(vertices))
    if not np.isfinite(output).all() or worst > 2e-5:
        raise ValueError(f'Could not preserve fitted surface under skeletal skinning ({worst:g} m)')
    rig['rig_vertices'] = output
    return worst
