"""Write whole per-frame animation curves at once (keyframe_insert per frame is slow in Python)."""
import bpy
import numpy as np

LINEAR = 1  # Keyframe.interpolation enum value


def fcurves(action):
    if bpy.app.version < (4, 4, 0):
        return list(action.fcurves)
    return [curve for layer in action.layers for strip in layer.strips for bag in strip.channelbags for curve in bag.fcurves]


def key_array(owner, prop, values, start, group=None):
    """Key owner.<prop> on frames start, start+1, ... with values (frames,) or (frames, components), linear.

    owner is any struct keyframe_insert accepts (object, pose bone, camera data, shape key).
    """
    values = np.asarray(values, dtype=np.float64)
    values = values.reshape(len(values), -1)
    count = len(values)
    if group:
        owner.keyframe_insert(prop, frame=start, group=group)
    else:
        owner.keyframe_insert(prop, frame=start)
    id_block = owner.id_data
    path = owner.path_from_id(prop) if owner != id_block else prop
    curves = {(c.data_path, c.array_index): c for c in fcurves(id_block.animation_data.action) if c.data_path == path}
    co = np.empty(2*count)
    co[0::2] = start+np.arange(count)
    for index in range(values.shape[1]):
        curve = curves[(path, index)]
        points = curve.keyframe_points
        if len(points) != 1:
            raise RuntimeError(f'{path} already has keys; bulk keying needs a fresh curve')
        points.add(count-1)
        co[1::2] = values[:, index]
        points.foreach_set('co', co)
        points.foreach_set('interpolation', np.full(count, LINEAR, dtype=np.int32))
        curve.update()


def continuous_quaternions(q):
    """Flip signs so consecutive quaternions (frames, ..., 4) never take the long way round."""
    q = np.array(q, dtype=np.float64)
    for frame in range(1, len(q)):
        flip = (q[frame]*q[frame-1]).sum(-1) < 0
        q[frame][flip] *= -1
    return q
