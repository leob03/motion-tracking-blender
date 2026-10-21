"""Sample a solved Blender camera (ViPE, 3DE, ...) for a mocap job; runs on Blender's main thread.

Matrices are expressed in the camera parent's space, converted to metres with the parent's scale,
so the imported character can be parented to the same object and follows later alignment edits.
"""
import json

import bpy
from mathutils import Matrix, Vector

CAMERA_AXES = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


def vipe_job(camera):
    """The ViPE Remote job collection that owns this camera, if any."""
    if camera is None:
        return None
    return next((c for c in bpy.data.collections if c.get('vipe_camera') == camera.name), None)


def vipe_timing(camera):
    """(timeline frame of ViPE source frame 0, first source frame, frame count, distortion) or None."""
    job = vipe_job(camera)
    if job is None:
        return None
    text = bpy.data.texts.get(job.get('vipe_manifest_text', ''))
    try:
        manifest = json.loads(text.as_string()) if text else {}
        selection = manifest['selection']
        first, count = int(selection.get('first_source_frame', 0)), int(selection['count'])
    except (ValueError, KeyError, TypeError):
        return None
    start = int(job.get('vipe_start_frame', 1))
    return dict(zero_frame=start-first, first_source=first, count=count, start=start, distortion=bool(job.get('vipe_distortion')),
                width=manifest.get('source', {}).get('width'), height=manifest.get('source', {}).get('height'))


def keyed_range(camera):
    ranges = [block.animation_data.action.frame_range for block in (camera, camera.parent) if block is not None and block.animation_data and block.animation_data.action]
    if not ranges:
        return None
    return int(min(r[0] for r in ranges)), int(max(r[1] for r in ranges))


def intrinsics(data, width, height):
    """[fx, fy, cx, cy] in plate pixels (OpenCV, square pixels) from Blender lens, filmback and shift."""
    fit = data.sensor_fit if data.sensor_fit != 'AUTO' else ('HORIZONTAL' if width >= height else 'VERTICAL')
    fit_px = width if fit == 'HORIZONTAL' else height
    sensor = data.sensor_height if data.sensor_fit == 'VERTICAL' else data.sensor_width
    f = data.lens/sensor*fit_px
    return [f, f, width/2-data.shift_x*fit_px, height/2+data.shift_y*fit_px]


def sample(context, camera, timeline_frames, width, height):
    """Camera JSON (version 1) for the worker: OpenCV camera-to-parent in metres, Z-up, and plate intrinsics."""
    if camera is None or camera.type != 'CAMERA':
        raise ValueError('Choose a camera object for the scene camera mode')
    if camera.data.type != 'PERSP':
        raise ValueError('The scene camera must be a perspective camera')
    scene = context.scene
    parent = camera.parent
    rows, lenses, scales = [], [], []
    old = scene.frame_current
    try:
        for frame in timeline_frames:
            scene.frame_set(frame)
            depsgraph = context.evaluated_depsgraph_get()
            world = camera.evaluated_get(depsgraph).matrix_world.copy()
            space = parent.evaluated_get(depsgraph).matrix_world.copy() if parent else Matrix.Identity(4)
            scale = sum(space.to_scale())/3
            local = space.inverted()@world
            location, rotation, _ = local.decompose()
            c2w = Matrix.LocRotScale(location*scale, rotation, None)@CAMERA_AXES
            rows.append([v for row in c2w for v in row])
            lenses.append(intrinsics(camera.evaluated_get(depsgraph).data, width, height))
            scales.append(scale)
        up = (space.inverted().to_3x3()@Vector((0, 0, 1))).normalized()
    finally:
        scene.frame_set(old)
    if max(scales)-min(scales) > 1e-6*max(scales):
        raise ValueError('The camera parent changes scale during the shot')
    timing = vipe_timing(camera)
    return dict(version=1, name=camera.name, parent=parent.name if parent else '', world_scale=scales[0],
                width=width, height=height, frames=len(rows), timeline_start=timeline_frames[0],
                scene_up_local=list(up), c2w=rows, K=lenses, source='vipe' if timing else 'scene',
                distortion=bool(timing and timing['distortion']))
