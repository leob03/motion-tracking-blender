"""Import a versioned, model-independent animation bundle into Blender."""
import io
import json
from pathlib import Path
import zipfile

import bpy
from mathutils import Matrix
import numpy as np
from .rig import create_armature, bind_mesh

Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
CAMERA_LOCAL = np.diag([1., -1., -1.])


def read_bundle(path):
    with zipfile.ZipFile(path) as bundle:
        names = [entry.filename for entry in bundle.infolist()]
        if len(names) != len(set(names)) or sum(entry.file_size for entry in bundle.infolist()) > 2*1024**3:
            raise ValueError('Invalid or oversized mocap bundle')
        for entry in bundle.infolist():
            if entry.filename not in {'manifest.json', 'animation.npz', 'actor_preview.jpg', 'tracking.json', 'face_parameters.npz'}:
                raise ValueError('Unexpected bundle member: '+entry.filename)
            if entry.file_size > 2*1024**3:
                raise ValueError('Bundle member exceeds 2 GiB')
        manifest = json.loads(bundle.read('manifest.json'))
        if manifest.get('schema') != 'mocap-blender-bundle' or manifest.get('version') not in (1, 2):
            raise ValueError('Unsupported mocap bundle schema')
        if manifest['version'] == 2 and not manifest.get('face_editing'):
            raise ValueError('Version 2 requires editable FLAME metadata')
        animation = io.BytesIO(bundle.read('animation.npz'))
        with zipfile.ZipFile(animation) as contents:
            if sum(entry.file_size for entry in contents.infolist()) > 2*1024**3:
                raise ValueError('Animation expands beyond 2 GiB')
        animation.seek(0)
        with np.load(animation, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    n, joints = manifest['frames'], len(manifest['joint_names'])
    for name, shape in [('camera_to_world', (n, 4, 4)), ('K', (n, 3, 3)), ('joints_world', (n, joints, 3))]:
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError('Invalid animation array: '+name)
    if n < 1 or not np.isfinite(manifest['fps']) or manifest['fps'] <= 0 or manifest['width'] <= 0 or manifest['height'] <= 0:
        raise ValueError('Invalid frame rate or dimensions')
    vertices, faces = arrays['vertices_world'], arrays['faces']
    if vertices.ndim != 3 or vertices.shape[0] != n or vertices.shape[2] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid mesh vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu' or np.any(faces < 0) or np.any(faces >= vertices.shape[1]):
        raise ValueError('Invalid mesh triangles')
    K = arrays['K']
    if np.any(K[:, 0, 0] <= 0) or np.any(K[:, 1, 1] <= 0) or np.any(np.abs(K[:, 0, 1]) > 1e-6):
        raise ValueError('Unsupported camera intrinsics')
    ratio = K[:, 0, 0]/K[:, 1, 1]
    if np.max(np.abs(ratio-ratio[0])) > 1e-6:
        raise ValueError('Changing pixel aspect ratio is unsupported')
    if 'actor_camera_to_world' in arrays and (arrays['actor_camera_to_world'].shape != (n, 4, 4) or not np.isfinite(arrays['actor_camera_to_world']).all()):
        raise ValueError('Invalid actor-fit camera')
    rig_keys = {'rig_names', 'rig_parents', 'rig_rest', 'rig_pose', 'rig_indices', 'rig_weights', 'rig_vertices'}
    if rig_keys.intersection(arrays):
        if not rig_keys.issubset(arrays):
            raise ValueError('Incomplete skeletal animation')
        names, parents = arrays['rig_names'], arrays['rig_parents']
        j = len(names)
        if names.ndim != 1 or names.dtype.kind not in 'US' or len(set(names.tolist())) != j or not 1 <= j <= 512:
            raise ValueError('Invalid bone names')
        if parents.shape != (j,) or parents.dtype.kind not in 'iu' or np.any(parents < -1) or np.any(parents >= np.arange(j)):
            raise ValueError('Invalid bone hierarchy')
        for key, shape in [('rig_rest', (j, 4, 4)), ('rig_pose', (n, j, 4, 4)), ('rig_vertices', vertices.shape)]:
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError('Invalid rig array: '+key)
        for key in ('rig_rest', 'rig_pose'):
            if np.max(np.abs(arrays[key][..., 3, :]-[0, 0, 0, 1])) > 1e-5 or np.min(np.abs(np.linalg.det(arrays[key][..., :3, :3]))) < 1e-8:
                raise ValueError('Invalid bone transforms')
        ids, weights = arrays['rig_indices'], arrays['rig_weights']
        if ids.ndim != 2 or ids.shape[0] != vertices.shape[1] or ids.shape != weights.shape or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= j):
            raise ValueError('Invalid skin indices')
        if not np.isfinite(weights).all() or np.any(weights < 0) or np.max(np.abs(weights.sum(1)-1)) > 1e-5:
            raise ValueError('Invalid skin weights')
    flame_keys = {'flame_expression_dirs', 'flame_expressions', 'flame_vertex_offset', 'flame_neck_mask', 'flame_bone_quaternions'}
    if flame_keys.intersection(arrays) or manifest.get('face_editing'):
        if manifest['version'] != 2:
            raise ValueError('Editable FLAME requires bundle version 2')
        if not flame_keys.issubset(arrays) or not rig_keys.issubset(arrays):
            raise ValueError('Incomplete editable FLAME data')
        if (manifest.get('face_editing') or {}).get('version') != 1:
            raise ValueError('Unsupported FLAME editing version')
        dirs = arrays['flame_expression_dirs']
        if dirs.ndim != 3 or dirs.shape[2] != 3 or not 1 <= dirs.shape[0] <= 100 or not 1 <= dirs.shape[1] <= 10000:
            raise ValueError('Invalid FLAME expression basis')
        e, v = dirs.shape[:2]
        offset = arrays['flame_vertex_offset']
        if offset.shape != () or offset.dtype.kind not in 'iu' or int(offset) < 0 or int(offset)+v != vertices.shape[1]:
            raise ValueError('Invalid FLAME vertex range')
        for key, shape in [('flame_expression_dirs', (e, v, 3)), ('flame_expressions', (n, e)), ('flame_neck_mask', (v,)), ('flame_bone_quaternions', (n, 3, 4))]:
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError('Invalid FLAME array: '+key)
        if np.any(arrays['flame_neck_mask'] < 0) or np.any(arrays['flame_neck_mask'] > 1):
            raise ValueError('Invalid FLAME neck mask')
        if np.max(np.abs(np.linalg.norm(arrays['flame_bone_quaternions'], axis=-1)-1)) > 1e-4:
            raise ValueError('Invalid FLAME captured rotations')
        if arrays['rig_names'].tolist()[-4:] != ['FLAME_root', 'FLAME_jaw', 'FLAME_eye.L', 'FLAME_eye.R']:
            raise ValueError('Missing FLAME bones')
    return manifest, arrays


def result_collection(obj):
    if obj is None:
        return None
    for collection in obj.users_collection:
        if 'mocap_root' in collection:
            return bpy.data.collections.get(collection['mocap_root'])
        if 'bundle' in collection:
            return collection
    return None


def linear(block):
    if block.animation_data and block.animation_data.action:
        for curve in block.animation_data.action.fcurves:
            for key in curve.keyframe_points:
                key.interpolation = 'LINEAR'


def make_camera(collection, name, matrices, K, width, height, start):
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.rotation_mode = 'QUATERNION'
    data.sensor_fit = 'HORIZONTAL'
    data.sensor_width = 36.
    data.clip_start, data.clip_end = .01, 10000.
    previous = None
    for i, matrix in enumerate(matrices):
        R = Y_UP_TO_Z_UP @ matrix[:3, :3] @ CAMERA_LOCAL
        obj.location = Y_UP_TO_Z_UP @ matrix[:3, 3]
        q = Matrix(R.tolist()).to_quaternion()
        if previous is not None and q.dot(previous) < 0:
            q.negate()
        previous = q.copy()
        obj.rotation_quaternion = q
        obj.keyframe_insert('location', frame=start+i)
        obj.keyframe_insert('rotation_quaternion', frame=start+i)
        fx, fy, cx, cy = K[i, 0, 0], K[i, 1, 1], K[i, 0, 2], K[i, 1, 2]
        data.lens = fx*data.sensor_width/width
        data.shift_x = (width/2-cx)/width
        data.shift_y = (cy-height/2)*fx/fy/width
        for prop in ('lens', 'shift_x', 'shift_y'):
            data.keyframe_insert(prop, frame=start+i)
    linear(obj)
    linear(data)
    return obj


def mesh_cache(collection, name, vertices, faces, start, relative=False):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices[0].tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.shape_key_add(name='Frame_0')
    mesh.shape_keys.use_relative = relative
    for i in range(1, len(vertices)):
        key = obj.shape_key_add(name=f'Frame_{i}')
        key.data.foreach_set('co', vertices[i].astype(np.float32).ravel())
    for i, key in enumerate(mesh.shape_keys.key_blocks):
        key.interpolation = 'KEY_LINEAR'
        if relative:
            if i:
                for frame, value in [(start, 0.), (start+i-1, 0.), (start+i, 1.)]+([(start+i+1, 0.)] if i < len(vertices)-1 else []):
                    key.value = value
                    key.keyframe_insert('value', frame=frame, group='Captured surface residual')
        else:
            mesh.shape_keys.eval_time = key.frame
            mesh.shape_keys.keyframe_insert('eval_time', frame=start+i)
    linear(mesh.shape_keys)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    material = bpy.data.materials.new('Mocap blue')
    material.diffuse_color = (.12, .48, .72, 1.)
    mesh.materials.append(material)
    return obj


def import_bundle(path, start=1001, include_mesh=True, video='', scene=None, camera_view='WORLD', include_rig=True):
    manifest, arrays = read_bundle(path)
    scene = scene or bpy.context.scene
    shot = Path(video).stem if video else Path(manifest.get('source', 'Shot')).stem
    collection = bpy.data.collections.new(f'Mocap | {shot[:24]} | {manifest["track_id"]}')
    scene.collection.children.link(collection)
    collection['bundle'] = str(path)
    collection['method'] = manifest['method']
    collection['track_id'] = manifest['track_id']
    collection['camera_note'] = manifest['camera_note']
    collection['face_capture'] = manifest.get('face_capture', 'none')
    collection['timings'] = json.dumps(manifest.get('timings', {}))
    children = {}
    for key, label in [('actor', '01 Character'), ('cameras', '02 Cameras')]:
        child = bpy.data.collections.new(f'{label} | {shot[:24]}')
        collection.children.link(child)
        child['mocap_root'] = collection.name
        children[key] = child
    n = manifest['frames']
    fps = manifest['fps']
    scene.render.fps = round(fps)
    scene.render.fps_base = round(fps)/fps
    scene.frame_start, scene.frame_end = start, start+n-1
    scene.render.resolution_x, scene.render.resolution_y = manifest['width'], manifest['height']
    scene.render.resolution_percentage = 100
    K = arrays['K']
    ratio = K[:, 0, 0]/K[:, 1, 1]
    scene.render.pixel_aspect_x = 1.
    scene.render.pixel_aspect_y = float(ratio[0])
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.
    cameras = [make_camera(children['cameras'], 'GVHMR World Camera', arrays['camera_to_world'], K, manifest['width'], manifest['height'], start)]
    cameras[0]['mocap_camera_role'] = 'WORLD'
    cameras[0]['description'] = 'Native inferred GVHMR camera. Hybrid actor placement may change the video alignment.'
    if 'actor_camera_to_world' in arrays:
        cameras.append(make_camera(children['cameras'], 'Video Fit Camera', arrays['actor_camera_to_world'], K, manifest['width'], manifest['height'], start))
        cameras[-1]['mocap_camera_role'] = 'VIDEO'
        cameras[-1]['description'] = 'Preserves this actor’s SAM projection after hybrid world placement. Actor-specific; not a shared scene matchmove.'
    scene.camera = cameras[-1] if camera_view == 'VIDEO' else cameras[0]
    for camera in cameras:
        camera.data.display_size = .15
        camera.hide_set(False)
    if video and Path(video).is_file():
        clip = bpy.data.movieclips.load(str(Path(video).resolve()), check_existing=False)
        clip.frame_start = start-manifest['source_start_frame']
        for camera in cameras:
            bg = camera.data.background_images.new()
            bg.source = 'MOVIE_CLIP'
            bg.clip = clip
            bg.frame_method = 'STRETCH'
            bg.alpha = .65
            camera.data.show_background_images = True
    arm = None
    if include_rig and 'rig_pose' in arrays:
        arm = create_armature(children['actor'], shot+' · '+manifest['track_id']+' · Rig', arrays, Y_UP_TO_Z_UP, start)
    if include_mesh:
        vertices = arrays['rig_vertices'] if arm else arrays['vertices_world']
        editable_face = arm is not None and 'flame_expressions' in arrays
        mesh = mesh_cache(children['actor'], shot+' · Body', vertices @ Y_UP_TO_Z_UP.T, arrays['faces'], start, relative=editable_face)
        if arm:
            bind_mesh(mesh, arm, arrays)
        if editable_face:
            from .face import add_expression_keys
            add_expression_keys(mesh, arm, arrays, Y_UP_TO_Z_UP, start)
    collection['editable_face'] = bool(include_mesh and arm is not None and 'flame_expressions' in arrays)
    collection['editable_rig'] = arm is not None
    collection['rig_note'] = 'Native bones with captured surface correctives' if arm else 'Cache only; submit a new job to obtain native bones' if include_rig else 'Rig import disabled'
    scene.frame_set(start)
    if arm:
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
    return collection
