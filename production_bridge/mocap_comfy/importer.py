"""Import a versioned, model-independent animation bundle into Blender."""
import io
import json
from pathlib import Path
import re
import zipfile

import bpy
from mathutils import Matrix
import numpy as np
from .keys import continuous_quaternions, fcurves, key_array
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
    for key in ('actor_camera_to_world', 'inferred_camera_to_world'):
        if key in arrays and (arrays[key].shape != (n, 4, 4) or not np.isfinite(arrays[key]).all()):
            raise ValueError('Invalid camera array: '+key)
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


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.exr', '.tif', '.tiff', '.dpx'}


def make_camera(collection, name, matrices, K, width, height, start, parent=None):
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    obj.parent = parent
    obj.rotation_mode = 'QUATERNION'
    data.sensor_fit = 'HORIZONTAL'
    data.sensor_width = 36.
    data.clip_start, data.clip_end = .01, 10000.
    rotations = np.einsum('ij,fjk,kl->fil', Y_UP_TO_Z_UP, matrices[:, :3, :3], CAMERA_LOCAL)
    quaternions = continuous_quaternions([Matrix(R.tolist()).to_quaternion() for R in rotations])
    locations = matrices[:, :3, 3]@Y_UP_TO_Z_UP.T
    obj.location, obj.rotation_quaternion = locations[0], quaternions[0]
    key_array(obj, 'location', locations, start)
    key_array(obj, 'rotation_quaternion', quaternions, start)
    fx, fy, cx, cy = K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
    lens = fx*data.sensor_width/width
    shift_x, shift_y = (width/2-cx)/width, (cy-height/2)*fx/fy/width
    constant = all(np.ptp(v) < 1e-9*max(1., abs(v[0])) for v in (lens, shift_x, shift_y))
    data.lens, data.shift_x, data.shift_y = float(lens[0]), float(shift_x[0]), float(shift_y[0])
    if not constant:
        for prop, values in (('lens', lens), ('shift_x', shift_x), ('shift_y', shift_y)):
            key_array(data, prop, values, start)
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
        if relative and i:
            for frame, value in [(start, 0.), (start+i-1, 0.), (start+i, 1.)]+([(start+i+1, 0.)] if i < len(vertices)-1 else []):
                key.value = value
                key.keyframe_insert('value', frame=frame, group='Captured surface residual')
    if relative:
        for curve in fcurves(mesh.shape_keys.animation_data.action):
            for point in curve.keyframe_points:
                point.interpolation = 'LINEAR'
    else:
        frames = [key.frame for key in mesh.shape_keys.key_blocks]
        mesh.shape_keys.eval_time = frames[0]
        key_array(mesh.shape_keys, 'eval_time', frames, start)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    material = bpy.data.materials.new('Mocap blue')
    material.diffuse_color = (.12, .48, .72, 1.)
    mesh.materials.append(material)
    return obj


PLATE_NODES = 'Mocap '


def plate_clip(video, start, source_start):
    """Movie, or numbered image sequence (local proxy frames), as a MovieClip on the job timeline."""
    clip = bpy.data.movieclips.load(str(Path(video).resolve()), check_existing=True)
    # A sequence clip plays its files in order from the loaded number; a movie's frame 1 is source frame 0.
    clip.frame_start = start if clip.source == 'SEQUENCE' else start-source_start
    return clip


def add_background(camera, clip):
    data = camera.data
    background = data.background_images.new()
    background.source = 'MOVIE_CLIP'
    background.clip = clip
    background.frame_method = 'STRETCH'
    background.display_depth = 'BACK'
    background.alpha = .65
    data.show_background_images = True


def plate_compositor(scene, clip):
    """Composite the plate under renders and Rendered view (camera backgrounds are viewport-only).

    Same layout as the ViPE add-on; an artist's own compositor setup is never replaced.
    """
    if not hasattr(scene, 'node_tree'):
        return False
    owned = lambda tree: all(n.type in ('R_LAYERS', 'COMPOSITE') or n.name.startswith(('ViPE ', PLATE_NODES)) for n in tree.nodes)
    if scene.node_tree is not None and scene.use_nodes and not owned(scene.node_tree):
        return False
    scene.use_nodes = True
    tree = scene.node_tree
    for node in [n for n in tree.nodes if n.name.startswith(('ViPE ', PLATE_NODES)) or n.type in ('R_LAYERS', 'COMPOSITE')]:
        tree.nodes.remove(node)
    def add(kind, name, x, y=0):
        node = tree.nodes.new(kind)
        node.name = node.label = PLATE_NODES+name
        node.location = (x, y)
        return node
    layers = add('CompositorNodeRLayers', 'Render', -500, -300)
    plate = add('CompositorNodeMovieClip', 'Plate', -700)
    plate.clip = clip
    scale = add('CompositorNodeScale', 'Plate to render size', -300)
    scale.space, scale.frame_method = 'RENDER_SIZE', 'STRETCH'
    tree.links.new(plate.outputs['Image'], scale.inputs['Image'])
    over = add('CompositorNodeAlphaOver', 'Render over plate', -50)
    tree.links.new(scale.outputs['Image'], over.inputs[1])
    tree.links.new(layers.outputs['Image'], over.inputs[2])
    composite = add('CompositorNodeComposite', 'Composite', 200)
    tree.links.new(over.outputs['Image'], composite.inputs['Image'])
    viewer = add('CompositorNodeViewer', 'Viewer', 200, -200)
    tree.links.new(over.outputs['Image'], viewer.inputs['Image'])
    scene.render.film_transparent = True
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    return True


def frame_timelines():
    """Fit Timeline/Dope Sheet and Graph Editors to the imported frames."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            operator = {'DOPESHEET_EDITOR': 'action', 'GRAPH_EDITOR': 'graph'}.get(area.type)
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if operator is None or region is None:
                continue
            try:
                with bpy.context.temp_override(window=window, area=area, region=region):
                    getattr(bpy.ops, operator).view_all()
            except RuntimeError:
                pass


def summary_lines(manifest, start, scene_camera):
    n = manifest['frames']
    lines = [f"{manifest['rig']['type'] if manifest.get('rig') else manifest['method']} · {n} frames · timeline {start}–{start+n-1}",
             f"Face: {manifest.get('face_capture', 'none').replace('_', ' ')} · {manifest['rig']['bones'] if manifest.get('rig') else 0} bones"]
    info = manifest.get('scene_camera')
    if info:
        error = info.get('plate_error_px', {})
        registration = info.get('registration', {})
        lines.append(f"Camera: {scene_camera or info.get('name')} (solved) · plate error {error.get('mean_px', 0):.1f} px mean, {error.get('p95_px', 0):.1f} px p95")
        lines.append(f"World fit residual {registration.get('residual_median_m', 0)*100:.1f} cm · plate lock up to {info.get('plate_lock', {}).get('max_offset_m', 0)*100:.0f} cm · rotation lock {info.get('orientation_lock', 0):.1f}")
        if info.get('gravity_tilt_degrees', 0) > 5:
            lines.append(f"GVHMR gravity is {info['gravity_tilt_degrees']:.0f}° from the scene's up axis: set the floor (ViPE scene alignment) if the scene should be ground-aligned")
    else:
        lines.append('Camera: inferred from the footage (' + manifest.get('intrinsics_source', 'estimated') + ')')
    timings = manifest.get('timings') or {}
    parts = [f'{label} {timings[key]:.0f}s' for label, key in [('prep', 'prepare_seconds'), ('GVHMR', 'gvhmr_seconds'), ('SAM', 'sam_seconds'), ('face', 'face_seconds'), ('export', 'export_seconds')] if key in timings]
    if parts:
        lines.append('Worker: '+' · '.join(parts))
    return lines


def import_bundle(path, start=1001, include_mesh=True, video='', scene=None, camera_view='WORLD', include_rig=True, plate=None):
    manifest, arrays = read_bundle(path)
    scene = scene or bpy.context.scene
    source_name = (plate or {}).get('video') or video or manifest.get('source', 'Shot')
    stem = Path(source_name).stem
    if plate and plate.get('sequence'):
        stem = re.sub(r'[._]?\d+$', '', stem)  # plate.1001.exr -> plate
    shot = re.sub(r'[._#]+$', '', stem) or 'Shot'
    collection = bpy.data.collections.new(f'Mocap | {shot[:24]} | {manifest["track_id"]}')
    scene.collection.children.link(collection)
    collection['bundle'] = str(path)
    collection['method'] = manifest['method']
    collection['track_id'] = manifest['track_id']
    collection['camera_note'] = manifest['camera_note']
    collection['face_capture'] = manifest.get('face_capture', 'none')
    collection['timings'] = json.dumps(manifest.get('timings', {}))
    collection['start_frame'] = start
    collection['frames'] = int(manifest['frames'])
    if plate and plate.get('plate'):
        collection['plate_pattern'] = str(Path(plate['video']).parent/plate['plate']['pattern'])
        collection['plate_first_frame'] = int(plate['plate']['first_frame'])
    elif video and Path(video).suffix.lower() not in IMAGE_SUFFIXES:
        collection['plate_movie'] = str(Path(video).resolve())
        collection['source_start_frame'] = int(manifest['source_start_frame'])
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
    scene.render.resolution_x = manifest.get('original_width', manifest['width'])
    scene.render.resolution_y = manifest.get('original_height', manifest['height'])
    scene.render.resolution_percentage = 100
    K = arrays['K']
    ratio = K[:, 0, 0]/K[:, 1, 1]
    scene.render.pixel_aspect_x = 1.
    scene.render.pixel_aspect_y = float(ratio[0])
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.
    width, height = manifest['width'], manifest['height']
    known = manifest.get('scene_camera')
    root = None
    existing = None
    if known:
        # The body lives in the solved camera's parent space; follow later alignment edits of that parent.
        root = bpy.data.objects.new(f'{shot[:24]} · scene placement', None)
        children['actor'].objects.link(root)
        root.empty_display_type = 'ARROWS'
        root.empty_display_size = .3
        root['mocap_role'] = 'SCENE_ROOT'
        root.parent = bpy.data.objects.get(known.get('parent') or '')
        if root.parent is not None:
            root.matrix_parent_inverse.identity()
        root.scale = (1/known.get('world_scale', 1.),)*3
        existing = bpy.data.objects.get(known.get('name', ''))
        if existing is not None and existing.type == 'CAMERA':
            collection['scene_camera'] = existing.name
            cameras = [existing]
        else:
            cameras = [make_camera(children['cameras'], 'Scene Camera (from bundle)', arrays['camera_to_world'], K, width, height, start, root)]
            cameras[0]['mocap_camera_role'] = 'SCENE'
            collection['scene_camera'] = cameras[0].name
        if 'inferred_camera_to_world' in arrays:
            inferred = make_camera(children['cameras'], 'GVHMR World Camera', arrays['inferred_camera_to_world'], K, width, height, start, root)
            inferred['mocap_camera_role'] = 'WORLD'
            inferred['description'] = "GVHMR's own inferred camera, registered into the scene; for comparison only."
            inferred.hide_set(True)
        scene.camera = cameras[0]
    else:
        cameras = [make_camera(children['cameras'], 'GVHMR World Camera', arrays['camera_to_world'], K, width, height, start)]
        cameras[0]['mocap_camera_role'] = 'WORLD'
        cameras[0]['description'] = 'Native inferred GVHMR camera. Hybrid actor placement may change the video alignment.'
        if 'actor_camera_to_world' in arrays:
            cameras.append(make_camera(children['cameras'], 'Video Fit Camera', arrays['actor_camera_to_world'], K, width, height, start))
            cameras[-1]['mocap_camera_role'] = 'VIDEO'
            cameras[-1]['description'] = 'Preserves this actor’s SAM projection after hybrid world placement. Actor-specific; not a shared scene matchmove.'
        scene.camera = cameras[-1] if camera_view == 'VIDEO' else cameras[0]
    clip = plate_clip(video, start, manifest['source_start_frame']) if video and Path(video).is_file() else None
    for camera in cameras:
        if camera is existing:
            continue
        camera.data.display_size = .15
        if camera.name in scene.objects:
            camera.hide_set(False)
        if clip is not None:
            add_background(camera, clip)
    if clip is not None and existing is None:
        collection['plate_composited'] = plate_compositor(scene, clip)
    arm = None
    if include_rig and 'rig_pose' in arrays:
        arm = create_armature(children['actor'], shot+' · '+manifest['track_id']+' · Rig', arrays, Y_UP_TO_Z_UP, start)
        arm.parent = root
    if include_mesh:
        vertices = arrays['rig_vertices'] if arm else arrays['vertices_world']
        editable_face = arm is not None and 'flame_expressions' in arrays
        mesh = mesh_cache(children['actor'], shot+' · Body', vertices @ Y_UP_TO_Z_UP.T, arrays['faces'], start, relative=editable_face)
        if arm:
            bind_mesh(mesh, arm, arrays)
        else:
            mesh.parent = root
        if editable_face:
            from .face import add_expression_keys
            add_expression_keys(mesh, arm, arrays, Y_UP_TO_Z_UP, start)
    collection['editable_face'] = bool(include_mesh and arm is not None and 'flame_expressions' in arrays)
    collection['editable_rig'] = arm is not None
    collection['rig_note'] = 'Native bones with captured surface correctives' if arm else 'Cache only; submit a new job to obtain native bones' if include_rig else 'Rig import disabled'
    collection['summary'] = json.dumps(dict(lines=summary_lines(manifest, start, collection.get('scene_camera'))+[collection['rig_note']]))
    scene.frame_set(start)
    frame_timelines()
    if arm:
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
    return collection
