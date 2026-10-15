"""Run inside Blender: validates evaluated geometry and camera reprojection."""
import json
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mocap_comfy
from mocap_comfy.importer import import_bundle, read_bundle, Y_UP_TO_Z_UP

args = sys.argv[sys.argv.index('--')+1:]
bundle, output = Path(args[0]), Path(args[1])
video = args[2] if len(args)>2 else ''
mocap_comfy.register()
assert hasattr(bpy.context.scene, 'mocap_comfy')
for obj in list(bpy.context.scene.objects):
    obj.hide_render = True
    obj.hide_set(True)
sentinel = bpy.data.objects.new('Existing_artist_object', None)
bpy.context.scene.collection.objects.link(sentinel)
sentinel.location = (19, 23, 29)
manifest, data = read_bundle(bundle)
collection = import_bundle(bundle, start=1001, include_mesh=True, video=video)
scene = bpy.context.scene
camera = scene.camera
assert not any('joint_name' in obj for obj in collection.all_objects)
assert all(not obj.hide_get() for obj in collection.all_objects if obj.type == 'CAMERA')
body = next(obj for obj in collection.all_objects if obj.type == 'MESH')
max_mesh = max_pixel = max_camera = 0.
for f in sorted(set([0, manifest['frames']//2, manifest['frames']-1])):
    scene.frame_set(1001+f)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for j, name in enumerate(manifest['joint_names']):
        got = Y_UP_TO_Z_UP @ data['joints_world'][f, j]
        p = world_to_camera_view(scene, camera, Vector(got))
        xy = np.array([p.x*manifest['width'], (1-p.y)*manifest['height']])
        max_pixel = max(max_pixel, float(np.max(np.abs(xy-data['joints_projected_pixels'][f, j]))))
    obj = body.evaluated_get(depsgraph)
    mesh = obj.to_mesh()
    vertices = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get('co', vertices.ravel())
    obj.to_mesh_clear()
    max_mesh = max(max_mesh, float(np.max(np.abs(vertices-data['vertices_world'][f]@Y_UP_TO_Z_UP.T))))
    max_camera = max(max_camera, float(np.max(np.abs(np.array(camera.matrix_world.translation)-Y_UP_TO_Z_UP@data['camera_to_world'][f, :3, 3]))))
assert max_mesh < 2e-5, max_mesh
assert max_camera < 2e-5, max_camera
assert max_pixel < .02, max_pixel
assert tuple(sentinel.location) == (19., 23., 29.)
assert abs(scene.render.fps/scene.render.fps_base-manifest['fps']) < 1e-4
assert scene.frame_end == 1001+manifest['frames']-1
if video:
    assert camera.data.background_images[0].clip.frame_start == 1001-manifest['source_start_frame']
scene.mocap_comfy.bundle = str(bundle.resolve())
scene.mocap_comfy.video = str(Path(video).resolve()) if video else ''
scene.mocap_comfy.status = 'Validated prototype result'
scene.frame_set(1001)
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
output.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(output.with_suffix('.blend')))
validation = dict(frames=manifest['frames'], method=manifest['method'], markers=0,
                  max_mesh_error_m=max_mesh,
                  max_camera_position_error_m=max_camera, max_camera_reprojection_error_px=max_pixel,
                  preserves_existing_objects=True, fps=manifest['fps'], first_frame=1001,
                  tested_frames=[0, manifest['frames']//2, manifest['frames']-1])
output.with_suffix('.json').write_text(json.dumps(validation, indent=2))
print(json.dumps(validation, indent=2))
mocap_comfy.unregister()
assert not hasattr(bpy.types.Scene, 'mocap_comfy')
