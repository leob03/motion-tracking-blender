"""End-to-end ViPE -> mocap hand-off in headless Blender.

blender -b --factory-startup --python vipe_handoff.py -- VIPE_RESULT.zip VIDEO OUT_DIR ENDPOINT [EXISTING_MOCAP_BUNDLE]
"""
import json, sys, time
from pathlib import Path
import bpy, numpy as np
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view
args = sys.argv[sys.argv.index('--')+1:]
vipe_zip, video, out, endpoint = args[0], args[1], Path(args[2]), args[3]
existing = args[4] if len(args) > 4 else None
out.mkdir(parents=True, exist_ok=True)
import addon_utils
for name in ('vipe_remote', 'mocap_comfy'):
    addon_utils.enable(name, default_set=True)
import mocap_comfy as mc
from vipe_remote.importer import import_bundle as vipe_import
scene = bpy.context.scene
vipe_collection, _ = vipe_import(vipe_zip, source_path=video, start_frame=1001)
cam = bpy.data.objects[vipe_collection['vipe_camera']]
s = scene.mocap_comfy
s.video = video
s.camera_mode, s.scene_camera, s.preset = 'SCENE', cam, 'BODY_HANDS'
sel = mc.selection(s)
start = mc.timeline_start(s, sel)
issues = mc.camera_check(s, sel)
assert not issues, issues
info = mc.plate(s)
report = dict(timeline_start=start, frames=sel['count'])
if existing:
    bundle = existing
else:
    camera_json = mc.scene_camera.sample(bpy.context, cam, list(range(start, start+sel['count'])), info['width'], info['height'])
    camera_file = out/'scene_camera.json'
    camera_file.write_text(json.dumps(camera_json))
    client = mc.ComfyClient(endpoint)
    client.check()
    t0 = time.time()
    uploaded = client.upload(video)
    options = mc.submit_options(s, sel)
    options['scene_camera'] = client.upload(camera_file)
    pid = client.submit(uploaded, options)
    print('submitted', pid, flush=True)
    result = client.wait(pid, report=lambda m: None)
    bundle = client.download(result, out/'result.mocap.zip')
    report['job_seconds'] = time.time()-t0
record = dict(video=video, timeline_start=start, include_mesh=True, include_rig=True, camera_view='VIDEO')
t0 = time.time()
collection = mc.import_bundle(bundle, start, True, video, scene, plate=record)
report['import_seconds'] = time.time()-t0
manifest, arrays = mc.importer.read_bundle(bundle)
report['manifest_scene_camera'] = {k: manifest['scene_camera'][k] for k in ('registration', 'gravity_tilt_degrees', 'plate_error_before_lock_px', 'plate_error_px', 'plate_lock')}
assert scene.camera == cam, 'The ViPE camera must stay the active camera'
assert collection.get('scene_camera') == cam.name
root = next(o for o in collection.all_objects if o.get('mocap_role') == 'SCENE_ROOT')
assert root.parent == cam.parent, 'Body root must share the ViPE alignment parent'
A = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
W, H = scene.render.resolution_x, scene.render.resolution_y
sx = W/manifest['width']
def projected(frame):
    scene.frame_set(frame)
    M = np.array(root.matrix_world)
    joints = arrays['joints_world'][frame-start]@A.T
    world = joints@M[:3, :3].T+M[:3, 3]
    pixels = []
    for p in world:
        c = world_to_camera_view(scene, cam, Vector(p))
        pixels.append(((c.x)*W, (1-c.y)*H))
    return np.array(pixels)
frames = [start, start+sel['count']//2, start+sel['count']-1]
blender_px = {f: projected(f) for f in frames}
worker_px = {f: arrays['joints_projected_pixels'][f-start]*sx for f in frames}
report['blender_vs_worker_projection_max_px'] = max(float(np.abs(blender_px[f]-worker_px[f]).max()) for f in frames)
# Evaluated skinned mesh against the bundle surface carried through the root.
mesh = next(o for o in collection.all_objects if o.type == 'MESH')
errors = []
for f in frames:
    scene.frame_set(f)
    ev = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get()); m = ev.to_mesh()
    co = np.empty(len(m.vertices)*3); m.vertices.foreach_get('co', co); ev.to_mesh_clear()
    co = co.reshape(-1, 3)@np.array(mesh.matrix_world)[:3, :3].T+np.array(mesh.matrix_world)[:3, 3]
    M = np.array(root.matrix_world)
    expected = (arrays['vertices_world'][f-start]@A.T)@M[:3, :3].T+M[:3, 3]
    errors.append(float(np.abs(co-expected).max()))
report['mesh_vs_bundle_max_m'] = max(errors)
# The body must follow later edits of the ViPE alignment parent (floor/scale/axis tools).
alignment = cam.parent
before = projected(frames[1])
alignment.rotation_euler.z += 0.4
alignment.scale = [v*1.3 for v in alignment.scale]
alignment.location.x += 2.
bpy.context.view_layer.update()
report['projection_change_after_alignment_edit_px'] = float(np.abs(projected(frames[1])-before).max())
alignment.rotation_euler.z -= 0.4
alignment.scale = [v/1.3 for v in alignment.scale]
alignment.location.x -= 2.
bpy.context.view_layer.update()
from mocap_comfy.nuke_export import export_nuke
nuke = export_nuke(bpy.context, collection, out/'nuke')
chan_rows = len((out/'nuke'/Path(nuke['files'][1]).name).read_text().splitlines())
report['nuke'] = dict(files=[Path(f).name for f in nuke['files']], warnings=nuke['warnings'], chan_rows=chan_rows)
assert chan_rows == sel['count']
summary = json.loads(collection['summary'])['lines']
report['summary'] = summary
scene.frame_set(start)
bpy.ops.wm.save_as_mainfile(filepath=str(out/'handoff_review.blend'))
(out/'handoff_report.json').write_text(json.dumps(report, indent=2))
print('HANDOFF', json.dumps(report, indent=2))
assert report['blender_vs_worker_projection_max_px'] < 0.05
assert report['mesh_vs_bundle_max_m'] < 1e-4
assert report['projection_change_after_alignment_edit_px'] < 0.05
