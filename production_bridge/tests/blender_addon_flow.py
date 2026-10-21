"""Drive the real add-on operators headless: plate probe, Detect Actors, Submit, monitoring and auto-import.

blender -b --factory-startup --python addon_flow.py -- PLATE OUT_DIR ENDPOINT PRESET [COLORSPACE]
"""
import json, sys, time
from pathlib import Path
import bpy
args = sys.argv[sys.argv.index('--')+1:]
plate, out, endpoint, preset = args[0], Path(args[1]), args[2], args[3]
out.mkdir(parents=True, exist_ok=True)
import addon_utils
addon_utils.enable('mocap_comfy', default_set=True)
import mocap_comfy as mc
prefs = bpy.context.preferences.addons['mocap_comfy'].preferences
prefs.endpoint, prefs.cache_dir = endpoint, str(out/'downloads')
scene = bpy.context.scene
s = scene.mocap_comfy
s.video = plate
if len(args) > 4:
    s.plate_colorspace = args[4]
s.preset = preset
info = mc.plate(s)
report = dict(plate_info=info, range=mc.range_text(s)[0])
assert bpy.ops.mocap_comfy.check() == {'FINISHED'}
report['connection'] = dict(mc._connection)
t0 = time.time()
assert bpy.ops.mocap_comfy.detect() == {'FINISHED'}
while mc._detect is not None:
    mc.poll_detect(); time.sleep(.3)
detected = json.loads(s.detect_json)
assert 'error' not in detected, detected
report['detect'] = dict(seconds=round(time.time()-t0, 1), actors=len(detected['boxes']), largest=detected['largest'], chosen=s.actor_index)
assert bpy.data.images.get('Mocap actor preview') is not None
bpy.data.images['Mocap actor preview'].save(filepath=str(out/'actor_preview.png'))
t0 = time.time()
assert bpy.ops.mocap_comfy.submit() == {'FINISHED'}
job = s.jobs[0]
states = []
while mc._runs:
    mc.poll_events()
    if not states or states[-1] != (job.state, mc._runs.get(job.key, {}).get('stage')):
        states.append((job.state, mc._runs.get(job.key, {}).get('stage')))
    time.sleep(.5)
mc.poll_events()
report['job'] = dict(seconds=round(time.time()-t0, 1), state=job.state, error=job.error, detail=job.detail, states=states,
                     options=json.loads(job.record)['options'])
assert job.state == 'IMPORTED', (job.state, job.error)
collection = mc.active_result(bpy.context)
report['summary'] = json.loads(collection['summary'])['lines']
report['timeline'] = [scene.frame_start, scene.frame_end]
camera = scene.camera
backgrounds = [(b.source, b.image.filepath if b.image else b.clip.filepath if b.clip else None,
                b.image_user.frame_start if b.image else None, b.image_user.frame_offset if b.image else None) for b in camera.data.background_images]
report['background'] = backgrounds
bpy.ops.wm.save_as_mainfile(filepath=str(out/'flow_review.blend'))
(out/'flow_report.json').write_text(json.dumps(report, indent=2))
print('FLOW', json.dumps(report, indent=2))
