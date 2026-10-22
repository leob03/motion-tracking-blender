"""Run: blender -b --factory-startup --python blender_actor_queue.py -- PLATE OUT ENDPOINT DOWNLOAD_DIR PICK_FRAME
Crowded-shot workflow: pick on a middle frame, close-ups, two actors -> two jobs; the second reuses the prepared plate."""
import json, sys, time
from pathlib import Path
import bpy
args = sys.argv[sys.argv.index('--')+1:]
plate, out, endpoint, cache, pick = args[0], Path(args[1]), args[2], args[3], int(args[4])
out.mkdir(parents=True, exist_ok=True)
import addon_utils; addon_utils.enable('mocap_comfy', default_set=True)
import mocap_comfy as mc
prefs = bpy.context.preferences.addons['mocap_comfy'].preferences
prefs.endpoint, prefs.cache_dir = endpoint, cache
s = bpy.context.scene.mocap_comfy
s.video = plate
s.preset = 'QUICK'
s.pick_frame = pick
assert bpy.ops.mocap_comfy.detect() == {'FINISHED'}
while mc._detect is not None:
    mc.poll_detect(); time.sleep(.3)
detected = json.loads(s.detect_json)
assert 'error' not in detected, detected
crops = [im for im in bpy.data.images if im.name.startswith('Mocap actor ') and im.name != 'Mocap actor preview']
for im in crops:
    im.save(filepath=str(out/f"{im.name.replace(' ', '_')}.png"))
bpy.data.images['Mocap actor preview'].save(filepath=str(out/'preview.png'))
largest = detected['largest']
other = next(i for i in range(len(detected['boxes'])) if i != largest)
bpy.ops.mocap_comfy.pick_actor(index=largest)
bpy.ops.mocap_comfy.pick_actor(index=other, extend=True)
chosen = json.loads(s.actor_selection)
assert bpy.ops.mocap_comfy.submit() == {'FINISHED'}
keys = [j.key for j in s.jobs]
t0 = time.time()
while mc._runs and time.time()-t0 < 900:
    mc.poll_events(); time.sleep(.5)
mc.poll_events()
jobs = []
for key in keys:
    j = mc.find_job(bpy.context.scene, key)
    r = json.loads(j.record)
    log = Path(r['folder'])/'job.log'
    text = log.read_text() if log.exists() else ''
    jobs.append(dict(label=j.label, state=j.state, error=j.error[:200], prompt=j.prompt_id[:8], options={k: r['options'].get(k) for k in ('actor_index', 'actor_box', 'seed_frame')},
                     reused_prepared='Reusing the prepared plate' in text, detected_all_frames='Detecting people on every frame' in text))
report = dict(pick=pick, actors_detected=len(detected['boxes']), crops=len(crops), largest=largest, chosen=chosen, seconds=round(time.time()-t0, 1), jobs=jobs)
bpy.ops.wm.save_as_mainfile(filepath=str(out/'actors_review.blend'))
(out/'actors_report.json').write_text(json.dumps(report, indent=2))
print('ACTORS', json.dumps(report, indent=1))
assert len(jobs) == 2 and all(j['state'] == 'IMPORTED' for j in jobs)
