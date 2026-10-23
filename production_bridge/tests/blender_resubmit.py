"""Run: blender -b --factory-startup --python blender_resubmit.py -- MOVIE OUT [ENDPOINT]
Submit, wait, change actor, submit again in one session (the artist's sequence). Second job must reach the worker."""
import json, sys, time, bpy
from pathlib import Path
args = sys.argv[sys.argv.index('--')+1:]
video, out = args[0], Path(args[1])
import addon_utils; addon_utils.enable('mocap_comfy', default_set=True)
import mocap_comfy as mc
p = bpy.context.preferences.addons['mocap_comfy'].preferences; p.cache_dir = str(out); p.endpoint = args[2] if len(args) > 2 else p.endpoint
s = bpy.context.scene.mocap_comfy
s.video = video; s.preset = 'QUICK'; s.first_frame, s.last_frame = 1, 8
def run():
    assert bpy.ops.mocap_comfy.submit() == {'FINISHED'}
    key = s.jobs[0].key
    t = time.time()
    while mc._runs and time.time()-t < 300:
        mc.poll_events(); time.sleep(.5)
    mc.poll_events()
    j = mc.find_job(bpy.context.scene, key)
    return dict(key=key, prompt=j.prompt_id, state=j.state, error=j.error[:100], folder=json.loads(j.record).get('folder'), actor=json.loads(j.record)['options']['actor_index'])
first = run()
s.actor_index = 1
second = run()
print('RESUBMIT', json.dumps(dict(first=first, second=second, jobs=[(j.key, j.state, j.prompt_id[:8]) for j in s.jobs])))
assert second['prompt'] and second['prompt'] != first['prompt'] and second['key'] != first['key'], 'second job did not reach the worker'
