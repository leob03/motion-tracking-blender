"""Run: blender -b --factory-startup --python blender_job_list.py
Two jobs in one session must keep their own key, record and folder (regression: stale reference after move)."""
import json, bpy
import addon_utils; addon_utils.enable('mocap_comfy', default_set=True)
import mocap_comfy as mc
scene = bpy.context.scene
key_a = mc.new_job(scene, 'first', dict(n=1)); mc.find_job(scene, key_a).prompt_id = 'aaaa'
key_b = mc.new_job(scene, 'second', dict(n=2)); b = mc.find_job(scene, key_b)
b.record = json.dumps(dict(n=2, touched=True))
assert scene.mocap_comfy.jobs[scene.mocap_comfy.jobs_index].key == key_b
first, second = mc.find_job(scene, key_a), mc.find_job(scene, key_b)
print('TWO_JOBS', key_a != key_b, b.label, b.prompt_id == '', first.label, first.prompt_id, json.loads(first.record), json.loads(second.record))
assert b.label == 'second' and b.prompt_id == '' and json.loads(first.record) == dict(n=1) and json.loads(second.record)['touched']
