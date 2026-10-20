"""Validate face edits, neck/body isolation, keyed restore and native persistence."""
import io
import json
from pathlib import Path
import sys
import zipfile

import bpy
from mathutils import Quaternion
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mocap_comfy
from mocap_comfy.importer import import_bundle, read_bundle, Y_UP_TO_Z_UP as AXIS
from mocap_comfy.face import BONES, PREFIX

args = sys.argv[sys.argv.index('--')+1:]
bundle, reference, output = map(Path, args)
mocap_comfy.register()
manifest, data = read_bundle(bundle)
collection = import_bundle(bundle, start=1001, camera_view='VIDEO')
body = next(o for o in collection.all_objects if o.type == 'MESH')
arm = body.parent
scene = bpy.context.scene
offset = int(data['flame_vertex_offset'])
full = data['flame_neck_mask'] > .999999
with np.load(reference,allow_pickle=False) as golden:
    full &= golden['face_support']
locked = data['flame_neck_mask'] == 0
assert full.any() and locked.any()
assert len(arm.data.bones) == 131
assert sum(k.name.startswith(PREFIX) for k in body.data.shape_keys.key_blocks)==100
assert len([m for m in body.modifiers if m.type=='ARMATURE'])==1

def evaluated():
    bpy.context.view_layer.update()
    obj = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = obj.to_mesh()
    values = np.empty((len(mesh.vertices), 3), np.float32)
    mesh.vertices.foreach_get('co', values.ravel());obj.to_mesh_clear()
    return values

worst_baseline=worst_edit=worst_protected=worst_body=0.
minimum_neck_normal_cosine=1.
face_mask=np.r_[np.zeros(offset),data['flame_neck_mask']]
faces=data['faces']
transition_faces=faces[(face_mask[faces].max(1)>0)&(face_mask[faces].min(1)<1)]
for f in range(manifest['frames']):
    scene.frame_set(1001+f)
    baseline = evaluated()
    worst_baseline=max(worst_baseline,float(np.max(np.abs(baseline-data['vertices_world'][f]@AXIS.T))))
    for j, name in enumerate(BONES):
        q=np.array(arm.pose.bones[name].rotation_quaternion)
        assert abs(abs(np.dot(q, data['flame_bone_quaternions'][f,j]))-1)<1e-5
with np.load(reference, allow_pickle=False) as golden:
    for fi,f in enumerate(golden['frames']):
        scene.frame_set(1001+int(f));baseline=evaluated()
        for case in range(4):
            scene.frame_set(1001+int(f))
            if case==0:
                body.data.shape_keys.key_blocks[PREFIX+'001'].value+=.75
            else:
                pb=arm.pose.bones[BONES[case-1]]
                axis=(1,0,0) if case==1 else (0,1,0)
                pb.rotation_quaternion=pb.rotation_quaternion@Quaternion(axis, np.deg2rad(8))
            values=evaluated();delta=values-baseline
            old_tri=baseline[transition_faces];new_tri=values[transition_faces]
            old_n=np.cross(old_tri[:,1]-old_tri[:,0],old_tri[:,2]-old_tri[:,0])
            new_n=np.cross(new_tri[:,1]-new_tri[:,0],new_tri[:,2]-new_tri[:,0])
            cosine=(old_n*new_n).sum(1)/np.maximum(np.linalg.norm(old_n,axis=1)*np.linalg.norm(new_n,axis=1),1e-20)
            minimum_neck_normal_cosine=min(minimum_neck_normal_cosine,float(cosine.min()))
            expected=golden['deltas'][fi,case]@AXIS.T
            worst_edit=max(worst_edit,float(np.max(np.abs(delta[offset:][full]-expected[full]))))
            worst_protected=max(worst_protected,float(np.max(np.abs(delta[offset:][locked]))))
            worst_body=max(worst_body,float(np.max(np.abs(delta[:offset]))))
            assert np.linalg.norm(delta[offset:],axis=-1).max()>1e-4
assert worst_baseline<2e-5,worst_baseline
assert worst_edit<2e-5,worst_edit
assert worst_protected<2e-6,worst_protected
assert worst_body<2e-6,worst_body
assert minimum_neck_normal_cosine>0,minimum_neck_normal_cosine

# A manual MHR head correction must carry the new FLAME controls and face.
scene.frame_set(1001)
baseline=evaluated()
head=arm.pose.bones['c_head']
before_head=np.array(head.matrix)
head.rotation_quaternion=head.rotation_quaternion@Quaternion((0,1,0),np.deg2rad(8))
values=evaluated()
delta_head=np.array(head.matrix)@np.linalg.inv(before_head)
expected=baseline[offset:][full]@delta_head[:3,:3].T+delta_head[:3,3]
head_follow_error=float(np.max(np.abs(values[offset:][full]-expected)))
assert head_follow_error<2e-5,head_follow_error
scene.frame_set(1002);scene.frame_set(1001)

# Key a correction, leave/return to its frame, and restore only that frame.
frame=1001+manifest['frames']//2
scene.frame_set(frame)
key=body.data.shape_keys.key_blocks[PREFIX+'001'];original=key.value
key.value=original+.75
assert bpy.ops.mocap_comfy.face_key()=={'FINISHED'}
edited=evaluated()
scene.frame_set(frame+1);scene.frame_set(frame)
assert abs(key.value-original-.75)<1e-5
assert np.max(np.abs(evaluated()-edited))<2e-6
assert bpy.ops.mocap_comfy.face_restore()=={'FINISHED'}
assert abs(key.value-original)<1e-5
assert np.max(np.abs(evaluated()-data['vertices_world'][frame-1001]@AXIS.T))<2e-5

# Unknown/malformed facial extensions must fail before creating scene objects.
before=len(bpy.data.objects)
with zipfile.ZipFile(bundle) as z:members={n:z.read(n) for n in z.namelist()}
bad=data.copy();bad['flame_vertex_offset']=np.array(-1)
buffer=io.BytesIO();np.savez_compressed(buffer,**bad)
invalid=output.parent/'invalid_flame.mocap.zip'
with zipfile.ZipFile(invalid,'w') as z:
    for n,v in members.items():z.writestr(n,buffer.getvalue() if n=='animation.npz' else v)
try:
    import_bundle(invalid)
    raise AssertionError('Malformed FLAME bundle accepted')
except ValueError:pass
assert len(bpy.data.objects)==before

# Native animation survives save/reopen even with our add-on unregistered.
key.value=original+.4;bpy.ops.mocap_comfy.face_key()
edited=evaluated();name=body.name
output.parent.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(output.with_suffix('.blend').resolve()))
mocap_comfy.unregister()
bpy.ops.wm.open_mainfile(filepath=str(output.with_suffix('.blend').resolve()))
body=bpy.data.objects[name]
bpy.context.scene.frame_set(frame)
reload_error=float(np.max(np.abs(evaluated()-edited)))
assert reload_error<2e-6,reload_error
report=dict(frames=manifest['frames'],expression_keys=100,rig_bones=131,max_baseline_error_m=worst_baseline,
            max_edit_reference_error_m=worst_edit,max_locked_neck_change_m=worst_protected,
            max_body_change_m=worst_body,keyed_correction_persists=True,restore_frame_passed=True,
            body_head_edit_follow_error_m=head_follow_error,
            minimum_neck_transition_normal_cosine=minimum_neck_normal_cosine,
            malformed_face_rejected_before_scene_changes=True,native_reload_without_addon_max_m=reload_error)
output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
