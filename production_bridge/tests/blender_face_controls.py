"""Check the viewport solver against Blender evaluation on an editable capture."""
import json
import io
import zipfile
from pathlib import Path
import sys
import time

import bpy
from mathutils import Quaternion
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mocap_comfy
from mocap_comfy.face import PREFIX, BONES, key_face
from mocap_comfy.face_controls import ExpressionDrag, BoneDrag, LANDMARKS, HANDLES, evaluated_points, handle_positions

args=sys.argv[sys.argv.index('--')+1:]
scene_path,report_path=map(Path,args)
mocap_comfy.register()
if scene_path.suffix=='.zip':
    from mocap_comfy.importer import import_bundle
    import_bundle(scene_path,start=1001)
else:
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
mesh=next(o for o in bpy.context.scene.objects if o.type=='MESH' and 'flame_capture' in o)
arm=mesh.parent
scene=bpy.context.scene
start=scene.frame_start
indices=np.arange(len(mesh.data.vertices))
bundle_path=scene_path if scene_path.suffix=='.zip' else Path(scene.mocap_comfy.bundle)
with zipfile.ZipFile(bundle_path) as z, np.load(io.BytesIO(z.read('animation.npz'))) as arrays:
    locked=arrays['flame_neck_mask']==0

def values():
    bpy.context.view_layer.update()
    return evaluated_points(mesh,indices)

# Each control starts from the captured frame, so the comparisons are independent.
rows=[]
max_jac_error=0.
for frame in (start,start+3):
    scene.frame_set(frame)
    if frame!=start:
        # Existing body/head edits and a nontrivial armature object transform must
        # be included in the solver; use transforms that leave the face on-screen.
        arm.rotation_euler[2]=.17
        arm.scale=(1.1,1.1,1.1)
        head=arm.pose.bones['c_head']
        head.rotation_quaternion=head.rotation_quaternion @ Quaternion((0,1,0),.12)
    bpy.context.view_layer.update()
    for i,(name,_,kind) in enumerate(HANDLES):
        if kind!='EXPR':continue
        before=values()
        t=time.perf_counter();drag=ExpressionDrag(mesh,arm,i);elapsed=time.perf_counter()-t
        up=np.array((arm.matrix_world @ arm.pose.bones['FLAME_root'].matrix).to_3x3().col[1].normalized())
        request=up*.003
        drag.apply(request);after=values()
        coefficient_delta=(np.array([k.value for k in drag.keys])-drag.original)*drag.scales
        predicted=np.einsum('vie,e->vi',drag.jacobian,coefficient_delta)
        actual=(after-before)[drag.indices]
        error=float(np.max(np.abs(predicted-actual)));max_jac_error=max(max_jac_error,error)
        movement=actual[drag.selected].mean(0)
        gain=float(movement@request/(request@request))
        body_error=float(np.max(np.abs((after-before)[:-5023])))
        target_error=float(np.linalg.norm(movement-request))
        collar_error=float(np.max(np.abs((after-before)[-5023:][locked])))
        far=np.linalg.norm(before[drag.indices]-drag.origin,axis=1)>.045
        far_rms=float(np.sqrt(np.mean(actual[far]**2)))
        assert collar_error<2e-6,(name,'Protected collar moved',collar_error)
        assert error<3e-6,(name,'Jacobian',error)
        assert body_error<2e-6,(name,'Body moved',body_error)
        assert gain>.45,(name,'Handle did not follow',gain)
        rows.append(dict(frame=frame,handle=name,gain=gain,target_error_m=target_error,setup_seconds=elapsed,body_error_m=body_error,collar_error_m=collar_error,far_region_rms_m=far_rms))
        drag.restore();restored=values();assert np.max(np.abs(before-restored))<2e-6

# True pose controls and cancel, for jaw and each eye.
scene.frame_set(start);arm.rotation_euler=(0,0,0);arm.scale=(1,1,1)
baseline=values()
for kind in ('JAW',*BONES[1:]):
    drag=BoneDrag(arm,kind)
    handle_index=next(i for i,h in enumerate(HANDLES) if h[2]==kind)
    handle_before=handle_positions(mesh,arm)[handle_index].copy()
    drag.apply((.015,0,.008),vertical_pixels=-30.)
    changed=values()
    assert (handle_positions(mesh,arm)[handle_index]-handle_before).length>1e-4
    assert np.max(np.abs(changed-baseline))>1e-4,kind
    assert np.max(np.abs((changed-baseline)[:-5023]))<2e-6
    assert drag.original.rotation_difference(drag.bone.rotation_quaternion).angle<=.3501
    drag.restore();assert np.max(np.abs(values()-baseline))<2e-6

# A drag can be baked to native shape-key animation, then restored at one frame.
drag=ExpressionDrag(mesh,arm,0);drag.apply((0,0,.004));key_face(mesh,arm,start)
edited=values()
scene.frame_set(start+1);scene.frame_set(start)
assert np.max(np.abs(values()-edited))<2e-6
assert bpy.ops.mocap_comfy.face_restore()=={'FINISHED'}
assert np.max(np.abs(values()-baseline))<2e-6

# Edited expressions remain standard Blender data with the add-on disabled.
drag=ExpressionDrag(mesh,arm,0);drag.apply((0,0,.002));key_face(mesh,arm,start)
edited=values();name=mesh.name
report_path.parent.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(report_path.with_suffix('.blend').resolve()))
mocap_comfy.unregister()
bpy.ops.wm.open_mainfile(filepath=str(report_path.with_suffix('.blend').resolve()))
mesh=bpy.data.objects[name]
reload_error=float(np.max(np.abs(values()-edited)))
assert reload_error<2e-6,reload_error

report=dict(native_reload_without_addon_max_m=reload_error,max_jacobian_error_m=max_jac_error,controls=rows,bone_controls_cancel=True,keyed_edit_and_restore=True)
report_path.parent.mkdir(parents=True,exist_ok=True)
report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(max_jacobian_error_m=max_jac_error,min_handle_gain=min(x['gain'] for x in rows),max_target_error_m=max(x['target_error_m'] for x in rows),max_setup_seconds=max(x['setup_seconds'] for x in rows)),indent=2))
