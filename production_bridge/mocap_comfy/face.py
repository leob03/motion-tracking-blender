"""Native FLAME shape keys and FK corrections; no evaluation handlers/runtime."""
import json

import bpy
from bpy.types import Operator, Panel
from mathutils import Quaternion
import numpy as np

BONES = ('FLAME_jaw', 'FLAME_eye.L', 'FLAME_eye.R')
PREFIX = 'FLAME Expression '


def add_expression_keys(mesh, arm, arrays, axis, start):
    offset = int(arrays['flame_vertex_offset'])
    expr = arrays['flame_expressions']
    basis = np.empty((len(mesh.data.vertices), 3), np.float32)
    mesh.data.shape_keys.key_blocks[0].data.foreach_get('co', basis.ravel())
    # Blender limits shape-key sliders to +/-10. Rare large fitted coefficients
    # get an explicit scale; no prediction is clipped on import.
    scales = np.maximum(1., np.ceil(np.max(np.abs(expr), axis=0)/9.))
    for i, direction in enumerate(arrays['flame_expression_dirs']):
        key = mesh.shape_key_add(name=f'{PREFIX}{i+1:03}')
        key.slider_min, key.slider_max = -10., 10.
        co = basis.copy(); co[offset:] += direction@axis.T*scales[i]
        key.data.foreach_set('co', co.ravel())
        for f, value in enumerate(expr[:, i]/scales[i]):
            key.value = float(value)
            key.keyframe_insert('value', frame=start+f, group='FLAME expressions')
    for fc in mesh.data.shape_keys.animation_data.action.fcurves:
        for k in fc.keyframe_points:
            k.interpolation = 'LINEAR'
    mesh['flame_capture'] = json.dumps(dict(start=start, expressions=expr.tolist(), scales=scales.tolist(), quaternions=arrays['flame_bone_quaternions'].tolist()))
    mesh['flame_expression_count'] = expr.shape[1]
    mesh['flame_expression_page'] = 0
    mesh.id_properties_ui('flame_expression_page').update(min=0, max=(expr.shape[1]-1)//10)
    mesh['flame_armature'] = arm.name
    mesh['flame_editing_note'] = 'Expression keys and jaw/eye bones are captured on every frame. Key changes to retain them. Lower neck is protected. PCA expressions are not named FACS/ARKit controls.'
    arm['flame_mesh'] = mesh.name
    arm['flame_editing'] = True


def target(context):
    from .importer import result_collection
    collection = result_collection(context.object) or result_collection(context.scene.camera)
    mesh = next((o for o in collection.all_objects if o.type == 'MESH' and 'flame_capture' in o), None) if collection else None
    arm = mesh.parent if mesh and mesh.parent and mesh.parent.type == 'ARMATURE' else None
    return mesh, arm


def key_face(mesh, arm, frame):
    for key in mesh.data.shape_keys.key_blocks:
        if key.name.startswith(PREFIX):
            key.keyframe_insert('value', frame=frame, group='FLAME expressions')
    for name in BONES:
        arm.pose.bones[name].keyframe_insert('rotation_quaternion', frame=frame, group=name)


class MOCAP_OT_face_pose(Operator):
    bl_idname = 'mocap_comfy.face_pose'
    bl_label = 'Edit Jaw / Eye Bones'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        mesh, arm = target(context)
        if not arm:
            self.report({'ERROR'}, 'Import an editable RealDenseFace + FLAME result first')
            return {'CANCELLED'}
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in context.selected_objects:
            obj.select_set(False)
        arm.hide_set(False); arm.select_set(True)
        context.view_layer.objects.active = arm
        arm.data.collections['FLAME face'].is_visible = True
        bpy.ops.object.mode_set(mode='POSE')
        for bone in arm.data.bones:
            bone.select = bone.name == BONES[0]
        arm.data.bones.active = arm.data.bones[BONES[0]]
        return {'FINISHED'}


class MOCAP_OT_face_key(Operator):
    bl_idname = 'mocap_comfy.face_key'
    bl_label = 'Key Face Corrections'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        mesh, arm = target(context)
        if not arm:
            return {'CANCELLED'}
        key_face(mesh, arm, context.scene.frame_current)
        return {'FINISHED'}


class MOCAP_OT_face_restore(Operator):
    bl_idname = 'mocap_comfy.face_restore'
    bl_label = 'Restore Captured Face at This Frame'
    bl_description = 'Restore and key the original expressions and jaw/eye rotations at the current frame; preserve corrections on other frames'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        mesh, arm = target(context)
        if not arm:
            return {'CANCELLED'}
        capture = json.loads(mesh['flame_capture'])
        f = int(np.clip(context.scene.frame_current-capture['start'], 0, len(capture['expressions'])-1))
        for i, (value, scale) in enumerate(zip(capture['expressions'][f], capture['scales'])):
            mesh.data.shape_keys.key_blocks[f'{PREFIX}{i+1:03}'].value = value/scale
        for name, q in zip(BONES, capture['quaternions'][f]):
            arm.pose.bones[name].rotation_quaternion = Quaternion(q)
        key_face(mesh, arm, context.scene.frame_current)
        context.view_layer.update()
        return {'FINISHED'}


class MOCAP_PT_face(Panel):
    bl_label = '6. Face corrections'
    bl_idname = 'MOCAP_PT_face'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Mocap'
    bl_parent_id = 'MOCAP_PT_comfy'
    bl_order = 5
    bl_options = {'DEFAULT_CLOSED'}
    def draw(self, context):
        layout = self.layout
        mesh, arm = target(context)
        if not arm:
            layout.label(text='Requires an editable FLAME result')
            layout.label(text='Old captures need a worker re-export')
            return
        layout.operator('mocap_comfy.face_pose', icon='BONE_DATA')
        layout.label(text='Adjust, then key before changing frame')
        layout.operator('mocap_comfy.face_key', icon='KEY_HLT')
        layout.operator('mocap_comfy.face_restore', icon='LOOP_BACK')
        layout.separator()
        count = int(mesh['flame_expression_count'])
        layout.prop(mesh, '["flame_expression_page"]', text='Expression page (0–9)')
        page = min(max(int(mesh['flame_expression_page']), 0), (count-1)//10)
        layout.label(text='FLAME PCA components; signed values')
        scales = json.loads(mesh['flame_capture'])['scales']
        for i in range(page*10, min(page*10+10, count)):
            key = mesh.data.shape_keys.key_blocks[f'{PREFIX}{i+1:03}']
            label = f'Expression {i+1:03}' + (f' (×{scales[i]:g})' if scales[i] != 1 else '')
            layout.prop(key, 'value', text=label)


CLASSES = (MOCAP_OT_face_pose, MOCAP_OT_face_key, MOCAP_OT_face_restore, MOCAP_PT_face)
