"""Blender-native armature and skinning; no model runtime on the artist machine."""
import bpy
from mathutils import Matrix, Vector
import numpy as np


def create_armature(collection, name, arrays, axis, start):
    names = arrays['rig_names'].tolist()
    parents = arrays['rig_parents']
    convert = np.eye(4)
    convert[:3, :3] = axis
    rest = arrays['rig_rest']
    positions = rest[:, :3, 3]@axis.T
    data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, data)
    collection.objects.link(arm)
    arm.show_in_front = True
    arm['mocap_role'] = 'RIG'
    arm['editing'] = 'Pose bones, insert LocRotScale keys; captured surface correctives remain tied to their original frames.'
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    for j, bone_name in enumerate(names):
        bone = data.edit_bones.new(bone_name)
        bone.head = positions[j]
        children = [k for k in range(len(names)) if parents[k] == j and np.linalg.norm(positions[k]-positions[j]) > .005]
        children.sort(key=lambda k: ('proc' in names[k] or 'null' in names[k], k))
        direction = positions[children[0]]-positions[j] if children else np.array([0., 0., .05])
        if bone_name.startswith('FLAME_'):
            direction = axis@rest[j, :3, 1]
        length = .025 if bone_name.startswith('FLAME_') else np.clip(np.linalg.norm(direction), .018, .35)
        bone.tail = positions[j]+direction/np.linalg.norm(direction)*length
        bone.align_roll(Vector((axis@rest[j, :3, 2]).tolist()))
        if parents[j] >= 0:
            bone.parent = data.edit_bones[names[int(parents[j])]]
        bone.use_connect = False
    bpy.ops.object.mode_set(mode='OBJECT')
    groups = {name: data.collections.new(name) for name in ('Body', 'Hands', 'Face', 'Helpers', 'FLAME face')}
    groups['Helpers'].is_visible = False
    groups['Face'].is_visible = False
    for bone in data.bones:
        lower = bone.name.lower()
        group = ('Helpers' if 'proc' in lower or 'null' in lower else
                 'Hands' if any(part in lower for part in ('thumb', 'index', 'middle', 'ring', 'pinky', 'palm')) else
                 'Face' if any(part in lower for part in ('jaw', 'teeth', 'tongue', 'eye')) else 'Body')
        if bone.name.startswith('FLAME_'):
            group = 'Helpers' if bone.name == 'FLAME_root' else 'FLAME face'
        groups[group].assign(bone)
        bone['native_joint_index'] = names.index(bone.name)
    blender_rest = np.stack([np.array(data.bones[n].matrix_local) for n in names])
    rest_inv = np.linalg.inv(blender_rest)
    change_basis = np.linalg.inv(rest)@convert.T@blender_rest
    last = {}
    for frame, pose in enumerate(arrays['rig_pose']):
        posed = convert@pose@change_basis
        for j, bone_name in enumerate(names):
            parent = int(parents[j])
            basis = rest_inv[j]@posed[j] if parent < 0 else rest_inv[j]@blender_rest[parent]@np.linalg.inv(posed[parent])@posed[j]
            pb = arm.pose.bones[bone_name]
            location, rotation, scale = Matrix(basis.tolist()).decompose()
            if bone_name in last and rotation.dot(last[bone_name]) < 0:
                rotation.negate()
            last[bone_name] = rotation.copy()
            pb.rotation_mode = 'QUATERNION'
            if bone_name.startswith('FLAME_'):
                pb.lock_location = (True, True, True)
                pb.lock_scale = (True, True, True)
            pb.location, pb.rotation_quaternion, pb.scale = location, rotation, scale
            for prop in ('location', 'rotation_quaternion', 'scale'):
                pb.keyframe_insert(prop, frame=start+frame, group=bone_name)
    action = arm.animation_data.action
    action.name = name+' · Captured motion'
    for curve in action.fcurves:
        for key in curve.keyframe_points:
            key.interpolation = 'LINEAR'
    return arm


def bind_mesh(mesh, arm, arrays):
    groups = [mesh.vertex_groups.new(name=name) for name in arrays['rig_names'].tolist()]
    indices, weights = arrays['rig_indices'], arrays['rig_weights']
    for vertex, (ids, values) in enumerate(zip(indices, weights)):
        for joint, weight in zip(ids, values):
            if weight > 0:
                groups[int(joint)].add([vertex], float(weight), 'REPLACE')
    modifier = mesh.modifiers.new('Captured skeleton', 'ARMATURE')
    modifier.object = arm
    modifier.use_deform_preserve_volume = False
    modifier.use_vertex_groups = True
    modifier.use_bone_envelopes = False
    mesh.parent = arm
    mesh['mocap_role'] = 'SKINNED_BODY'
