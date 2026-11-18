"""Level from actor: make the ground under a mocap actor horizontal (Z = 0) in a Camera Track scene.

The actor's planted feet (foot_contacts from SAM 3D Body on camera) lie on the ground, so their positions
over the shot describe the floor; Camera Track's points near the feet refine it. The fitted ground is
turned to Z = 0 by moving the Camera Track alignment object, like Camera Track's own Set Floor: camera,
points, dense environment, splats and the mocap body (parented to the same alignment) move together and the
plate match is unchanged.

Choice of ground, in order:
  feet + points  Camera Track points form a plane under the feet that agrees with the feet (<= 10 degrees):
                 its normal and height (many points, wide extent), feet decide which plane.
  feet           The planted feet span an area (not one spot or one line): plane through them, lowered by the
                 ankle height.
  points         Feet span too little, but there is a point plane right under them.
  posture        Only the actor's standing posture (hips to shoulders while planted) gives up; height from feet.
"""
import json
import zipfile
import io

import bpy
import numpy as np
from bpy.props import BoolProperty
from mathutils import Matrix, Vector

from .importer import Y_UP_TO_Z_UP, result_collection
from . import scene_camera


def active_result(context):
    """Same lookup as the Result panel: the selected object's result, else the scene camera's, else the latest."""
    for obj in (context.object, context.scene.camera):
        collection = result_collection(obj)
        if collection is not None:
            return collection
    return next((c for c in reversed(bpy.data.collections) if 'bundle' in c and c.get('method')), None)

ANKLE_HEIGHT_M = .08


def bundle_arrays(collection):
    with zipfile.ZipFile(collection['bundle']) as bundle:
        manifest = json.loads(bundle.read('manifest.json'))
        with np.load(io.BytesIO(bundle.read('animation.npz')), allow_pickle=False) as data:
            keys = [k for k in ('joints_world', 'foot_contacts') if k in data.files]
            return manifest, {k: data[k] for k in keys}


def scene_root(collection):
    return next((o for o in collection.all_objects if o.get('mocap_role') == 'SCENE_ROOT'), None)


def plane(points):
    centre = points.mean(0)
    _, spread, axes = np.linalg.svd(points-centre)
    return centre, axes[2], spread/np.sqrt(len(points))


def ransac_plane(points, threshold, rounds=500, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(rounds):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b-a, c-a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        inliers = np.abs((points-a)@n) < threshold
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < 12:
        return None
    centre, normal, _ = plane(points[best])
    return centre, normal, int(best.sum())


def camera_track_points(camera):
    job = scene_camera.vipe_job(camera)
    if job is None:
        return np.zeros((0, 3))
    points = []
    for obj in job.all_objects:
        if obj.get('vipe_role') in ('native_cloud', 'dense_surface') and obj.type == 'MESH' and len(obj.data.vertices):
            local = np.empty(len(obj.data.vertices)*3)
            obj.data.vertices.foreach_get('co', local)
            local = local.reshape(-1, 3)
            if len(local) > 200000:
                local = local[np.random.default_rng(0).choice(len(local), 200000, replace=False)]
            M = np.array(obj.matrix_world)
            points.append(local@M[:3, :3].T+M[:3, 3])
    return np.concatenate(points) if points else np.zeros((0, 3))


def level(collection, camera, move_origin=True):
    """(world transform for the alignment object, report dict)."""
    manifest, arrays = bundle_arrays(collection)
    if 'foot_contacts' not in arrays:
        raise ValueError('This result has no foot contacts; Level from Actor needs a "SAM 3D Body on camera" result')
    root = scene_root(collection)
    if root is None:
        raise ValueError('This result is not placed on a scene camera')
    M = np.array(root.matrix_world)
    to_world = lambda p: (p@Y_UP_TO_Z_UP.T)@M[:3, :3].T+M[:3, 3]
    joints = to_world(arrays['joints_world'])  # (F, 12, 3) Blender world
    contacts = arrays['foot_contacts'].astype(bool)
    feet = np.concatenate([joints[contacts[:, 0], 10], joints[contacts[:, 1], 11]])
    if len(feet) < 3:
        raise ValueError('The actor has fewer than 3 planted-foot frames (feet out of frame?); use Camera Track Set Floor instead')
    unit = float(np.linalg.norm(M[:3, 0]))  # metres -> world units of this scene
    camera_position = np.array(camera.matrix_world.translation)
    feet_centre, feet_normal, spread = plane(feet)
    if (camera_position-feet_centre)@feet_normal < 0:
        feet_normal = -feet_normal
    standing = contacts.any(1)
    up_posture = ((joints[standing][:, [0, 1]].mean(1))-(joints[standing][:, [6, 7]].mean(1))).mean(0)
    up_posture /= np.linalg.norm(up_posture)
    area = spread[1] > .03*unit  # feet span an area, not one spot or one line
    rms_feet = float(np.sqrt((((feet-feet_centre)@feet_normal)**2).mean()))/unit
    cloud = camera_track_points(camera)
    near = cloud[np.linalg.norm(cloud-feet_centre, axis=1) < max(3*spread[0], 1.5*unit)] if len(cloud) else cloud
    guess = feet_normal if area else up_posture
    found = None
    if len(near) >= 30:
        offset = (near-feet_centre)@guess
        candidates = near[(offset > -.6*unit) & (offset < .1*unit)]
        if len(candidates) >= 30:
            found = ransac_plane(candidates, .02*unit)
    method, normal, ground = None, None, None
    if found is not None:
        c_centre, c_normal, inliers = found
        if c_normal@guess < 0:
            c_normal = -c_normal
        angle = float(np.degrees(np.arccos(np.clip(c_normal@guess, -1, 1))))
        if angle <= (10 if area else 20):
            method, normal = ('feet + points' if area else 'points'), c_normal
            ground = feet_centre-((feet_centre-c_centre)@c_normal)*c_normal
    if method is None and area:
        method, normal = 'feet', feet_normal
        ground = feet_centre-ANKLE_HEIGHT_M*unit*feet_normal
    if method is None:
        method, normal = 'posture', up_posture
        ground = feet_centre-ANKLE_HEIGHT_M*unit*up_posture
    rotation = Vector(normal.tolist()).rotation_difference(Vector((0, 0, 1))).to_matrix().to_4x4()
    if move_origin:
        first = joints[np.flatnonzero(standing)[0]][[10, 11]].mean(0)
        origin = first-((first-ground)@normal)*normal  # first planted position, on the ground
        transform = rotation@Matrix.Translation(Vector((-origin).tolist()))
    else:
        # Turn about the ground point under the feet, then drop it to Z = 0; the horizontal position stays.
        transform = (Matrix.Translation(Vector((ground[0], ground[1], 0.)))@rotation@Matrix.Translation(Vector((-ground).tolist())))
    tilt = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
    residual = float(np.sqrt((((feet-ground)@normal-ANKLE_HEIGHT_M*unit)**2).mean()))/unit
    report = dict(method=method, tilt_corrected_degrees=tilt, planted_samples=int(len(feet)),
                  feet_spread_m=[float(spread[0]/unit), float(spread[1]/unit)], feet_plane_rms_m=rms_feet,
                  ankle_height_residual_rms_m=residual, camera_track_points_used=int(found[2]) if found is not None and method != 'feet' else 0)
    return transform, report


class MOCAP_OT_level_from_actor(bpy.types.Operator):
    bl_idname = 'mocap_comfy.level_from_actor'
    bl_label = 'Level from Actor'
    bl_description = ("Make the ground under this actor horizontal (Z = 0) from their planted feet and nearby Camera Track points, "
                      "by moving the Camera Track alignment (camera, points, environment and body move together)")
    bl_options = {'REGISTER', 'UNDO'}
    move_origin: BoolProperty(name='Origin under the actor', default=True,
                              description="Put the world origin where the actor first stands; off keeps the current horizontal position")

    @classmethod
    def poll(cls, context):
        collection = active_result(context)
        if collection is None or collection.get('method') != 'sam_camera':
            cls.poll_message_set('Select a "SAM 3D Body on camera" result')
            return False
        camera = bpy.data.objects.get(collection.get('scene_camera', ''))
        if camera is None or scene_camera.vipe_job(camera) is None or camera.parent is None:
            cls.poll_message_set('Needs a result placed on a Camera Track camera')
            return False
        return True

    def execute(self, context):
        collection = active_result(context)
        camera = bpy.data.objects[collection['scene_camera']]
        alignment = camera.parent
        try:
            transform, report = level(collection, camera, self.move_origin)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        if 'vipe_original_matrix' not in alignment:  # Camera Track's alignment reset keeps working
            alignment['vipe_original_matrix'] = [value for row in alignment.matrix_world for value in row]
        alignment.matrix_world = transform@alignment.matrix_world
        context.view_layer.update()
        collection['level_from_actor'] = json.dumps(report)
        self.report({'INFO'}, f"Levelled from the actor ({report['method']}): tilt {report['tilt_corrected_degrees']:.1f}°, "
                              f"feet within {report['ankle_height_residual_rms_m']*100:.1f} cm of the ground")
        return {'FINISHED'}


CLASSES = (MOCAP_OT_level_from_actor,)
