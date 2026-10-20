"""Viewport FLAME handles. Edits bake to native keys/bones; no playback handler."""
import math

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
from bpy.props import StringProperty
from bpy.types import Gizmo, GizmoGroup, Operator
from bpy_extras.view3d_utils import location_3d_to_region_2d, region_2d_to_location_3d
from mathutils import Quaternion, Vector

from .face import BONES, PREFIX, key_face, target

# Fixed FLAME2023 topology: nearest vertices to standard face landmarks 17..67.
# Semantic indices only, not model geometry or expression weights. Their locations
# follow the fitted identity and all existing deformation in the artist's scene.
LANDMARKS = (1983,3157,335,3705,3684,3851,3863,16,2138,570,3553,3561,
             3501,3526,2748,2792,3556,1675,1612,2437,2383,2494,3632,2293,
             2296,3833,1343,1034,1175,884,881,2715,2813,2774,3543,1657,
             1696,1579,1795,1865,3503,2948,2898,2845,2785,3533,1668,
             1730,1848,3509,2937)
# name, semantic landmarks, control kind; L/R are the actor's left/right.
HANDLES = (
    ('R mouth corner', (48,60), 'EXPR'), ('L mouth corner', (54,64), 'EXPR'),
    ('Upper lip', (50,51,52,61,62,63), 'EXPR'),
    ('Lower lip', (56,57,58,65,66,67), 'EXPR'),
    ('R brow', (18,19,20), 'EXPR'), ('L brow', (23,24,25), 'EXPR'),
    ('R upper lid', (37,38), 'EXPR'), ('R lower lid', (40,41), 'EXPR'),
    ('L upper lid', (43,44), 'EXPR'), ('L lower lid', (46,47), 'EXPR'),
    ('Jaw open / close', (), 'JAW'), ('L eye aim', (), BONES[1]),
    ('R eye aim', (), BONES[2]),
)
COLORS = {'EXPR': (.1,.85,.65), 'JAW': (1.,.55,.12),
          BONES[1]: (.65,.45,1.), BONES[2]: (.65,.45,1.)}
_label_handle = None
_active_drag = None
_focus_views = {}


def offset_for(mesh):
    offset = int(mesh.get('flame_vertex_offset', len(mesh.data.vertices)-5023))
    if len(mesh.data.vertices)-offset != 5023 or offset < 0:
        raise ValueError('Visual controls require the FLAME2023 5023-vertex head')
    return offset


def evaluated_points(mesh, indices):
    obj = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    if len(obj.data.vertices) != len(mesh.data.vertices):
        raise ValueError('Disable modifiers that change topology before editing the face')
    points = np.array([obj.data.vertices[int(i)].co[:] for i in indices])
    matrix = np.array(obj.matrix_world)
    return points@matrix[:3,:3].T+matrix[:3,3]


def frame_axes(arm):
    m = arm.matrix_world @ arm.pose.bones['FLAME_root'].matrix
    return m.to_3x3().normalized(), m.translation


def head_scale(arm):
    a, b = (arm.matrix_world @ arm.pose.bones[n].head for n in BONES[1:])
    return max((a-b).length/.062, 1e-4)


def handle_positions(mesh, arm):
    points = evaluated_points(mesh, np.array(LANDMARKS)+offset_for(mesh))
    axes, origin = frame_axes(arm)
    scale = head_scale(arm)
    result = []
    for _, landmarks, kind in HANDLES:
        if kind == 'EXPR':
            point = points[np.array(landmarks)-17].mean(0)
        elif kind == 'JAW':
            point = points[57-17]-np.array(axes.col[1])*.04*scale
        else:
            # Side-mounted aim widgets leave the eyelids and pupils unobscured.
            pivot = arm.matrix_world @ arm.pose.bones[kind].head
            side = 1 if kind.endswith('.L') else -1
            gaze = (arm.matrix_world @ arm.pose.bones[kind].matrix).to_3x3().col[2].normalized()
            point = (np.array(pivot)+np.array(axes.col[0])*.065*scale*side
                     + np.array(gaze-axes.col[2])*.12*scale)
        result.append(Vector(point))
    return result


class ExpressionDrag:
    """Regularized FLAME fit at the current pose, in world units.

    Solve once for three displacement directions. Interactive updates then need
    only a 100x3 matrix multiply. Other landmarks and dense surface samples act
    as soft pins. The identity, jaw/eyes, body and collar are never solved here.
    """
    def __init__(self, mesh, arm, handle):
        self.mesh = mesh
        self.keys = [k for k in mesh.data.shape_keys.key_blocks if k.name.startswith(PREFIX)]
        if not self.keys or any(k.mute or k.vertex_group or k.relative_key != mesh.data.shape_keys.key_blocks[0] for k in self.keys):
            raise ValueError('Expression keys must be unmuted and relative to Basis, without vertex masks')
        modifiers = [m for m in mesh.modifiers if m.show_viewport]
        if len(modifiers) != 1 or modifiers[0].type != 'ARMATURE' or modifiers[0].object != arm or modifiers[0].use_deform_preserve_volume:
            raise ValueError('Visual editing needs the original linear armature modifier; disable extra modifiers')
        import json
        self.scales = np.asarray(json.loads(mesh['flame_capture'])['scales'])
        self.original = np.array([k.value for k in self.keys])
        self.limit = .015*head_scale(arm)
        offset = offset_for(mesh)
        selected = np.array(HANDLES[handle][1])-17
        indices = offset+np.r_[LANDMARKS, np.linspace(0,5022,200,dtype=int)]
        count = len(mesh.data.vertices)
        buffer = np.empty((count,3),np.float32)
        mesh.data.shape_keys.key_blocks[0].data.foreach_get('co',buffer.ravel())
        basis = buffer[indices].astype(np.float64)
        dirs=[]
        for key,scale in zip(self.keys,self.scales):
            key.data.foreach_get('co',buffer.ravel())
            dirs.append((buffer[indices]-basis)/scale)
        dirs=np.array(dirs)
        # Exact derivative of the exported linear blend skinning. Work in armature
        # space then map to world, including user object transforms and head edits.
        transform = np.array(arm.matrix_world.inverted() @ mesh.matrix_world)
        skin = {g.index: np.array(arm.pose.bones[g.name].matrix @ arm.data.bones[g.name].matrix_local.inverted())[:3,:3]
                for g in mesh.vertex_groups if g.name in arm.pose.bones}
        jac=[]
        for vertex in indices:
            influences=[(g.weight,skin[g.group]) for g in mesh.data.vertices[int(vertex)].groups if g.group in skin]
            weight=sum(w for w,_ in influences)
            mat=sum((w*m for w,m in influences),np.zeros((3,3)))/weight if weight>0 else np.eye(3)
            jac.append(np.array(arm.matrix_world)[:3,:3] @ mat @ transform[:3,:3])
        jac=np.einsum('vij,evj->vie',np.array(jac),dirs)
        positions=evaluated_points(mesh,indices)
        self.origin=positions[selected].mean(0)
        distance=np.linalg.norm(positions-self.origin,axis=1)
        weights=np.clip(distance/(.025*head_scale(arm)),.05,1.)
        weights[len(LANDMARKS):]*=.3
        weights[selected]=12.
        A=(jac*weights[:,None,None]).reshape(-1,len(self.keys))
        B=np.zeros((len(indices),3,3));B[selected]=np.eye(3)*weights[selected,None,None]
        # Penalize coefficient changes in true FLAME units, including keys whose
        # displayed values were rescaled to fit Blender's slider range.
        damping=.0006*head_scale(arm)
        self.mapping=np.linalg.solve(A.T@A+np.eye(len(self.keys))*damping**2,A.T@B.reshape(-1,3))
        self.jacobian=jac
        self.indices=indices
        self.selected=selected
        self.last_delta=np.zeros(3)

    def apply(self, delta):
        delta=np.asarray(delta,dtype=float)
        length=np.linalg.norm(delta)
        if length>self.limit:delta=delta*self.limit/length
        dc=self.mapping@delta
        dc=np.clip(dc,-3.,3.)
        values=self.original+dc/self.scales
        for key,value in zip(self.keys,values):
            key.value=float(np.clip(value,key.slider_min,key.slider_max))
        self.last_delta=delta

    def restore(self):
        for key,value in zip(self.keys,self.original):key.value=float(value)


class BoneDrag:
    def __init__(self, arm, kind):
        self.kind=kind
        self.bone=arm.pose.bones[BONES[0] if kind=='JAW' else kind]
        self.original=self.bone.rotation_quaternion.copy()
        self.world=arm.matrix_world @ self.bone.matrix
        self.origin=self.world.translation.copy()
        self.aim=self.origin+self.world.to_3x3().col[2].normalized()*.12*head_scale(arm)

    def apply(self, delta, vertical_pixels=0.):
        if self.kind=='JAW':
            self.bone.rotation_quaternion=self.original @ Quaternion((1,0,0),max(-.35,min(.35,-vertical_pixels*.004)))
        else:
            direction=(self.aim+Vector(delta)-self.origin).normalized()
            local=self.world.to_3x3().inverted() @ direction
            q=Vector((0,0,1)).rotation_difference(local.normalized())
            if q.angle>.35:q=Quaternion(q.axis,.35)
            self.bone.rotation_quaternion=self.original @ q

    def restore(self):self.bone.rotation_quaternion=self.original


class MOCAP_OT_face_drag(Operator):
    bl_idname='mocap_comfy.face_drag'
    bl_label='Drag Face Control'
    bl_options={'UNDO','BLOCKING'}
    handle: StringProperty()

    @classmethod
    def description(cls,context,properties):
        try:name,_,kind=HANDLES[int(properties.handle)]
        except (ValueError,IndexError):return 'Drag a facial control'
        return name+(' — drag up/down to close/open the jaw' if kind=='JAW' else ' — drag in the view plane')+'; Shift: precision; Esc: cancel'

    def invoke(self,context,event):
        global _active_drag
        mesh,arm=target(context)
        if not mesh or context.region.type!='WINDOW':return {'CANCELLED'}
        try:
            self.index=int(self.handle)
            _,_,kind=HANDLES[self.index]
            self.drag=ExpressionDrag(mesh,arm,self.index) if kind=='EXPR' else BoneDrag(arm,kind)
            self.origin=handle_positions(mesh,arm)[self.index]
        except (ValueError,KeyError,IndexError,np.linalg.LinAlgError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        self.mesh,self.arm=mesh,arm
        self.frame=context.scene.frame_current
        self.initial=Vector((event.mouse_region_x,event.mouse_region_y))
        self.area=context.area
        self.changed=False
        self.closed=False
        self.key_on_release=context.scene.mocap_comfy.face_auto_key
        _active_drag=self
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(HANDLES[self.index][0]+' | Drag to adjust · Shift: precision · Esc: cancel')
        return {'RUNNING_MODAL'}

    def finish(self,context,cancel=False):
        global _active_drag
        if self.closed:return {'CANCELLED'}
        self.closed=True
        if cancel and context.scene.frame_current==self.frame:self.drag.restore()
        if not cancel and self.changed and self.key_on_release:key_face(self.mesh,self.arm,self.frame)
        if self.area:
            self.area.header_text_set(None);self.area.tag_redraw()
        _active_drag=None
        return {'CANCELLED'} if cancel or not self.changed else {'FINISHED'}

    def modal(self,context,event):
        if self.closed:return {'CANCELLED'}
        if context.scene.frame_current!=self.frame or not context.scene.mocap_comfy.face_visual:
            return self.finish(context,cancel=True)
        if event.type in {'ESC','RIGHTMOUSE'}:return self.finish(context,cancel=True)
        if event.type=='LEFTMOUSE' and event.value=='RELEASE':return self.finish(context)
        if event.type=='MOUSEMOVE':
            mouse=Vector((event.mouse_region_x,event.mouse_region_y))
            factor=.2 if event.shift else 1.
            start=region_2d_to_location_3d(context.region,context.region_data,self.initial,self.origin)
            end=region_2d_to_location_3d(context.region,context.region_data,mouse,self.origin)
            delta=(end-start)*factor
            if isinstance(self.drag,ExpressionDrag):self.drag.apply(delta)
            else:self.drag.apply(delta,(mouse.y-self.initial.y)*factor)
            self.changed=(mouse-self.initial).length>0.01
            context.view_layer.update();context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def cancel(self,context):self.finish(context,cancel=True)


class MOCAP_GT_face_handle(Gizmo):
    bl_idname='MOCAP_GT_face_handle'
    __slots__=('shape',)

    def setup(self):
        # Filled screen-sized disk, including its center, for easy selection.
        points=[]
        for i in range(24):
            a,b=i*math.tau/24,(i+1)*math.tau/24
            points.extend(((0,0,0),(math.cos(a),math.sin(a),0),(math.cos(b),math.sin(b),0)))
        self.shape=self.new_custom_shape('TRIS',points)

    def draw(self,context):self.draw_custom_shape(self.shape)
    def draw_select(self,context,select_id):self.draw_custom_shape(self.shape,select_id=select_id)


class MOCAP_GGT_face_controls(GizmoGroup):
    bl_idname='MOCAP_GGT_face_controls'
    bl_label='FLAME visual corrections'
    bl_space_type='VIEW_3D'
    bl_region_type='WINDOW'
    bl_options={'3D','PERSISTENT','SHOW_MODAL_ALL'}

    @classmethod
    def poll(cls,context):
        settings=getattr(context.scene,'mocap_comfy',None)
        return bool(settings and settings.face_visual and context.mode in {'OBJECT','POSE'} and target(context)[1])

    def setup(self,context):
        self.handles=[]
        for i,(_,_,kind) in enumerate(HANDLES):
            gizmo=self.gizmos.new(MOCAP_GT_face_handle.bl_idname)
            gizmo.color=COLORS[kind];gizmo.alpha=.85
            gizmo.color_highlight=(1.,1.,1.);gizmo.alpha_highlight=1.
            gizmo.scale_basis=.075
            gizmo.use_draw_modal=True
            gizmo.target_set_operator(MOCAP_OT_face_drag.bl_idname).handle=str(i)
            self.handles.append(gizmo)

    def draw_prepare(self,context):
        mesh,arm=target(context)
        try:points=handle_positions(mesh,arm)
        except (ValueError,ReferenceError):
            for gizmo in self.handles:gizmo.hide=True
            return
        for gizmo,point in zip(self.handles,points):
            gizmo.hide=False
            matrix=context.region_data.view_rotation.to_matrix().to_4x4()
            matrix.translation=point
            gizmo.matrix_basis=matrix


class MOCAP_OT_face_focus(Operator):
    bl_idname='mocap_comfy.face_focus'
    bl_label='Face Close-up'
    bl_description='Frame the face from the front and show draggable controls; leave the shot camera unchanged'

    def execute(self,context):
        mesh,arm=target(context)
        if not arm or context.area.type!='VIEW_3D':return {'CANCELLED'}
        axes,_=frame_axes(arm)
        points=evaluated_points(mesh,np.array(LANDMARKS)+offset_for(mesh))
        view=context.space_data.region_3d
        overlay=context.space_data.overlay
        _focus_views.setdefault(context.area.as_pointer(),dict(rotation=view.view_rotation.copy(), location=view.view_location.copy(), distance=view.view_distance, perspective=view.view_perspective, extras=overlay.show_extras, bones=overlay.show_bones))
        overlay.show_extras=False
        overlay.show_bones=False
        view.view_rotation=axes.to_quaternion()
        view.view_location=Vector(points.mean(0))
        view.view_distance=.32*head_scale(arm)
        view.view_perspective='ORTHO'
        # Blender's sidebar overlays the window region. Center the face in the
        # unobscured area, so right-side controls remain reachable with N open.
        window=next(r for r in context.area.regions if r.type=='WINDOW')
        sidebar=next((r.width for r in context.area.regions if r.type=='UI'),0) if context.space_data.show_region_ui else 0
        if sidebar:
            view.update()
            center=(window.width/2,window.height/2)
            shifted=(center[0]+sidebar/2,center[1])
            a=region_2d_to_location_3d(window,view,center,view.view_location)
            b=region_2d_to_location_3d(window,view,shifted,view.view_location)
            view.view_location+=b-a
        context.scene.mocap_comfy.face_visual=True
        context.space_data.show_gizmo=True
        context.space_data.show_gizmo_context=True
        context.area.tag_redraw()
        return {'FINISHED'}


class MOCAP_OT_face_previous_view(Operator):
    bl_idname='mocap_comfy.face_previous_view'
    bl_label='Previous View'
    bl_description='Restore the view and guide visibility from before Face Close-up'

    @classmethod
    def poll(cls,context):
        return bool(context.area and context.area.as_pointer() in _focus_views)

    def execute(self,context):
        previous=_focus_views.pop(context.area.as_pointer())
        view=context.space_data.region_3d
        view.view_rotation=previous['rotation'];view.view_location=previous['location']
        view.view_distance=previous['distance'];view.view_perspective=previous['perspective']
        context.space_data.overlay.show_extras=previous['extras']
        context.space_data.overlay.show_bones=previous['bones']
        context.area.tag_redraw()
        return {'FINISHED'}


def draw_labels():
    context=bpy.context
    settings=getattr(context.scene,'mocap_comfy',None)
    if not (settings and settings.face_visual and settings.face_labels and context.area and context.area.type=='VIEW_3D' and context.space_data.show_gizmo and context.space_data.show_gizmo_context):return
    mesh,arm=target(context)
    if not arm or context.mode not in {'OBJECT','POSE'}:return
    try:points=handle_positions(mesh,arm)
    except (ValueError,ReferenceError):return
    projected=[location_3d_to_region_2d(context.region,context.region_data,p) for p in points]
    if any(p is None for p in projected):return
    blf.size(0,12)
    left=min(p.x for p in projected)-18
    right=max(p.x for p in projected)+18
    shader=gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    try:
        for side,ids in ((-1,[0,2,4,6,7,12]),(1,[1,3,5,8,9,11])):
            last_y=-10000.
            for i in sorted(ids,key=lambda i:projected[i].y):
                name,_,kind=HANDLES[i];p=projected[i]
                y=max(p.y,last_y+21);last_y=y
                width=blf.dimensions(0,name)[0]
                x=max(8,left-width) if side<0 else min(right,context.region.width-width-8)
                endpoint=x+width+5 if side<0 else x-5
                batch=batch_for_shader(shader,'LINES',{'pos':[(p.x,p.y,0),(endpoint,y+5,0)]})
                shader.bind();shader.uniform_float('color',(*COLORS[kind],.45));batch.draw(shader)
                blf.color(0,*COLORS[kind],1.)
                blf.position(0,x,y,0);blf.draw(0,name)
        p=projected[10]
        blf.color(0,*COLORS['JAW'],1.)
        blf.position(0,p.x+12,p.y-8,0);blf.draw(0,'Jaw open / close')
    finally:gpu.state.blend_set('NONE')


def register_visuals():
    global _label_handle
    if _label_handle is None:
        _label_handle=bpy.types.SpaceView3D.draw_handler_add(draw_labels,(),'WINDOW','POST_PIXEL')


def stop_drag(_=None):
    global _active_drag
    _focus_views.clear()
    if _active_drag is not None:
        try:_active_drag.finish(bpy.context,cancel=True)
        except (ReferenceError,AttributeError):_active_drag=None


def unregister_visuals():
    global _label_handle
    stop_drag()
    if _label_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_label_handle,'WINDOW')
        _label_handle=None


CLASSES=(MOCAP_OT_face_drag,MOCAP_GT_face_handle,MOCAP_GGT_face_controls,MOCAP_OT_face_focus,MOCAP_OT_face_previous_view)
