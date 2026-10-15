bl_info = {
    'name': 'Mocap ComfyUI', 'author': 'Innovation', 'version': (0, 3, 1),
    'blender': (4, 2, 0), 'location': '3D View > Sidebar > Mocap',
    'description': 'Submit footage to ComfyUI and import editable body rigs, mesh, and cameras',
    'category': 'Animation',
}

from pathlib import Path
import json
import queue
import re
import threading

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup

from .client import ComfyClient
from .importer import import_bundle, result_collection
from .log_view import MOCAP_OT_live_log, MOCAP_OT_close_log, draw_log_header, find_log_area, refresh_log_views

_run = None


def method_changed(settings, context):
    if settings.method == 'gvhmr':
        settings.face_capture = 'none'


class MocapSettings(PropertyGroup):
    endpoint: StringProperty(name='ComfyUI URL', default='http://127.0.0.1:8188')
    token: StringProperty(name='Bearer token', subtype='PASSWORD', options={'SKIP_SAVE'})
    video: StringProperty(name='Video', subtype='FILE_PATH')
    method: EnumProperty(name='Body method', items=[('hybrid', 'SAM + GVHMR world/root', 'SAM body, optional hands/face, GVHMR hip position and root rotation'), ('gvhmr', 'GVHMR body + world (faster)', 'Runs only GVHMR; no SAM hands or facial stage')], default='hybrid', update=method_changed)
    face_capture: EnumProperty(name='Face capture', items=[('none', 'Off', 'Skip facial inference'), ('sam_native', 'SAM native face', 'MHR facial expressions from the existing SAM face stage'), ('realdenseface_flame', 'RealDenseFace + FLAME head', 'Additional face fitting and joined FLAME head on SAM body; slower')], default='none')
    camera_view: EnumProperty(name='Review camera', items=[('VIDEO', 'Video fit', 'Actor-specific camera preserving SAM projection; not a shared scene camera'), ('WORLD', 'GVHMR world camera', 'Native inferred GVHMR camera; hybrid body may not line up with the video')], default='VIDEO')
    start_frame: IntProperty(name='Video start (zero-based)', default=0, min=0)
    max_frames: IntProperty(name='Frame count (0 = all)', default=0, min=0, max=100000)
    actor_index: IntProperty(name='Actor (-1 = largest)', default=-1, min=-1, max=1000, description='First selected frame: actors numbered 0, 1, 2 from left to right; tracked through the range')
    moving_camera: BoolProperty(name='Moving camera', default=True)
    fps_override: FloatProperty(name='FPS override (0 = video)', default=0., min=0., max=240.)
    focal_px: FloatProperty(name='Focal pixels (0 = estimate)', default=0., min=0., max=100000., description='Original video width units; principal point assumed centered')
    max_side: IntProperty(name='Processing longest side', default=1280, min=128, max=4096)
    sam_smoothing: FloatProperty(name='SAM smoothing', default=1., min=0., max=1.)
    hand_refinement: BoolProperty(name='SAM hand refinement', default=True)
    frame_start: IntProperty(name='Blender start frame', default=1001, min=1)
    include_mesh: BoolProperty(name='Import animated body mesh', default=True)
    include_rig: BoolProperty(name='Import editable armature', default=True)
    auto_import: BoolProperty(name='Import when complete', default=True)
    cache_dir: StringProperty(name='Download folder', subtype='DIR_PATH')
    prompt_id: StringProperty(name='Job ID')
    bundle: StringProperty(name='Result bundle', subtype='FILE_PATH')
    status: StringProperty(name='Status', default='Ready')
    error: StringProperty(name='Last error')
    last_timing: StringProperty(name='Stage timings')
    advanced: BoolProperty(name='Advanced settings', default=False)
    progress_fraction: FloatProperty(default=0., min=0., max=1.)
    progress_detail: StringProperty()
    elapsed_detail: StringProperty()
    eta_detail: StringProperty()
    log_text: StringProperty()
    log_path: StringProperty(subtype='FILE_PATH')
    last_log_line: StringProperty()
    activity_detail: StringProperty()
    rig_note: StringProperty()


def begin_job(scene, resume=False):
    global _run
    if _run is not None:
        raise RuntimeError('A job is already being monitored')
    settings = scene.mocap_comfy
    video = bpy.path.abspath(settings.video)
    if not resume and not Path(video).is_file():
        raise ValueError('Choose a local video file')
    if resume and not settings.prompt_id:
        raise ValueError('Enter a job ID to resume')
    options = {key: getattr(settings, key) for key in ('method', 'start_frame', 'max_frames', 'actor_index', 'moving_camera', 'fps_override', 'focal_px', 'max_side', 'sam_smoothing', 'hand_refinement', 'face_capture')}
    folder = bpy.path.abspath(settings.cache_dir) if settings.cache_dir else bpy.utils.user_resource('DATAFILES', path='mocap_comfy', create=True)
    client = ComfyClient(settings.endpoint, settings.token)
    prompt_id = settings.prompt_id if resume else ''
    events, stop = queue.Queue(), threading.Event()
    job = dict(events=events, stop=stop, client=client, prompt_id=prompt_id, scene=scene.name,
               video=video, frame_start=settings.frame_start, include_mesh=settings.include_mesh,
               auto_import=settings.auto_import, camera_view=settings.camera_view, include_rig=settings.include_rig)
    _run = job
    settings.error = ''
    settings.status = 'Connecting to ComfyUI'
    settings.progress_fraction = 0.
    settings.progress_detail = settings.elapsed_detail = settings.eta_detail = settings.last_log_line = ''
    settings.last_timing = ''
    settings.activity_detail = ''
    settings.log_text = bpy.data.texts.new('Mocap job log').name
    refresh_log_views(scene, bpy.data.texts[settings.log_text])
    settings.log_path = ''
    def worker():
        try:
            info = client.check()
            if options['face_capture'] != 'none' and 'face_capture' not in info['MocapProcessVideo']['input'].get('optional', {}):
                raise RuntimeError('Restart ComfyUI to load the updated facial capture node')
            pid = prompt_id
            log_started = False
            def progress(packet):
                nonlocal log_started
                if packet.get('log_chunk') and pid:
                    log = Path(folder)/pid/'job.log'
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open('a' if log_started else 'w', encoding='utf-8') as stream:
                        stream.write(packet['log_chunk'])
                    log_started = True
                    packet['log_path'] = str(log)
                events.put(('progress', packet))
            if not resume:
                events.put(('status', 'Uploading video'))
                uploaded = client.upload(video, on_progress=progress)
                pid = client.submit(uploaded, options)
                events.put(('prompt_id', pid))
            output = client.wait(pid, report=lambda value: events.put(('status', value)), stop=stop, on_progress=progress)
            events.put(('status', 'Downloading result'))
            path = client.download(output, Path(folder)/pid/'result.mocap.zip', on_progress=progress)
            events.put(('done', path))
        except Exception as exc:
            events.put(('error', str(exc)))
    threading.Thread(target=worker, name='MocapComfyHTTP', daemon=True).start()
    bpy.app.timers.register(poll_events, first_interval=.25)


def poll_events():
    global _run
    job = _run
    if job is None:
        return None
    scene = bpy.data.scenes.get(job['scene'])
    if scene is None:
        job['stop'].set()
        _run = None
        return None
    settings = scene.mocap_comfy
    while not job['events'].empty():
        kind, value = job['events'].get_nowait()
        if kind == 'prompt_id':
            settings.prompt_id = job['prompt_id'] = value
        elif kind == 'status':
            settings.status = value
        elif kind == 'progress':
            settings.status = value.get('stage', settings.status)
            done, total = value.get('done'), value.get('total')
            settings.progress_fraction = min(1., done/total) if done is not None and total else 0.
            if done is not None and total:
                settings.progress_detail = (f'{done/1048576:.1f} / {total/1048576:.1f} MiB' if value.get('unit') == 'bytes' else f'{done} / {total} frames in this step')
            else:
                settings.progress_detail = ''
            if value.get('stage_number'):
                settings.progress_detail = f'Stage {value["stage_number"]}/{value["stage_count"]} · '+settings.progress_detail
            if 'elapsed_seconds' in value:
                settings.elapsed_detail = 'Processing elapsed: '+duration(value['elapsed_seconds'])
            phase_eta = value.get('phase_eta_seconds')
            estimated = value.get('estimated_total_seconds')
            settings.eta_detail = 'This step: approximately '+duration(phase_eta)+' left' if phase_eta is not None else ''
            if estimated:
                remaining = estimated-value.get('elapsed_seconds', 0.)
                settings.eta_detail = 'Job: roughly '+duration(remaining)+' left (recent runs)' if remaining > 0 else 'Taking longer than the recent-run estimate'
            elif not settings.eta_detail and value.get('state') == 'running':
                settings.eta_detail = 'Remaining time: not enough information yet'
            if value.get('log_path'):
                settings.log_path = value['log_path']
            if 'seconds_since_log' in value:
                settings.activity_detail = ('Worker running · last log '+duration(value['seconds_since_log'])+' ago') if value.get('state') == 'running' else 'Worker '+value.get('state', '')
            if value.get('log_chunk'):
                text = bpy.data.texts.get(settings.log_text)
                clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', value['log_chunk']).replace('\r', '\n')
                if text:
                    content = (text.as_string()+clean)[-100000:]
                    text.clear()
                    text.write(content)
                lines = [line for line in clean.splitlines() if line.strip()]
                if lines:
                    settings.last_log_line = lines[-1]
        elif kind == 'error':
            settings.error = value
            settings.status = 'Failed — see last error; completed jobs can be resumed'
            print('[Mocap ComfyUI]', value)
            text = bpy.data.texts.get(settings.log_text)
            if text:
                text.write('\n[Client error] '+value+'\n')
            _run = None
        elif kind == 'done':
            settings.bundle = value
            settings.status = 'Downloaded'
            _run = None
            if job['auto_import']:
                try:
                    settings.status = 'Importing mesh, bones and cameras into Blender'
                    collection = import_bundle(value, job['frame_start'], job['include_mesh'], job['video'], scene, camera_view=job['camera_view'], include_rig=job['include_rig'])
                    settings.last_timing = timing_summary(collection)
                    settings.rig_note = collection['rig_note']
                    settings.status = 'Complete — cameras and requested body imported'
                    settings.progress_fraction = 1.
                    settings.progress_detail = 'Complete'
                    settings.eta_detail = ''
                except Exception as exc:
                    settings.error = str(exc)
                    settings.status = 'Downloaded; import failed — see last error'
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type in {'VIEW_3D', 'TEXT_EDITOR'}:
                area.tag_redraw()
    return .25 if _run is not None else None


class MOCAP_OT_submit(Operator):
    bl_idname = 'mocap_comfy.submit'
    bl_label = 'Submit to ComfyUI'
    def execute(self, context):
        try:
            begin_job(context.scene)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class MOCAP_OT_resume(Operator):
    bl_idname = 'mocap_comfy.resume'
    bl_label = 'Resume Job'
    def execute(self, context):
        try:
            begin_job(context.scene, resume=True)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class MOCAP_OT_cancel(Operator):
    bl_idname = 'mocap_comfy.cancel'
    bl_label = 'Cancel This Job'
    def execute(self, context):
        job = _run
        if not job or not job['prompt_id']:
            self.report({'WARNING'}, 'Wait until the upload has returned a job ID')
            return {'CANCELLED'}
        def cancel():
            try:
                response = job['client'].cancel(job['prompt_id'])
                if response.get('cancelled'):
                    job['stop'].set()
                    job['events'].put(('error', 'This job was cancelled'))
            except Exception as exc:
                job['events'].put(('status', 'Cancel failed: '+str(exc)))
        threading.Thread(target=cancel, daemon=True).start()
        return {'FINISHED'}


class MOCAP_OT_import(Operator):
    bl_idname = 'mocap_comfy.import_result'
    bl_label = 'Import Downloaded Bundle'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        settings = context.scene.mocap_comfy
        try:
            collection = import_bundle(bpy.path.abspath(settings.bundle), settings.frame_start, settings.include_mesh, bpy.path.abspath(settings.video), camera_view=settings.camera_view, include_rig=settings.include_rig)
            settings.last_timing = timing_summary(collection)
            settings.rig_note = collection['rig_note']
            settings.status = 'Imported into a new Mocap collection'
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


def timing_summary(collection):
    timings = json.loads(collection.get('timings', '{}'))
    return ' | '.join(f'{name}: {timings[key]:.0f}s' for name, key in [('Prep', 'prepare_seconds'), ('GVHMR', 'gvhmr_seconds'), ('SAM', 'sam_seconds'), ('Face', 'face_seconds'), ('Export', 'export_seconds')] if key in timings)


def duration(seconds):
    seconds = max(0, round(seconds))
    return f'{seconds//60}m {seconds%60:02}s' if seconds >= 60 else f'{seconds}s'


class MOCAP_OT_review_camera(Operator):
    bl_idname = 'mocap_comfy.review_camera'
    bl_label = 'Switch Review Camera'
    view: EnumProperty(items=[('VIDEO', 'Video fit', ''), ('WORLD', 'GVHMR world', '')])

    def execute(self, context):
        scene = context.scene
        current = scene.camera
        if current is None:
            self.report({'ERROR'}, 'Import a result first')
            return {'CANCELLED'}
        collection = result_collection(current)
        if collection is None:
            self.report({'ERROR'}, 'Activate a camera from the desired Mocap collection first')
            return {'CANCELLED'}
        prefixes = ('Video Fit Camera', 'Actor_fit_camera') if self.view == 'VIDEO' else ('GVHMR World Camera', 'GVHMR_inferred_camera')
        camera = next((o for o in collection.all_objects if o.type == 'CAMERA' and (o.get('mocap_camera_role') == self.view or o.name.startswith(prefixes))), None)
        if camera is None:
            camera = next((o for o in collection.all_objects if o.type == 'CAMERA' and (o.get('mocap_camera_role') == 'WORLD' or o.name.startswith(('GVHMR World Camera', 'GVHMR_inferred_camera')))), None)
        if camera is None:
            return {'CANCELLED'}
        scene.camera = camera
        scene.mocap_comfy.camera_view = self.view
        if context.area and context.area.type == 'VIEW_3D':
            context.area.spaces.active.region_3d.view_perspective = 'CAMERA'
        return {'FINISHED'}


class MOCAP_OT_pose_rig(Operator):
    bl_idname = 'mocap_comfy.pose_rig'
    bl_label = 'Edit Body / Finger Bones'
    def execute(self, context):
        collection = result_collection(context.object) or result_collection(context.scene.camera)
        arm = next((o for o in collection.all_objects if o.type == 'ARMATURE'), None) if collection else None
        if arm is None:
            self.report({'ERROR'}, 'This result has no armature. Submit a new job with the updated server, then import with Editable armature enabled.')
            return {'CANCELLED'}
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in context.selected_objects:
            obj.select_set(False)
        arm.hide_set(False)
        arm.select_set(True)
        context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode='POSE')
        return {'FINISHED'}


class MOCAP_PT_panel(Panel):
    bl_label = 'Mocap • ComfyUI'
    bl_idname = 'MOCAP_PT_comfy'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Mocap'
    def draw(self, context):
        layout, s = self.layout, context.scene.mocap_comfy
        layout.prop(s, 'endpoint')
        row = layout.row()
        row.enabled = _run is None
        row.operator('mocap_comfy.submit', icon='PLAY')
        if _run is not None:
            layout.operator('mocap_comfy.cancel', icon='CANCEL')
        layout.label(text=s.status[:90])
        if s.progress_detail:
            layout.progress(factor=s.progress_fraction, text=s.progress_detail)
        if s.error:
            layout.label(text=s.error[:110], icon='ERROR')


class MocapChildPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Mocap'
    bl_parent_id = 'MOCAP_PT_comfy'


class MOCAP_PT_footage(MocapChildPanel, Panel):
    bl_label = '1. Footage and actor'
    bl_idname = 'MOCAP_PT_footage'
    bl_order = 0
    def draw(self, context):
        layout, s = self.layout, context.scene.mocap_comfy
        for prop in ('video', 'start_frame', 'max_frames', 'actor_index'):
            layout.prop(s, prop)
        layout.label(text='Actor indices: 0, 1, 2 from left to right')


class MOCAP_PT_process(MocapChildPanel, Panel):
    bl_label = '2. Process'
    bl_idname = 'MOCAP_PT_process'
    bl_order = 1
    def draw(self, context):
        layout, s = self.layout, context.scene.mocap_comfy
        layout.prop(s, 'method')
        layout.prop(s, 'moving_camera')
        if s.method == 'hybrid':
            for prop in ('sam_smoothing', 'hand_refinement', 'face_capture'):
                layout.prop(s, prop)
        else:
            layout.label(text='Body/world only; SAM and face skipped')
        layout.prop(s, 'advanced', icon='TRIA_DOWN' if s.advanced else 'TRIA_RIGHT', emboss=False)
        if s.advanced:
            for prop in ('max_side', 'fps_override', 'focal_px', 'token', 'cache_dir'):
                layout.prop(s, prop)


class MOCAP_PT_import(MocapChildPanel, Panel):
    bl_label = '3. Bring into Blender'
    bl_idname = 'MOCAP_PT_import'
    bl_order = 2
    def draw(self, context):
        layout, s = self.layout, context.scene.mocap_comfy
        for prop in ('frame_start', 'include_mesh', 'include_rig', 'auto_import', 'camera_view'):
            layout.prop(s, prop)
        row = layout.row(align=True)
        row.operator('mocap_comfy.review_camera', text='Video fit').view = 'VIDEO'
        row.operator('mocap_comfy.review_camera', text='GVHMR world').view = 'WORLD'
        layout.label(text='Video fit camera is specific to this actor')
        layout.operator('mocap_comfy.pose_rig', icon='ARMATURE_DATA')
        if s.rig_note:
            layout.label(text=s.rig_note[:90])


class MOCAP_PT_job(MocapChildPanel, Panel):
    bl_label = '4. Progress and logs'
    bl_idname = 'MOCAP_PT_job'
    bl_order = 3
    def draw(self, context):
        layout, s = self.layout, context.scene.mocap_comfy
        if s.elapsed_detail:
            layout.label(text=s.elapsed_detail)
        if s.eta_detail:
            layout.label(text=s.eta_detail)
        if s.activity_detail:
            layout.label(text=s.activity_detail)
        for part in s.last_timing.split(' | ') if s.last_timing else []:
            layout.label(text=part)
        if s.last_log_line:
            layout.label(text=s.last_log_line[:95])
        layout.operator('mocap_comfy.live_log', icon='TEXT')
        if find_log_area(context.screen):
            layout.operator('mocap_comfy.close_log', icon='X')
        if s.log_path:
            layout.prop(s, 'log_path', text='Saved log')
        layout.prop(s, 'prompt_id')
        row = layout.row()
        row.enabled = _run is None
        row.operator('mocap_comfy.resume')


class MOCAP_PT_bundle(MocapChildPanel, Panel):
    bl_label = '5. Saved results'
    bl_idname = 'MOCAP_PT_bundle'
    bl_order = 4
    bl_options = {'DEFAULT_CLOSED'}
    def draw(self, context):
        self.layout.prop(context.scene.mocap_comfy, 'bundle')
        self.layout.operator('mocap_comfy.import_result', icon='IMPORT')


@persistent
def stop_monitoring(_):
    global _run
    if _run is not None:
        _run['stop'].set()
        _run = None
    if bpy.app.timers.is_registered(poll_events):
        bpy.app.timers.unregister(poll_events)


CLASSES = (MocapSettings, MOCAP_OT_submit, MOCAP_OT_resume, MOCAP_OT_cancel, MOCAP_OT_import, MOCAP_OT_review_camera, MOCAP_OT_pose_rig, MOCAP_OT_live_log, MOCAP_OT_close_log, MOCAP_PT_panel, MOCAP_PT_footage, MOCAP_PT_process, MOCAP_PT_import, MOCAP_PT_job, MOCAP_PT_bundle)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mocap_comfy = PointerProperty(type=MocapSettings)
    bpy.app.handlers.load_pre.append(stop_monitoring)
    bpy.types.TEXT_HT_header.prepend(draw_log_header)


def unregister():
    stop_monitoring(None)
    bpy.types.TEXT_HT_header.remove(draw_log_header)
    bpy.app.handlers.load_pre.remove(stop_monitoring)
    del bpy.types.Scene.mocap_comfy
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
