bl_info = {
    'name': 'Mocap ComfyUI', 'author': 'Innovation', 'version': (0, 6, 1),
    'blender': (4, 2, 0), 'location': '3D View > Sidebar > Mocap',
    'description': 'Submit footage to ComfyUI and import editable body rigs, mesh, and cameras',
    'category': 'Animation',
}

from pathlib import Path
import base64
import json
import os
import queue
import re
import tempfile
import textwrap
import threading
import time
import uuid

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup, UIList
from bpy_extras.io_utils import ImportHelper

from .client import ComfyClient, VIDEO_SUFFIXES
from .importer import import_bundle, result_collection
from .log_view import MOCAP_OT_live_log, MOCAP_OT_close_log, draw_log_header, find_log_area, refresh_log_views
from . import plate_proxy, scene_camera
from .face import CLASSES as FACE_CLASSES
from .face_controls import CLASSES as VISUAL_CLASSES, register_visuals, unregister_visuals, stop_drag

# Monitored jobs by local key; the scene's job list keeps what survives a restart.
_runs = {}
_detect = None
_connection = {}
STAGES = [('upload', 'Upload plate'), ('prepare', 'Detect and track actor'), ('gvhmr', 'GVHMR body and world'),
          ('sam', 'SAM body and hands'), ('face', 'Face fit (FLAME)'), ('export', 'Build bundle'), ('download', 'Download'), ('import', 'Import')]
PRESETS = {
    'QUICK': dict(method='gvhmr', hand_refinement=False, face_capture='none'),
    'BODY_HANDS': dict(method='hybrid', hand_refinement=True, face_capture='none'),
    'FULL_FACE': dict(method='hybrid', hand_refinement=True, face_capture='realdenseface_flame'),
}
SENSOR_PRESETS = [('FULL_FRAME', 'Full frame 36 mm', '', 36.0), ('ALEXA35', 'ARRI ALEXA 35 open gate 27.99 mm', '', 27.99),
                  ('ALEXA_LF', 'ARRI ALEXA LF open gate 36.70 mm', '', 36.70), ('S35', 'Super 35 24.89 mm', '', 24.89), ('CUSTOM', 'Custom', '', 0.0)]


def prefs(context=None):
    return (context or bpy.context).preferences.addons[__package__].preferences


def loads(text):
    try:
        value = json.loads(text) if text else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def wrap_width(context):
    region = context.region
    return max(24, int((region.width if region else 300)/7.5))


def hint(layout, text, width, icon='NONE'):
    column = layout.column(align=True)
    column.active = False
    for index, line in enumerate(textwrap.wrap(text, width)):
        column.label(text=line, icon=icon if index == 0 else 'NONE' if icon == 'NONE' else 'BLANK1')


def duration(seconds):
    seconds = max(0, round(seconds))
    return f'{seconds//60}m {seconds%60:02}s' if seconds >= 60 else f'{seconds}s'


# ---------------------------------------------------------------- settings

def preset_changed(settings, context):
    for key, value in PRESETS.get(settings.preset, {}).items():
        setattr(settings, key, value)


def custom_changed(settings, context):
    preset = next((name for name, values in PRESETS.items() if all(getattr(settings, k) == v for k, v in values.items())), 'CUSTOM')
    if settings.preset != preset:
        settings.preset = preset
    if settings.method == 'gvhmr' and settings.face_capture != 'none':
        settings.face_capture = 'none'


def sensor_preset_changed(settings, context):
    width = dict((key, mm) for key, _, _, mm in SENSOR_PRESETS)[settings.sensor_preset]
    if width:
        settings.sensor_mm = width


def face_visual_changed(settings, context):
    if settings.face_visual and context.area and context.area.type == 'VIEW_3D':
        context.space_data.show_gizmo = True
        context.space_data.show_gizmo_context = True


def plate_file(value):
    """Absolute plate path with '..' collapsed the way Blender resolves '//' paths.

    Blender makes paths relative lexically (from the .blend path as opened), while the OS resolves
    '..' physically: behind a symlinked folder such as ~/dev-lbringer -> /mnt/share/dev-lbringer,
    the raw path would leave the tree and point at nothing.
    """
    return os.path.normpath(bpy.path.abspath(value)) if value else ''


# EXR chromaticities (rx, ry, gx, gy, bx, by) of the colour spaces offered for proxies.
PRIMARIES = {'ACEScg': (.713, .293, .165, .830, .128, .044), 'ACES2065-1': (.7347, .2653, 0., 1., .0001, -.077),
             'Linear Rec.709': (.64, .33, .30, .60, .15, .06), 'Linear Rec.2020': (.708, .292, .170, .797, .131, .046)}


def declared_colorspace(path):
    """Colour space from the EXR's declared primaries, or None (OIIO's 'oiio:ColorSpace' is only a guess)."""
    if Path(path).suffix.lower() != '.exr':
        return None
    try:
        import OpenImageIO as oiio
        image = oiio.ImageInput.open(str(path))
        if image is None:
            return None
        try:
            chromaticities = image.spec().getattribute('chromaticities')
        finally:
            image.close()
    except Exception:
        return None
    if not chromaticities or len(chromaticities) < 6:
        return None
    return next((name for name, primaries in PRIMARIES.items() if max(abs(a-b) for a, b in zip(chromaticities[:6], primaries)) < .005), None)


def probe_plate(settings, context):
    """Header-only description of the chosen video or image sequence; no pixels are read."""
    if settings.video and settings.video != plate_file(settings.video):
        settings.video = plate_file(settings.video)  # store absolute; this update runs again
        return
    settings.detect_json = ''
    path = Path(settings.video) if settings.video else None
    if path is None or not path.is_file():
        settings.plate_info = json.dumps({'error': 'File not found'} if settings.video else {})
        return
    if plate_proxy.is_sequence(path):
        try:
            seq = plate_proxy.describe(path)
        except Exception as exc:
            settings.plate_info = json.dumps({'error': f'Cannot read this image sequence: {exc}'})
            return
        fps = seq['fps'][0]/seq['fps'][1] if seq['fps'] else None
        declared = declared_colorspace(path)
        if declared:
            settings.plate_colorspace = declared
        settings.plate_info = json.dumps(dict(sequence=True, width=seq['width'], height=seq['height'], frames=seq['count'], fps=fps,
                                              fps_rational=list(seq['fps']) if seq['fps'] else None, first=seq['first'], last=seq['last'],
                                              pattern=seq['pattern'], gaps=seq['gaps'], pixel_aspect=seq['pixel_aspect'], declared_colorspace=declared))
        settings.first_frame, settings.last_frame = seq['first'], seq['last']
        return
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        settings.plate_info = json.dumps({'error': 'Choose a movie (MP4, MOV, MKV, AVI, WebM, M4V) or one frame of an image sequence'})
        return
    clip, info = None, {'error': 'Blender cannot read this video'}
    try:
        clip = bpy.data.movieclips.load(str(path), check_existing=False)
        if clip.size[0] > 0 and clip.frame_duration > 0:
            info = dict(width=int(clip.size[0]), height=int(clip.size[1]), frames=int(clip.frame_duration), fps=float(clip.fps))
    except RuntimeError:
        pass
    finally:
        if clip is not None:
            bpy.data.movieclips.remove(clip)
    settings.plate_info = json.dumps(info)
    settings.first_frame, settings.last_frame = 1, info.get('frames', 0)


def scene_camera_poll(settings, obj):
    return obj.type == 'CAMERA'


class MocapPreferences(AddonPreferences):
    bl_idname = __package__
    endpoint: StringProperty(name='ComfyUI worker', default='http://127.0.0.1:8188')
    token: StringProperty(name='Bearer token', subtype='PASSWORD', options={'SKIP_SAVE'}, description='Optional gateway token for this Blender session')
    cache_dir: StringProperty(name='Download folder', subtype='DIR_PATH', description="Leave empty to use Blender's user data folder")

    def draw(self, context):
        for prop in ('endpoint', 'token', 'cache_dir'):
            self.layout.prop(self, prop)
        self.layout.label(text='GPU inference runs on the worker; this machine needs no CUDA or models.')


class MocapJob(PropertyGroup):
    key: StringProperty()
    prompt_id: StringProperty(name='Job ID')
    label: StringProperty()
    state: StringProperty(default='NEW')
    detail: StringProperty()
    error: StringProperty()
    bundle: StringProperty(subtype='FILE_PATH')
    log_text: StringProperty()
    log_path: StringProperty(subtype='FILE_PATH')
    record: StringProperty()
    created: FloatProperty()


class MocapSettings(PropertyGroup):
    # Plate and actor
    video: StringProperty(name='Plate', subtype='FILE_PATH', update=probe_plate, description='A movie, or any frame of an EXR/DPX/TIFF/PNG/JPEG image sequence')
    plate_info: StringProperty(options={'HIDDEN'})
    plate_colorspace: EnumProperty(name='Plate colour space', items=[(key, key if key != 'RAW' else name, description) for key, name, description in plate_proxy.COLORSPACES], default='ACEScg',
                                   description='Colour space of image-sequence pixels; the upload proxy is tone-mapped to display sRGB for the networks')
    first_frame: IntProperty(name='First', default=1, description='First frame to process: 1-based for movies, the file frame number for sequences')
    last_frame: IntProperty(name='Last', default=0, min=0, description='Last frame to process (inclusive); 0 = to the end')
    actor_index: IntProperty(name='Actor', default=-1, min=-1, max=1000, description='Actors are numbered 0, 1, 2 from left to right on the first frame; -1 = largest')
    detect_json: StringProperty(options={'HIDDEN'})
    # Processing
    preset: EnumProperty(name='Quality', items=[('QUICK', 'Quick body (GVHMR)', 'Body and world motion only; fastest'),
                                                ('BODY_HANDS', 'Body + hands', 'SAM body and fingers placed by GVHMR world motion'),
                                                ('FULL_FACE', 'Body + hands + face', 'Adds RealDenseFace/FLAME head with editable expressions; slowest'),
                                                ('CUSTOM', 'Custom', 'Settings below were changed')], default='BODY_HANDS', update=preset_changed)
    method: EnumProperty(name='Body method', items=[('hybrid', 'SAM + GVHMR world/root', 'SAM body, optional hands/face, GVHMR hip position and root rotation'), ('gvhmr', 'GVHMR body + world (faster)', 'Runs only GVHMR; no SAM hands or facial stage')], default='hybrid', update=custom_changed)
    face_capture: EnumProperty(name='Face capture', items=[('none', 'Off', 'Skip facial inference'), ('sam_native', 'SAM native face', 'MHR facial expressions from the existing SAM face stage'), ('realdenseface_flame', 'RealDenseFace + FLAME head', 'Additional face fitting and joined FLAME head on SAM body; slower')], default='none', update=custom_changed)
    hand_refinement: BoolProperty(name='Hand refinement', default=True, update=custom_changed)
    sam_smoothing: FloatProperty(name='SAM smoothing', default=1., min=0., max=1.)
    max_side: IntProperty(name='Processing long edge', default=1280, min=128, max=4096, description='Pixels; frames are downscaled to this before inference')
    fps_override: FloatProperty(name='FPS override (0 = plate)', default=0., min=0., max=240.)
    warm_models: BoolProperty(name='Keep models warm on worker', default=False, description='The worker keeps GVHMR/SAM loaded in RAM between jobs (GPU memory is still freed); speeds up repeated jobs')
    # Camera
    camera_mode: EnumProperty(name='Camera', items=[('ESTIMATE', 'Estimate from footage', 'GVHMR infers the camera; focal assumed from the image diagonal'),
                                                    ('FOCAL', 'Known lens', 'Focal length and filmback known; camera motion still inferred'),
                                                    ('SCENE', 'Solved scene camera', 'Use a tracked camera in this scene (ViPE, 3DE, ...); the body is placed in its world')], default='ESTIMATE')
    moving_camera: BoolProperty(name='Moving camera', default=True)
    focal_mm: FloatProperty(name='Focal length (mm)', default=35., min=.1)
    sensor_preset: EnumProperty(name='Filmback', items=[item[:3] for item in SENSOR_PRESETS], default='FULL_FRAME', update=sensor_preset_changed)
    sensor_mm: FloatProperty(name='Sensor width (mm)', default=36., min=.1)
    scene_camera: PointerProperty(name='Scene camera', type=bpy.types.Object, poll=scene_camera_poll)
    plate_lock: FloatProperty(name='Plate lock', default=1., min=0., max=1., subtype='FACTOR', description='Sideways correction (no depth change) keeping the body on the plate under the solved camera')
    orientation_lock: FloatProperty(name='Rotation lock', default=.5, min=0., max=1., subtype='FACTOR', description="Turn the body's root toward the orientation seen in the plate (smoothed); 1 fits the plate best, 0 keeps GVHMR's world orientation")
    plate_lock_smoothing: FloatProperty(name='Lock smoothing (frames)', default=3., min=0., max=12., description='Gaussian smoothing of the plate and rotation locks; higher is steadier, lower follows the plate more closely')
    # Import
    frame_start: IntProperty(name='Blender start frame', default=1001, min=0, description='Timeline frame for the first processed movie frame (sequences keep their own numbers)')
    include_mesh: BoolProperty(name='Animated body mesh', default=True)
    include_rig: BoolProperty(name='Editable armature', default=True)
    auto_import: BoolProperty(name='Import when complete', default=True)
    camera_view: EnumProperty(name='Review camera', items=[('VIDEO', 'Video fit', 'Actor-specific camera preserving SAM projection; not a shared scene camera'), ('WORLD', 'GVHMR world camera', 'Native inferred GVHMR camera; hybrid body may not line up with the video')], default='VIDEO')
    bundle: StringProperty(name='Result bundle', subtype='FILE_PATH')
    nuke_dir: StringProperty(name='Nuke export folder', subtype='DIR_PATH')
    # Jobs and logs
    jobs: CollectionProperty(type=MocapJob)
    jobs_index: IntProperty(default=0)
    log_text: StringProperty()
    # Face corrections
    face_visual: BoolProperty(name='Show Visual Face Controls', default=False, update=face_visual_changed, description='Drag the colored handles in the viewport to correct this face')
    face_auto_key: BoolProperty(name='Key on release', default=True, description='Key facial corrections when a drag finishes; Ctrl Z undoes the drag')
    face_labels: BoolProperty(name='Control labels', default=True)
    face_advanced: BoolProperty(name='Advanced expression sliders', default=False)


# ---------------------------------------------------------------- plate selection and timing

def plate(settings):
    return loads(settings.plate_info)


def selection(settings):
    """Frames to process: dict(first, last, count, start_index) in plate numbering (start_index is zero-based)."""
    info = plate(settings)
    if 'width' not in info:
        raise ValueError(info.get('error') or 'Choose a plate')
    if info.get('sequence'):
        lo, hi = info['first'], info['last']
    else:
        lo, hi = 1, info['frames']
    first = max(lo, settings.first_frame)
    last = min(hi, settings.last_frame) if settings.last_frame else hi
    if last-first+1 < 8:
        raise ValueError('Select at least 8 frames')
    return dict(first=first, last=last, count=last-first+1, start_index=first-lo, sequence=bool(info.get('sequence')))


def timeline_start(settings, sel):
    """Timeline frame of the first processed frame."""
    if settings.camera_mode == 'SCENE':
        timing = scene_camera.vipe_timing(settings.scene_camera)
        if timing and not sel['sequence']:
            return timing['zero_frame']+sel['start_index']
        if timing and timing['plate_first_frame'] is not None:
            # ViPE stores the file number of its source frame 0; map plate numbers through it.
            return timing['zero_frame']+sel['first']-timing['plate_first_frame']
    if sel['sequence']:
        return sel['first']
    return settings.frame_start


def range_text(settings):
    try:
        sel = selection(settings)
    except ValueError as exc:
        return str(exc), 'ERROR'
    start = timeline_start(settings, sel)
    return f"{sel['count']} frames · timeline {start}–{start+sel['count']-1}", 'TIME'


def camera_check(settings, sel):
    """Problems with the chosen scene camera for this range; empty when it can be used."""
    camera = settings.scene_camera
    if camera is None:
        return ['Choose the solved camera']
    issues = []
    info = plate(settings)
    timing = scene_camera.vipe_timing(camera)
    if timing and timing['width'] and abs(timing['width']/timing['height']-info['width']/info['height']) > .01:
        issues.append(f"ViPE solved a {timing['width']}x{timing['height']} plate; this plate is {info['width']}x{info['height']}")
    keyed = scene_camera.keyed_range(camera)
    start = timeline_start(settings, sel)
    if keyed and (start < keyed[0] or start+sel['count']-1 > keyed[1]):
        issues.append(f'Camera is keyed on {keyed[0]}–{keyed[1]}; the job needs {start}–{start+sel["count"]-1}')
    return issues


# ---------------------------------------------------------------- jobs

def cache_root():
    return Path(plate_file(prefs().cache_dir) if prefs().cache_dir else bpy.utils.user_resource('DATAFILES', path='mocap_comfy', create=True))


def job_folder(key):
    folder = cache_root()/key
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def find_job(scene, key):
    return next((job for job in scene.mocap_comfy.jobs if job.key == key), None)


def selected_job(context):
    settings = context.scene.mocap_comfy
    if 0 <= settings.jobs_index < len(settings.jobs):
        return settings.jobs[settings.jobs_index]
    return None


def new_job(scene, label, record):
    settings = scene.mocap_comfy
    job = settings.jobs.add()
    job.key = uuid.uuid4().hex[:12]
    job.label, job.created, job.state = label, time.time(), 'STARTING'
    job.record = json.dumps(record)
    text = bpy.data.texts.new(f'Mocap log {label[:24]}')
    text['mocap_job_log'] = True
    job.log_text = settings.log_text = text.name
    settings.jobs.move(len(settings.jobs)-1, 0)
    settings.jobs_index = 0
    refresh_log_views(scene, text)
    return job


def start_monitor(scene, job, work):
    """Run work(run) in a thread; it reports through run['events'] and is polled on the main thread."""
    run = dict(key=job.key, scene=scene.name, events=queue.Queue(), stop=threading.Event(), client=ComfyClient(prefs().endpoint, prefs().token),
               prompt_id=job.prompt_id, started=time.monotonic(), stage='upload', packet={}, record=loads(job.record))
    _runs[job.key] = run

    def target():
        try:
            work(run)
        except Exception as exc:
            run['events'].put(('error', str(exc)))
    threading.Thread(target=target, name='MocapComfyHTTP', daemon=True).start()
    if not bpy.app.timers.is_registered(poll_events):
        bpy.app.timers.register(poll_events, first_interval=.25)
    return run


def remote_job(run, uploaded, options, camera_file=None):
    """Submit (or resume) and download; shared by Submit and Resume."""
    client, events, stop = run['client'], run['events'], run['stop']
    folder = Path(run['record']['folder'])
    log_state = dict(started=False)

    def progress(packet):
        if packet.get('log_chunk') and run['prompt_id']:
            log = folder/'job.log'
            with log.open('a' if log_state['started'] else 'w', encoding='utf-8') as stream:
                stream.write(packet['log_chunk'])
            log_state['started'] = True
            packet['log_path'] = str(log)
        events.put(('progress', packet))
    if not run['prompt_id']:
        if stop.is_set():
            raise RuntimeError('This job was cancelled')
        if camera_file:
            options['scene_camera'] = client.upload(camera_file)
        pid = client.submit(uploaded, options)
        run['prompt_id'] = pid
        events.put(('prompt_id', pid))
    output = client.wait(run['prompt_id'], report=lambda value: events.put(('status', value)), stop=stop, on_progress=progress)
    events.put(('stage', 'download'))
    path = client.download(output, folder/'result.mocap.zip', on_progress=progress)
    events.put(('done', path))


def submit_options(settings, sel):
    options = dict(method=settings.method, start_frame=0 if sel['sequence'] else sel['start_index'], max_frames=0 if sel['sequence'] else sel['count'],
                   actor_index=settings.actor_index, moving_camera=settings.moving_camera, fps_override=settings.fps_override,
                   focal_px=0., max_side=settings.max_side, sam_smoothing=settings.sam_smoothing,
                   hand_refinement=settings.hand_refinement, face_capture=settings.face_capture if settings.method == 'hybrid' else 'none')
    if settings.camera_mode == 'FOCAL':
        options['focal_px'] = settings.focal_mm/settings.sensor_mm*plate(settings)['width']
    if settings.camera_mode == 'SCENE':
        options.update(plate_lock=settings.plate_lock, orientation_lock=settings.orientation_lock, plate_lock_smoothing=settings.plate_lock_smoothing)
    if settings.warm_models:
        options['warm_models'] = True
    detected = loads(settings.detect_json)
    if detected.get('key') == detect_key(settings, sel) and 0 <= settings.actor_index < len(detected.get('boxes', [])):
        options['actor_box'] = json.dumps(detected['boxes'][settings.actor_index])
    return options


def detect_key(settings, sel):
    return f"{plate_file(settings.video)}|{sel['first']}|{settings.max_side}|{settings.plate_colorspace if sel['sequence'] else ''}"


def plate_fps(settings):
    info = plate(settings)
    if info.get('fps_rational'):
        return tuple(info['fps_rational'])
    fps = settings.fps_override or info.get('fps') or bpy.context.scene.render.fps/bpy.context.scene.render.fps_base
    return (round(fps*1000), 1000)


def ocio_config():
    """Blender's bundled OCIO config, so proxies never depend on $OCIO or the launch environment."""
    return os.path.join(bpy.utils.resource_path('LOCAL'), 'datafiles', 'colormanagement', 'config.ocio')


def sequence_proxy(video, cache_root, snapshot, progress=lambda *args: None, stop=lambda: False):
    """Proxy archive for a sequence selection, cached by the selected files and proxy settings.

    Rebuilt proxies are byte-identical, so the worker also recognises the upload.
    """
    options = dict(first_index=snapshot['first_index'], last_index=snapshot['last_index'], long_edge=snapshot['max_side'],
                   colorspace=snapshot['colorspace'], fps=snapshot['fps'])
    cache = Path(cache_root)/'proxies'/plate_proxy.proxy_key(video, **options)
    found = plate_proxy.cached_proxy(cache)
    if found:
        progress('Plate proxy reused', 1.)
        return found
    return plate_proxy.build_proxy(video, cache, ocio_config=snapshot['ocio_config'], progress=progress, stop=stop, **options)


def build_upload(run, settings_snapshot, progress):
    """Movie path, or the proxy ZIP of an image-sequence selection (built on the artist machine)."""
    record = run['record']
    if not record['sequence']:
        return record['video']
    archive, plate_json, first = sequence_proxy(record['video'], Path(record['folder']).parent, settings_snapshot,
                                                lambda stage, fraction: progress(dict(stage=stage, done=fraction, total=1.)), run['stop'].is_set)
    record['background'] = str(first)
    record['plate'] = plate_json
    run['events'].put(('record', record))
    return str(archive)


def begin_submit(context):
    scene, settings = context.scene, context.scene.mocap_comfy
    sel = selection(settings)
    info = plate(settings)
    if sel['sequence'] and abs(info.get('pixel_aspect', 1.)-1) > 1e-3:
        raise ValueError('Anamorphic plates are not supported yet; desqueeze the plate first')
    start = timeline_start(settings, sel)
    camera_json = None
    if settings.camera_mode == 'SCENE':
        issues = camera_check(settings, sel)
        if issues:
            raise ValueError(issues[0])
        camera_json = scene_camera.sample(context, settings.scene_camera, list(range(start, start+sel['count'])), info['width'], info['height'])
    options = submit_options(settings, sel)
    video = plate_file(settings.video)
    label = f"{Path(video).stem[:20]} · {sel['first']}–{sel['last']} · {settings.preset.replace('_', ' ').lower()}"
    record = dict(video=video, sequence=sel['sequence'], timeline_start=start, first=sel['first'], count=sel['count'],
                  include_mesh=settings.include_mesh, include_rig=settings.include_rig, auto_import=settings.auto_import,
                  camera_view=settings.camera_view, options=options, camera=settings.scene_camera.name if camera_json else '',
                  plate=None, background='')
    job = new_job(scene, label, record)
    record['folder'] = str(job_folder(job.key))
    job.record = json.dumps(record)
    camera_file = None
    if camera_json:
        camera_file = Path(record['folder'])/'scene_camera.json'
        camera_file.write_text(json.dumps(camera_json))
    snapshot = dict(first_index=sel['start_index']+1, last_index=sel['start_index']+sel['count'], max_side=settings.max_side,
                    colorspace=settings.plate_colorspace, fps=plate_fps(settings), ocio_config=ocio_config())

    def work(run):
        client = run['client']
        info = client.check()
        if options['face_capture'] != 'none' and 'face_capture' not in info['MocapProcessVideo']['input'].get('optional', {}):
            raise RuntimeError('Restart ComfyUI to load the updated facial capture node')

        def progress(packet):
            run['events'].put(('progress', packet))
        path = build_upload(run, snapshot, progress)
        uploaded = client.upload(path, on_progress=progress)
        run['events'].put(('stage', 'prepare'))
        remote_job(run, uploaded, options, camera_file)
    start_monitor(scene, job, work)


def begin_resume(context, job):
    if job.key in _runs:
        raise RuntimeError('This job is already being monitored')
    if not job.prompt_id:
        raise ValueError('This job never reached the worker; submit it again')
    record = loads(job.record)
    if not record.get('folder'):
        record['folder'] = str(job_folder(job.key))
        job.record = json.dumps(record)
    job.state, job.error = 'RESUMING', ''
    start_monitor(context.scene, job, lambda run: remote_job(run, None, None))


def import_job(scene, job):
    record = loads(job.record)
    background = record.get('background') or record.get('video', '')
    collection = import_bundle(job.bundle, record.get('timeline_start', 1001), record.get('include_mesh', True), background, scene,
                               camera_view=record.get('camera_view', 'VIDEO'), include_rig=record.get('include_rig', True), plate=record)
    job.state = 'IMPORTED'
    job.detail = collection.name
    return collection


def poll_events():
    for key, run in list(_runs.items()):
        scene = bpy.data.scenes.get(run['scene'])
        job = find_job(scene, key) if scene else None
        if job is None:
            run['stop'].set()
            del _runs[key]
            continue
        while not run['events'].empty():
            kind, value = run['events'].get_nowait()
            if kind == 'prompt_id':
                job.prompt_id = value
                job.state = 'QUEUED'
            elif kind == 'record':
                job.record = json.dumps(value)
            elif kind == 'stage':
                run['stage'] = value
            elif kind == 'status':
                job.detail = value
                if value.startswith('Running'):
                    job.state = 'RUNNING'
            elif kind == 'progress':
                run['packet'] = value
                job.detail = value.get('stage', job.detail)
                if value.get('stage_key') in dict(STAGES):
                    run['stage'] = value['stage_key']
                    job.state = 'RUNNING'
                if value.get('log_path'):
                    job.log_path = value['log_path']
                if value.get('log_chunk'):
                    text = bpy.data.texts.get(job.log_text)
                    clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', value['log_chunk']).replace('\r', '\n')
                    if text:
                        content = (text.as_string()+clean)[-100000:]
                        text.clear()
                        text.write(content)
            elif kind == 'error':
                job.error = value
                job.state = 'CANCELLED' if value == 'This job was cancelled' else 'FAILED'
                print('[Mocap ComfyUI]', value)
                text = bpy.data.texts.get(job.log_text)
                if text:
                    text.write('\n[Client error] '+value+'\n')
                del _runs[key]
                break
            elif kind == 'done':
                job.bundle, job.state, job.detail = value, 'DOWNLOADED', 'Result downloaded'
                del _runs[key]
                if loads(job.record).get('auto_import', True):
                    run['stage'] = 'import'
                    try:
                        import_job(scene, job)
                    except Exception as exc:
                        job.error = 'Import failed: '+str(exc)
                break
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type in {'VIEW_3D', 'TEXT_EDITOR'}:
                area.tag_redraw()
    return .25 if _runs or _detect else None


# ---------------------------------------------------------------- operators

class MOCAP_OT_check(Operator):
    bl_idname = 'mocap_comfy.check'
    bl_label = 'Check Connection'
    bl_description = 'Ask the worker whether the mocap node is installed and which features it supports'

    def execute(self, context):
        url = prefs(context).endpoint
        try:
            client = ComfyClient(url, prefs(context).token)
            features = client.features()
            queue_info = client.request('/queue', timeout=10)
        except Exception as exc:
            _connection.update(url=url, state='ERROR', detail=str(exc)[:200])
            self.report({'ERROR'}, f'Worker check failed: {exc}')
            return {'CANCELLED'}
        missing = [name.replace('_', ' ') for name, ok in features.items() if not ok]
        busy = len(queue_info.get('queue_running', []))+len(queue_info.get('queue_pending', []))
        detail = (f'{busy} job(s) in the worker queue' if busy else 'Worker idle')+(f" · older node: no {', '.join(missing)}" if missing else '')
        _connection.update(url=url, state='OK', detail=detail)
        self.report({'INFO'}, detail)
        return {'FINISHED'}


class MOCAP_OT_submit(Operator):
    bl_idname = 'mocap_comfy.submit'
    bl_label = 'Submit to ComfyUI'
    bl_description = 'Upload the plate (once) and queue a mocap job; you can queue several actors or ranges'

    def execute(self, context):
        try:
            begin_submit(context)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class MOCAP_OT_full_clip(Operator):
    bl_idname = 'mocap_comfy.full_clip'
    bl_label = 'Full Clip'
    bl_description = 'Select every frame of the plate'

    def execute(self, context):
        settings = context.scene.mocap_comfy
        info = plate(settings)
        if 'width' not in info:
            return {'CANCELLED'}
        settings.first_frame, settings.last_frame = (info['first'], info['last']) if info.get('sequence') else (1, info['frames'])
        return {'FINISHED'}


class MOCAP_OT_detect(Operator):
    bl_idname = 'mocap_comfy.detect'
    bl_label = 'Detect Actors'
    bl_description = 'Upload the plate (reused by the job) and show numbered actors on the first selected frame'

    def execute(self, context):
        global _detect
        settings = context.scene.mocap_comfy
        if _detect is not None:
            self.report({'INFO'}, 'Actor detection is already running')
            return {'CANCELLED'}
        try:
            sel = selection(settings)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        video, key, scene_name = plate_file(settings.video), detect_key(settings, sel), context.scene.name
        # A two-frame proxy (the archive minimum) at the job's processing size gives identical boxes.
        snapshot = dict(first_index=sel['start_index']+1, last_index=sel['start_index']+2, max_side=settings.max_side,
                        colorspace=settings.plate_colorspace, fps=plate_fps(settings), fps_override=settings.fps_override, ocio_config=ocio_config(),
                        cache_root=str(cache_root()))
        client = ComfyClient(prefs(context).endpoint, prefs(context).token)
        _detect = dict(scene=scene_name, events=queue.Queue(), detail='Uploading the plate')

        def work():
            try:
                if sel['sequence']:
                    _detect['events'].put(('detail', 'Preparing the first frames of the plate'))
                    path, _, _ = sequence_proxy(video, snapshot['cache_root'], snapshot)
                    frame = 0
                else:
                    path, frame = video, sel['start_index']
                uploaded = client.upload(path)
                _detect['events'].put(('detail', 'Detecting actors on the worker'))
                result = client.detect(uploaded, frame, snapshot['max_side'], snapshot['fps_override'])
                result['key'] = key
                _detect['events'].put(('done', result))
            except Exception as exc:
                _detect['events'].put(('error', str(exc)))
        threading.Thread(target=work, daemon=True).start()
        bpy.app.timers.register(poll_detect, first_interval=.25)
        return {'FINISHED'}


def poll_detect():
    global _detect
    if _detect is None:
        return None
    scene = bpy.data.scenes.get(_detect['scene'])
    while not _detect['events'].empty():
        kind, value = _detect['events'].get_nowait()
        if kind == 'detail':
            _detect['detail'] = value
        elif kind == 'error':
            _detect = None
            if scene:
                scene.mocap_comfy.detect_json = json.dumps(dict(error=value))
            break
        elif kind == 'done':
            _detect = None
            if scene:
                image = bpy.data.images.get('Mocap actor preview')
                if image is not None:
                    bpy.data.images.remove(image)
                path = Path(tempfile.gettempdir())/f'mocap-actors-{uuid.uuid4().hex[:8]}.jpg'
                path.write_bytes(base64.b64decode(value.pop('preview_jpeg')))
                image = bpy.data.images.load(str(path))
                image.name = 'Mocap actor preview'
                image.use_fake_user = True  # Kept when the .blend is saved, although nothing uses it.
                image.pack()
                path.unlink(missing_ok=True)
                image.preview_ensure()
                scene.mocap_comfy.detect_json = json.dumps(value)
                if value['boxes'] and not 0 <= scene.mocap_comfy.actor_index < len(value['boxes']):
                    scene.mocap_comfy.actor_index = value['largest']
            break
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
    return .25 if _detect is not None else None


class MOCAP_OT_pick_actor(Operator):
    bl_idname = 'mocap_comfy.pick_actor'
    bl_label = 'Pick Actor'
    bl_description = 'Process this actor'
    index: IntProperty()

    def execute(self, context):
        context.scene.mocap_comfy.actor_index = self.index
        return {'FINISHED'}


class MOCAP_OT_show_actors(Operator):
    bl_idname = 'mocap_comfy.show_actors'
    bl_label = 'Show Full Size'
    bl_description = 'Open the actor preview in an image window'

    def execute(self, context):
        image = bpy.data.images.get('Mocap actor preview')
        if image is None:
            return {'CANCELLED'}
        bpy.ops.wm.window_new()
        area = context.window_manager.windows[-1].screen.areas[0]
        area.type = 'IMAGE_EDITOR'
        area.spaces.active.image = image
        return {'FINISHED'}


class MOCAP_OT_use_scene_camera(Operator):
    bl_idname = 'mocap_comfy.use_scene_camera'
    bl_label = 'Use Active Scene Camera'
    bl_description = "Use the scene's active camera as the solved camera"

    def execute(self, context):
        if context.scene.camera is None:
            self.report({'ERROR'}, 'The scene has no active camera')
            return {'CANCELLED'}
        context.scene.mocap_comfy.scene_camera = context.scene.camera
        return {'FINISHED'}


class JobOperator:
    def job(self, context):
        job = selected_job(context)
        if job is None:
            self.report({'ERROR'}, 'Select a job in the list')
        return job


class MOCAP_OT_resume(JobOperator, Operator):
    bl_idname = 'mocap_comfy.resume'
    bl_label = 'Resume'
    bl_description = 'Reconnect to the selected job on the worker without submitting it again'

    def execute(self, context):
        job = self.job(context)
        if job is None:
            return {'CANCELLED'}
        try:
            begin_resume(context, job)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class MOCAP_OT_cancel(JobOperator, Operator):
    bl_idname = 'mocap_comfy.cancel'
    bl_label = 'Cancel Job'
    bl_description = 'Cancel only this job on the worker'

    def execute(self, context):
        job = self.job(context)
        run = _runs.get(job.key) if job else None
        if run is None:
            return {'CANCELLED'}
        if not run['prompt_id']:
            run['stop'].set()
            run['events'].put(('error', 'This job was cancelled'))
            return {'FINISHED'}

        def cancel():
            try:
                if run['client'].cancel(run['prompt_id']).get('cancelled'):
                    run['stop'].set()
                    run['events'].put(('error', 'This job was cancelled'))
            except Exception as exc:
                run['events'].put(('status', 'Cancel failed: '+str(exc)))
        threading.Thread(target=cancel, daemon=True).start()
        return {'FINISHED'}


class MOCAP_OT_import_job(JobOperator, Operator):
    bl_idname = 'mocap_comfy.import_job'
    bl_label = 'Import'
    bl_description = "Import the selected job's downloaded result as a new collection"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        job = self.job(context)
        if job is None or not job.bundle or not Path(job.bundle).is_file():
            self.report({'ERROR'}, 'This job has no downloaded result yet')
            return {'CANCELLED'}
        try:
            import_job(context.scene, job)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class MOCAP_OT_remove_job(JobOperator, Operator):
    bl_idname = 'mocap_comfy.remove_job'
    bl_label = 'Remove from List'
    bl_description = 'Remove the selected job from this list (downloaded files and imported objects stay)'

    def execute(self, context):
        settings = context.scene.mocap_comfy
        job = self.job(context)
        if job is None or job.key in _runs:
            return {'CANCELLED'}
        settings.jobs.remove(settings.jobs_index)
        settings.jobs_index = min(settings.jobs_index, len(settings.jobs)-1)
        return {'FINISHED'}


class MOCAP_OT_import_file(Operator, ImportHelper):
    bl_idname = 'mocap_comfy.import_result'
    bl_label = 'Import Saved Bundle'
    bl_description = 'Import a downloaded .mocap.zip without running inference'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.zip'
    filter_glob: StringProperty(default='*.zip', options={'HIDDEN'})

    def execute(self, context):
        settings = context.scene.mocap_comfy
        try:
            video = plate_file(settings.video)
            import_bundle(self.filepath, settings.frame_start, settings.include_mesh, video if Path(video).suffix.lower() in VIDEO_SUFFIXES else '',
                          camera_view=settings.camera_view, include_rig=settings.include_rig)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class MOCAP_OT_open_job_log(JobOperator, Operator):
    bl_idname = 'mocap_comfy.open_job_log'
    bl_label = 'Show Log'
    bl_description = "Show the selected job's log in the live log pane"

    def execute(self, context):
        job = self.job(context)
        if job is None or bpy.data.texts.get(job.log_text) is None:
            return {'CANCELLED'}
        context.scene.mocap_comfy.log_text = job.log_text
        return bpy.ops.mocap_comfy.live_log()


class MOCAP_OT_review_camera(Operator):
    bl_idname = 'mocap_comfy.review_camera'
    bl_label = 'Switch Review Camera'
    view: EnumProperty(items=[('VIDEO', 'Video fit', ''), ('WORLD', 'GVHMR world', ''), ('SCENE', 'Scene camera', '')])

    def execute(self, context):
        scene = context.scene
        collection = active_result(context)
        if collection is None:
            self.report({'ERROR'}, 'Import a result first')
            return {'CANCELLED'}
        if self.view == 'SCENE':
            camera = bpy.data.objects.get(collection.get('scene_camera', ''))
        else:
            prefixes = ('Video Fit Camera', 'Actor_fit_camera') if self.view == 'VIDEO' else ('GVHMR World Camera', 'GVHMR_inferred_camera')
            camera = next((o for o in collection.all_objects if o.type == 'CAMERA' and (o.get('mocap_camera_role') == self.view or o.name.startswith(prefixes))), None)
            if camera is None:
                camera = next((o for o in collection.all_objects if o.type == 'CAMERA' and o.get('mocap_camera_role') == 'WORLD'), None)
        if camera is None:
            return {'CANCELLED'}
        camera.hide_set(False)
        scene.camera = camera
        if self.view != 'SCENE':
            scene.mocap_comfy.camera_view = self.view
        if context.area and context.area.type == 'VIEW_3D':
            context.area.spaces.active.region_3d.view_perspective = 'CAMERA'
        return {'FINISHED'}


class MOCAP_OT_pose_rig(Operator):
    bl_idname = 'mocap_comfy.pose_rig'
    bl_label = 'Edit Body / Finger Bones'
    def execute(self, context):
        collection = active_result(context)
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
        for name in ('Body', 'Hands'):
            if name in arm.data.collections:
                arm.data.collections[name].is_visible = True
        bpy.ops.object.mode_set(mode='POSE')
        return {'FINISHED'}


class MOCAP_OT_export_nuke(Operator):
    bl_idname = 'mocap_comfy.export_nuke'
    bl_label = 'Export to Nuke'
    bl_description = 'Write a Nuke script (plate, camera, body Alembic), a .chan camera and an Alembic of the animated body'
    directory: StringProperty(subtype='DIR_PATH')

    def invoke(self, context, event):
        settings = context.scene.mocap_comfy
        if settings.nuke_dir:
            self.directory = plate_file(settings.nuke_dir)
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from .nuke_export import export_nuke
        try:
            result = export_nuke(context, active_result(context), self.directory)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        context.scene.mocap_comfy.nuke_dir = self.directory
        for warning in result['warnings']:
            self.report({'WARNING'}, warning)
        self.report({'INFO'}, 'Nuke export written: '+result['nk'])
        return {'FINISHED'}


def active_result(context):
    for obj in (context.object, context.scene.camera):
        collection = result_collection(obj)
        if collection is not None:
            return collection
    return next((c for c in reversed(bpy.data.collections) if 'bundle' in c and c.get('method')), None)


# ---------------------------------------------------------------- panels

class MOCAP_UL_jobs(UIList):
    ICONS = {'IMPORTED': 'CHECKMARK', 'DOWNLOADED': 'IMPORT', 'FAILED': 'ERROR', 'CANCELLED': 'CANCEL', 'RUNNING': 'PLAY', 'QUEUED': 'SORTTIME'}

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        state = 'RUNNING' if item.key in _runs and item.state not in ('QUEUED',) else item.state
        layout.label(text=item.label or item.prompt_id[:8], icon=self.ICONS.get(state, 'TIME'))


def draw_job_status(layout, job, width):
    run = _runs.get(job.key)
    box = layout.box()
    box.label(text=job.label, icon='SEQUENCE')
    if run is None:
        text = {'IMPORTED': 'Imported', 'DOWNLOADED': 'Downloaded, not imported', 'FAILED': 'Failed', 'CANCELLED': 'Cancelled'}.get(job.state, job.state.title())
        for line in textwrap.wrap(text+(' · '+job.detail if job.detail and job.state != 'FAILED' else ''), width)[:3]:
            box.label(text=line)
        if job.error:
            column = box.column(align=True)
            column.alert = True
            for line in textwrap.wrap(job.error, width)[:6]:
                column.label(text=line)
        return
    packet = run['packet']
    done, total = packet.get('done'), packet.get('total')
    if done is not None and total:
        detail = f'{done/1048576:.0f} / {total/1048576:.0f} MiB' if packet.get('unit') == 'bytes' else f'{done:.0%}' if total == 1. else f'{done} / {total} frames'
        box.progress(factor=min(1., done/total), type='BAR', text=detail)
    clock = f"Elapsed {duration(time.monotonic()-run['started'])}"
    estimated = packet.get('estimated_total_seconds')
    if estimated and packet.get('elapsed_seconds') is not None:
        remaining = estimated-packet['elapsed_seconds']
        clock += f' · about {duration(remaining)} left (recent runs)' if remaining > 0 else ' · longer than recent runs'
    elif packet.get('phase_eta_seconds') is not None:
        clock += f" · this step about {duration(packet['phase_eta_seconds'])}"
    box.label(text=clock, icon='TIME')
    if job.state == 'QUEUED':
        box.label(text=job.detail or 'Queued on the worker', icon='SORTTIME')
    plan = loads(job.record).get('options', {})
    keys = ['upload', 'prepare', 'gvhmr']+(['sam'] if plan.get('method') == 'hybrid' else [])+(['face'] if plan.get('face_capture') == 'realdenseface_flame' else [])+['export', 'download', 'import']
    current = keys.index(run['stage']) if run['stage'] in keys else 0
    column = box.column(align=True)
    for index, key in enumerate(keys):
        row = column.row()
        row.active = index <= current
        row.label(text=dict(STAGES)[key], icon='CHECKMARK' if index < current else 'PLAY' if index == current else 'RADIOBUT_OFF')
        if index == current and job.detail:
            hint(column, job.detail, width-4, 'BLANK1')
    if packet.get('seconds_since_log') is not None and packet.get('state') == 'running':
        hint(box, f"Worker active · last log line {duration(packet['seconds_since_log'])} ago", width)


class MOCAP_PT_panel(Panel):
    bl_label = 'Mocap • ComfyUI'
    bl_idname = 'MOCAP_PT_comfy'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Mocap'

    def draw(self, context):
        layout, settings, width = self.layout, context.scene.mocap_comfy, wrap_width(context)
        url = prefs(context).endpoint
        state = _connection.get('state') if _connection.get('url') == url else None
        text, icon = {'OK': (_connection.get('detail', 'Worker connected'), 'CHECKMARK'), 'ERROR': ('Worker check failed', 'ERROR')}.get(state, ('Worker not checked', 'QUESTION'))
        row = layout.row(align=True)
        row.label(text=text[:60], icon=icon)
        row.operator('mocap_comfy.check', text='', icon='FILE_REFRESH')
        row = layout.row()
        row.scale_y = 1.4
        row.operator('mocap_comfy.submit', icon='PLAY')
        job = selected_job(context)
        if job is not None:
            draw_job_status(layout, job, width)
            if job.key in _runs:
                layout.operator('mocap_comfy.cancel', icon='CANCEL')
            elif job.state == 'DOWNLOADED':
                layout.operator('mocap_comfy.import_job', text='Import Result', icon='IMPORT')
        running = sum(1 for key in _runs if find_job(context.scene, key))
        if running > 1:
            hint(layout, f'{running} jobs monitored · see Jobs', width)


class SubPanel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Mocap'
    bl_parent_id = 'MOCAP_PT_comfy'


class MOCAP_PT_worker(SubPanel, Panel):
    bl_label = 'Worker'
    bl_idname = 'MOCAP_PT_worker'
    bl_order = 0
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout, p = self.layout, prefs(context)
        layout.prop(p, 'endpoint', text='URL')
        layout.prop(p, 'token')
        layout.prop(p, 'cache_dir')
        layout.prop(context.scene.mocap_comfy, 'warm_models')
        layout.operator('mocap_comfy.check', icon='CHECKMARK')


class MOCAP_PT_footage(SubPanel, Panel):
    bl_label = 'Plate and actor'
    bl_idname = 'MOCAP_PT_footage'
    bl_order = 1

    def draw(self, context):
        layout, settings, width = self.layout, context.scene.mocap_comfy, wrap_width(context)
        layout.prop(settings, 'video')
        info = plate(settings)
        if info.get('error'):
            row = layout.row()
            row.alert = True
            row.label(text=info['error'][:80], icon='ERROR')
        elif 'width' in info:
            fps = f"{info['fps']:.3f}".rstrip('0').rstrip('.')+' fps' if info.get('fps') else 'fps unknown'
            kind = f"Sequence {info['pattern']} {info['first']}–{info['last']}" if info.get('sequence') else f"Movie · {info['frames']} frames"
            hint(layout, f"{kind} · {info['width']}×{info['height']} · {fps}", width, 'SEQUENCE' if info.get('sequence') else 'FILE_MOVIE')
            if info.get('gaps'):
                layout.label(text=f"{info['gaps']} frames missing in the sequence", icon='ERROR')
        row = layout.row(align=True)
        row.prop(settings, 'first_frame')
        row.prop(settings, 'last_frame')
        row.operator('mocap_comfy.full_clip', text='', icon='ARROW_LEFTRIGHT')
        text, icon = range_text(settings)
        layout.label(text=text, icon=icon)
        if info.get('sequence'):
            layout.prop(settings, 'plate_colorspace', text='Colour')
            hint(layout, 'Primaries declared in the EXR header' if info.get('declared_colorspace') == settings.plate_colorspace else
                 'The files do not declare their primaries: check the plate colour space', width, 'INFO')
        elif settings.camera_mode != 'SCENE':
            layout.prop(settings, 'frame_start')
        layout.separator()
        row = layout.row(align=True)
        row.operator('mocap_comfy.detect', icon='VIEWZOOM', text='Detecting…' if _detect else 'Detect Actors')
        row.prop(settings, 'actor_index', text='Actor')
        if _detect:
            hint(layout, _detect['detail'], width)
        detected = loads(settings.detect_json)
        if detected.get('error'):
            column = layout.column(align=True)
            column.alert = True
            for line in textwrap.wrap(detected['error'], width)[:3]:
                column.label(text=line)
        elif detected.get('boxes') is not None:
            try:
                current = detected.get('key') == detect_key(settings, selection(settings))
            except ValueError:
                current = False
            image = bpy.data.images.get('Mocap actor preview')
            if image is not None:
                # Previews are not saved with the .blend; regenerate after reopening.
                layout.template_icon(icon_value=image.preview_ensure().icon_id, scale=9)
            if not current:
                hint(layout, 'Preview is for another plate, range or size; detect again', width, 'ERROR')
            elif not detected['boxes']:
                layout.label(text='No person on this frame; change First', icon='ERROR')
            grid = layout.grid_flow(columns=4, even_columns=True, align=True)
            for index in range(len(detected['boxes'])):
                grid.operator('mocap_comfy.pick_actor', text=str(index), depress=settings.actor_index == index).index = index
            if image is not None:
                layout.operator('mocap_comfy.show_actors', icon='WINDOW')
        else:
            hint(layout, 'Detect Actors shows numbered people on the first frame. Without it: 0, 1, 2 left to right; -1 = largest.', width)


class MOCAP_PT_camera(SubPanel, Panel):
    bl_label = 'Camera'
    bl_idname = 'MOCAP_PT_camera'
    bl_order = 2

    def draw(self, context):
        layout, settings, width = self.layout, context.scene.mocap_comfy, wrap_width(context)
        layout.prop(settings, 'camera_mode', text='')
        if settings.camera_mode in ('ESTIMATE', 'FOCAL'):
            layout.prop(settings, 'moving_camera')
        if settings.camera_mode == 'FOCAL':
            column = layout.column(align=True)
            column.prop(settings, 'focal_mm')
            column.prop(settings, 'sensor_preset', text='')
            column.prop(settings, 'sensor_mm')
        if settings.camera_mode != 'SCENE':
            return
        row = layout.row(align=True)
        row.prop(settings, 'scene_camera', text='')
        row.operator('mocap_comfy.use_scene_camera', text='', icon='OUTLINER_OB_CAMERA')
        camera = settings.scene_camera
        timing = scene_camera.vipe_timing(camera)
        if camera is not None:
            hint(layout, ('ViPE solve: timing taken from its job' if timing else 'Generic camera: video frames start at the Blender start frame below'), width, 'INFO')
            if not timing:
                layout.prop(settings, 'frame_start')
            if timing and timing['distortion']:
                hint(layout, 'The ViPE camera has lens distortion; the body is fitted on the distorted plate', width, 'ERROR')
        try:
            issues = camera_check(settings, selection(settings))
        except ValueError:
            issues = []
        for issue in issues:
            column = layout.column(align=True)
            column.alert = True
            for line in textwrap.wrap(issue, width):
                column.label(text=line)
        column = layout.column(align=True)
        column.prop(settings, 'plate_lock')
        column.prop(settings, 'orientation_lock')
        column.prop(settings, 'plate_lock_smoothing')
        hint(layout, "World motion stays GVHMR's. Plate lock slides the body sideways (never in depth); rotation lock turns it toward the plate. Rotation lock 1 fits best on plate-critical shots.", width)


class MOCAP_PT_process(SubPanel, Panel):
    bl_label = 'Processing'
    bl_idname = 'MOCAP_PT_process'
    bl_order = 3

    def draw(self, context):
        layout, settings = self.layout, context.scene.mocap_comfy
        layout.prop(settings, 'preset')
        column = layout.column(align=True)
        column.prop(settings, 'method', text='')
        if settings.method == 'hybrid':
            column.prop(settings, 'hand_refinement')
            column.prop(settings, 'face_capture', text='Face')
            column.prop(settings, 'sam_smoothing')
        layout.prop(settings, 'max_side')
        layout.prop(settings, 'fps_override')


class MOCAP_PT_import(SubPanel, Panel):
    bl_label = 'Import'
    bl_idname = 'MOCAP_PT_import'
    bl_order = 4
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout, settings = self.layout, context.scene.mocap_comfy
        for prop in ('include_mesh', 'include_rig', 'auto_import'):
            layout.prop(settings, prop)
        if settings.camera_mode != 'SCENE':
            layout.prop(settings, 'camera_view')
        layout.operator('mocap_comfy.import_result', icon='FILE_FOLDER')


class MOCAP_PT_results(SubPanel, Panel):
    bl_label = 'Result'
    bl_idname = 'MOCAP_PT_results'
    bl_order = 5

    @classmethod
    def poll(cls, context):
        return active_result(context) is not None

    def draw(self, context):
        layout, width = self.layout, wrap_width(context)
        collection = active_result(context)
        summary = loads(collection.get('summary', ''))
        column = layout.column(align=True)
        column.label(text=collection.name[:60], icon='OUTLINER_COLLECTION')
        for line in summary.get('lines', [collection.get('rig_note', '')]):
            for wrapped in textwrap.wrap(line, width):
                column.label(text=wrapped)
        row = layout.row(align=True)
        if collection.get('scene_camera'):
            row.operator('mocap_comfy.review_camera', text='Scene camera', icon='CAMERA_DATA').view = 'SCENE'
        else:
            row.operator('mocap_comfy.review_camera', text='Video fit').view = 'VIDEO'
        row.operator('mocap_comfy.review_camera', text='GVHMR camera').view = 'WORLD'
        layout.operator('mocap_comfy.pose_rig', icon='ARMATURE_DATA')
        layout.operator('mocap_comfy.export_nuke', text='Export to Nuke…', icon='EXPORT')


class MOCAP_PT_jobs(SubPanel, Panel):
    bl_label = 'Jobs'
    bl_idname = 'MOCAP_PT_jobs'
    bl_order = 7
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout, settings = self.layout, context.scene.mocap_comfy
        layout.template_list('MOCAP_UL_jobs', '', settings, 'jobs', settings, 'jobs_index', rows=3)
        job = selected_job(context)
        if job is None:
            return
        hint(layout, f"Job {job.prompt_id[:8] or 'not submitted'} · {job.state.lower()}", wrap_width(context))
        row = layout.row(align=True)
        row.enabled = job.key not in _runs
        row.operator('mocap_comfy.resume', icon='FILE_REFRESH')
        row.operator('mocap_comfy.import_job', icon='IMPORT')
        row = layout.row(align=True)
        row.operator('mocap_comfy.open_job_log', icon='TEXT')
        row.operator('mocap_comfy.remove_job', text='', icon='X')


class MOCAP_PT_logs(SubPanel, Panel):
    bl_label = 'Logs'
    bl_idname = 'MOCAP_PT_logs'
    bl_order = 8
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.operator('mocap_comfy.live_log', icon='TEXT')
        if find_log_area(context.screen):
            layout.operator('mocap_comfy.close_log', icon='X')
        job = selected_job(context)
        if job and job.log_path:
            layout.prop(job, 'log_path', text='Saved log')


@persistent
def stop_monitoring(_):
    global _detect
    stop_drag()
    for run in _runs.values():
        run['stop'].set()
    _runs.clear()
    _detect = None
    for timer in (poll_events, poll_detect):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)


CLASSES = (MocapPreferences, MocapJob, MocapSettings, MOCAP_OT_check, MOCAP_OT_submit, MOCAP_OT_full_clip, MOCAP_OT_detect, MOCAP_OT_pick_actor,
           MOCAP_OT_show_actors, MOCAP_OT_use_scene_camera, MOCAP_OT_resume, MOCAP_OT_cancel, MOCAP_OT_import_job, MOCAP_OT_remove_job,
           MOCAP_OT_import_file, MOCAP_OT_open_job_log, MOCAP_OT_review_camera, MOCAP_OT_pose_rig, MOCAP_OT_export_nuke,
           MOCAP_OT_live_log, MOCAP_OT_close_log, MOCAP_UL_jobs, MOCAP_PT_panel, MOCAP_PT_worker, MOCAP_PT_footage, MOCAP_PT_camera,
           MOCAP_PT_process, MOCAP_PT_import, MOCAP_PT_results, MOCAP_PT_jobs, MOCAP_PT_logs)+FACE_CLASSES+VISUAL_CLASSES


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mocap_comfy = PointerProperty(type=MocapSettings)
    bpy.app.handlers.load_pre.append(stop_monitoring)
    bpy.types.TEXT_HT_header.prepend(draw_log_header)
    register_visuals()


def unregister():
    unregister_visuals()
    stop_monitoring(None)
    bpy.types.TEXT_HT_header.remove(draw_log_header)
    bpy.app.handlers.load_pre.remove(stop_monitoring)
    del bpy.types.Scene.mocap_comfy
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
