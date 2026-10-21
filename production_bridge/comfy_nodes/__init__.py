"""A ComfyUI output node exposing the tested mocap pipeline as an API job."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import uuid

import folder_paths
import comfy.model_management
import comfy.utils
from .progress_api import atomic_json, estimate_seconds, record_timing

BRIDGE = Path(__file__).resolve().parents[1]
STUDY = Path(os.environ.get('MOCAP_STUDY_ROOT', str(BRIDGE.parent/'comparison_study'))).resolve()
PYTHON = os.environ.get('MOCAP_PYTHON', str(STUDY/'.venv/bin/python'))


class MocapProcessVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'video': ('STRING', {'default': ''}),
            'method': (['hybrid', 'gvhmr'],),
            'start_frame': ('INT', {'default': 0, 'min': 0}),
            'max_frames': ('INT', {'default': 120, 'min': 0, 'max': 100000}),
            'actor_index': ('INT', {'default': -1, 'min': -1, 'max': 1000}),
            'moving_camera': ('BOOLEAN', {'default': True}),
            'fps_override': ('FLOAT', {'default': 0., 'min': 0., 'max': 240.}),
            'focal_px': ('FLOAT', {'default': 0., 'min': 0., 'max': 100000.}),
            'max_side': ('INT', {'default': 1280, 'min': 128, 'max': 4096}),
            'sam_smoothing': ('FLOAT', {'default': 1., 'min': 0., 'max': 1.}),
            'hand_refinement': ('BOOLEAN', {'default': True}),
        }, 'optional': {
            'face_capture': (['none', 'sam_native', 'realdenseface_flame'], {'default': 'none'}),
            'progress_id': ('STRING', {'default': ''}),
            'scene_camera': ('STRING', {'default': '', 'tooltip': 'Uploaded scene-camera JSON from the Blender add-on (solved camera)'}),
            'actor_box': ('STRING', {'default': '', 'tooltip': 'Normalized [x1,y1,x2,y2] picked in the actor preview'}),
            'plate_lock': ('FLOAT', {'default': 1., 'min': 0., 'max': 1.}),
            'plate_lock_smoothing': ('FLOAT', {'default': 3., 'min': 0., 'max': 12.}),
            'orientation_lock': ('FLOAT', {'default': .5, 'min': 0., 'max': 1.}),
            'warm_models': ('BOOLEAN', {'default': False, 'tooltip': 'Keep GVHMR/SAM loaded in host RAM between jobs'}),
        }}

    RETURN_TYPES = ()
    FUNCTION = 'process'
    CATEGORY = 'MotionCapture/Production'
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')

    def process(self, video, **options):
        progress_id = options.pop('progress_id', '')
        options.setdefault('face_capture', 'none')
        if options['method'] not in ('gvhmr', 'hybrid'):
            raise ValueError('Unknown mocap method')
        if options['face_capture'] not in ('none', 'sam_native', 'realdenseface_flame'):
            raise ValueError('Unknown facial capture method')
        if options['method'] != 'hybrid' and options['face_capture'] != 'none':
            raise ValueError('Facial capture currently requires the hybrid SAM body')
        input_root = Path(folder_paths.get_input_directory()).resolve()
        source = (input_root/video).resolve()
        if not source.is_relative_to(input_root) or not source.is_file():
            raise ValueError('Video must be an uploaded file inside the ComfyUI input directory')
        camera_name = options.pop('scene_camera', '')
        camera = (input_root/camera_name).resolve() if camera_name else None
        if camera is not None and (not camera.is_relative_to(input_root) or not camera.is_file()):
            raise ValueError('Scene camera must be an uploaded file inside the ComfyUI input directory')
        box = options.pop('actor_box', '')
        if box:
            box = json.loads(box)
            if len(box) != 4 or not all(0 <= float(v) <= 1 for v in box):
                raise ValueError('Actor box must be four normalized coordinates')
            options['actor_box'] = [float(v) for v in box]
        job_id = uuid.UUID(progress_id).hex if progress_id else uuid.uuid4().hex
        output = Path(folder_paths.get_output_directory())/'mocap'/job_id
        output.mkdir(parents=True)
        bundle = output/'result.mocap.zip'
        if camera is not None:
            options['scene_camera'] = camera_name
        plan = ['prepare', 'gvhmr']+(['sam'] if options['method'] == 'hybrid' else [])+(['face'] if options['face_capture'] == 'realdenseface_flame' else [])+['export']
        progress = comfy.utils.ProgressBar(len(plan))
        started = time.time()
        state = dict(state='running', stage_key='prepare', stage='Starting worker', stages=plan,
                     stage_number=1, stage_count=len(plan), elapsed_seconds=0., heartbeat_at=started)
        atomic_json(output/'status.json', state)
        with tempfile.TemporaryDirectory(prefix='mocap-job-') as scratch:
            work = Path(scratch)
            config = work/'options.json'
            config.write_text(json.dumps(options))
            command = [PYTHON, '-u', str(BRIDGE/'worker.py'), '--study', str(STUDY), '--job-root', str(work), '--video', str(source), '--options', str(config), '--output', str(bundle)]
            if camera is not None:
                command += ['--camera', str(camera)]
            env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', MOCAP_STATUS_FILE=str(work/'status.json'), PYTHONUNBUFFERED='1')
            with (output/'job.log').open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                rate_phase = None
                rate_sample = None
                try:
                    while True:
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        try:
                            code = child.wait(timeout=1)
                            break
                        except subprocess.TimeoutExpired:
                            path = work/'status.json'
                            if path.exists():
                                update = json.loads(path.read_text())
                                state.update({key: value for key, value in update.items() if value is not None})
                                # Counts from a previous step must not survive a stage change.
                                state['done'], state['total'] = update.get('done'), update.get('total')
                            now = time.time()
                            state.update(elapsed_seconds=now-started, heartbeat_at=now)
                            stage = state['stage_key']
                            stage_index = plan.index(stage) if stage in plan else len(plan)-1
                            state['stage_number'] = stage_index+1
                            progress.update_absolute(stage_index, len(plan))
                            done, total = state.get('done'), state.get('total')
                            state['phase_eta_seconds'] = None
                            phase = (stage, state.get('phase_started'))
                            if phase != rate_phase:
                                rate_phase, rate_sample = phase, None
                            if done and total and 0 < done < total:
                                if rate_sample is None:
                                    rate_sample = (done, now)
                                elif done > rate_sample[0] and now-rate_sample[1] > 1:
                                    state['phase_eta_seconds'] = (now-rate_sample[1])/(done-rate_sample[0])*(total-done)
                            expected = estimate_seconds(output.parent, options, state.get('frames'))
                            state['estimated_total_seconds'] = expected
                            atomic_json(output/'status.json', state)
                    if code:
                        with (output/'job.log').open('rb') as tail:
                            tail.seek(max(0, (output/'job.log').stat().st_size-2500))
                            detail = tail.read().decode('utf-8', errors='replace')
                        raise RuntimeError('Mocap job failed:\n'+detail)
                    state.update(state='complete', stage_key='complete', stage='Complete', done=None, total=None,
                                 elapsed_seconds=time.time()-started, heartbeat_at=time.time(), phase_eta_seconds=None)
                    atomic_json(output/'status.json', state)
                    progress.update_absolute(len(plan), len(plan))
                    if state.get('frames'):
                        record_timing(output.parent, options, state['frames'], state['elapsed_seconds'])
                except (Exception, comfy.model_management.InterruptProcessingException) as exc:
                    state.update(state='cancelled' if isinstance(exc, comfy.model_management.InterruptProcessingException) else 'error',
                                 error=str(exc)[-2500:], heartbeat_at=time.time(), elapsed_seconds=time.time()-started)
                    atomic_json(output/'status.json', state)
                    raise
                finally:
                    # Kill the worker AND its inference child on cancellation/failure.
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
        digest = hashlib.sha256()
        with bundle.open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''):
                digest.update(block)
        return {'ui': {'mocap_bundle': [{'filename': bundle.name, 'subfolder': f'mocap/{job_id}', 'type': 'output', 'sha256': digest.hexdigest(), 'progress_id': job_id}]}}


NODE_CLASS_MAPPINGS = {'MocapProcessVideo': MocapProcessVideo}
NODE_DISPLAY_NAME_MAPPINGS = {'MocapProcessVideo': 'Process Mocap for Blender'}
