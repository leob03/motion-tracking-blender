"""Job-scoped, bounded log reads through the existing ComfyUI HTTP server."""
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import uuid

from aiohttp import web
import folder_paths
from server import PromptServer


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def signature(options):
    return {key: options.get(key) for key in ('method', 'face_capture', 'hand_refinement', 'moving_camera', 'max_side', 'warm_models')}


def estimate_seconds(root, options, frames):
    path = root/'timing_history.json'
    if not frames or not path.exists():
        return None
    samples = [row for row in json.loads(path.read_text())
               if row['options'] == signature(options) and .5 <= frames/row['frames'] <= 2]
    if not samples:
        return None
    # Comparable frame ranges only; this remains a rough estimate including startup.
    values = sorted(row['seconds']*frames/row['frames'] for row in samples[-5:])
    return values[len(values)//2]


def record_timing(root, options, frames, seconds):
    path = root/'timing_history.json'
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(dict(options=signature(options), frames=frames, seconds=seconds))
    atomic_json(path, history[-40:])


@PromptServer.instance.routes.get('/mocap/status/{job_id}')
async def status(request):
    try:
        job_id = uuid.UUID(request.match_info['job_id']).hex
        offset = max(0, int(request.query.get('offset', '0')))
    except ValueError:
        return web.json_response({'error': 'Invalid job id or offset'}, status=400)
    root = Path(folder_paths.get_output_directory())/'mocap'/job_id
    path = root/'status.json'
    if not path.is_file():
        return web.json_response({'error': 'No worker status yet'}, status=404)
    result = json.loads(path.read_text())
    log = root/'job.log'
    if log.exists():
        size = log.stat().st_size
        offset = min(offset, size)
        with log.open('rb') as stream:
            stream.seek(offset)
            chunk = stream.read(65536)
            result['log_offset'] = stream.tell()
        result['log_chunk'] = chunk.decode('utf-8', errors='replace')
        result['log_size'] = size
        result['seconds_since_log'] = max(0., time.time()-log.stat().st_mtime)
    return web.json_response(result)


UPLOAD_NAME = re.compile(r'^[0-9a-f]{64}\.(mp4|mov|mkv|avi|webm|m4v|zip|json)$')


def uploaded(name):
    """Content-addressed upload in input/mocap; names are <sha256>.<ext> only."""
    if not UPLOAD_NAME.match(name or ''):
        raise ValueError('Invalid upload name')
    return Path(folder_paths.get_input_directory())/'mocap'/name


@PromptServer.instance.routes.get('/mocap/upload/{name}')
async def upload_exists(request):
    """Lets the add-on skip re-uploading a plate the worker already has."""
    try:
        path = uploaded(request.match_info['name'])
    except ValueError as exc:
        return web.json_response({'error': str(exc)}, status=400)
    return web.json_response({'exists': path.is_file(), 'size': path.stat().st_size if path.is_file() else None})


_detect_lock = asyncio.Lock()


@PromptServer.instance.routes.post('/mocap/detect')
async def detect(request):
    """Numbered person boxes on one frame of an uploaded plate (a few seconds; no inference queue)."""
    from . import BRIDGE, PYTHON, STUDY
    try:
        body = await request.json()
        source = uploaded(body['video'])
        frame, max_side = int(body.get('frame', 0)), int(body.get('max_side', 1280))
        fps = float(body.get('fps_override', 0.))
    except (ValueError, KeyError, TypeError) as exc:
        return web.json_response({'error': f'Invalid request: {exc}'}, status=400)
    if not source.is_file():
        return web.json_response({'error': 'Upload the video first'}, status=404)
    async with _detect_lock:
        with tempfile.TemporaryDirectory(prefix='mocap-detect-') as scratch:
            output = Path(scratch)/'detect.json'
            child = await asyncio.create_subprocess_exec(
                PYTHON, '-u', str(BRIDGE/'worker.py'), '--study', str(STUDY), '--video', str(source),
                '--detect', str(frame), '--max-side', str(max_side), '--fps-override', str(fps), '--output', str(output),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1'))
            try:
                log, _ = await asyncio.wait_for(child.communicate(), timeout=300)
            except asyncio.TimeoutError:
                child.kill()
                return web.json_response({'error': 'Actor detection timed out'}, status=504)
            if child.returncode or not output.is_file():
                tail = log.decode('utf-8', errors='replace').strip().splitlines()[-1:] or ['unknown error']
                return web.json_response({'error': 'Actor detection failed: '+tail[0]}, status=500)
            return web.json_response(json.loads(output.read_text()))
