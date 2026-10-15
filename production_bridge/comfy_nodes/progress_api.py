"""Job-scoped, bounded log reads through the existing ComfyUI HTTP server."""
import json
from pathlib import Path
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
    return {key: options.get(key) for key in ('method', 'face_capture', 'hand_refinement', 'moving_camera', 'max_side')}


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
