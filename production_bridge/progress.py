"""Small atomic progress snapshots shared by the isolated inference workers."""
import json
import os
from pathlib import Path
import time

_last_write = 0.
_phase = None
_phase_started = None


def report(stage, detail, done=None, total=None, phase=None, frames=None):
    global _last_write, _phase, _phase_started
    target = os.environ.get('MOCAP_STATUS_FILE')
    if not target:
        return
    now = time.time()
    key = (stage, phase or detail)
    changed = key != _phase
    if changed:
        _phase, _phase_started = key, now
        print('[Mocap]', detail, flush=True)
    if not changed and now-_last_write < .5 and done != total:
        return
    _last_write = now
    value = dict(stage_key=stage, stage=detail, phase=key[1], phase_started=_phase_started,
                 updated_at=now, done=done, total=total, frames=frames)
    path = Path(target)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)
