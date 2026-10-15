import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mocap_client', ROOT/'mocap_comfy/client.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
p = argparse.ArgumentParser()
p.add_argument('--video', type=Path, required=True)
p.add_argument('--frames', type=int, default=24)
p.add_argument('--method', choices=['hybrid','gvhmr'], default='hybrid')
p.add_argument('--face', choices=['none','sam_native','realdenseface_flame'], default='none')
p.add_argument('--no-hands', action='store_true')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--endpoint', default='http://127.0.0.1:8188')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
client = module.ComfyClient(a.endpoint)
client.check()
video = client.upload(a.video)
options = dict(method=a.method, start_frame=0, max_frames=a.frames, actor_index=-1, moving_camera=True,
               fps_override=0., focal_px=0., max_side=960, sam_smoothing=1., hand_refinement=not a.no_hands, face_capture=a.face)
prompt_id = client.submit(video, options)
(a.out/'job.json').write_text(json.dumps(dict(prompt_id=prompt_id, uploaded_video=video, options=options), indent=2))
print('Submitted', prompt_id, flush=True)
last = ''
def report(message):
    global last
    if message != last:
        print(message, flush=True)
        last = message
def progress(packet):
    snapshot = {key: value for key, value in packet.items() if key != 'log_chunk'}
    snapshot['log_bytes_received'] = len(packet.get('log_chunk', '').encode())
    with (a.out/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(snapshot)+'\n')
    if packet.get('log_chunk'):
        with (a.out/'job.log').open('a') as stream:
            stream.write(packet['log_chunk'])
result = client.wait(prompt_id, report=report, on_progress=progress)
path = client.download(result, a.out/'result.mocap.zip')
(a.out/'api_result.json').write_text(json.dumps(dict(prompt_id=prompt_id, output=result, bundle=path), indent=2))
print('Downloaded', path, flush=True)
