"""Standard-library ComfyUI client; usable in Blender on Windows and Linux."""
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import uuid


class ComfyClient:
    def __init__(self, endpoint, token=''):
        self.endpoint = endpoint.rstrip('/')
        if urlparse(endpoint).scheme not in ('http', 'https'):
            raise ValueError('ComfyUI URL must begin with http:// or https://')
        self.headers = {'Authorization': 'Bearer '+token} if token else {}
        self.progress_id = ''
        self.log_offset = 0

    def request(self, route, payload=None, timeout=30):
        data = None if payload is None else json.dumps(payload).encode()
        headers = dict(self.headers)
        if data is not None:
            headers['Content-Type'] = 'application/json'
        request = Request(self.endpoint+route, data=data, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read(6000).decode('utf-8', errors='replace')
            raise RuntimeError(f'ComfyUI HTTP {exc.code}: {detail}') from exc

    def check(self):
        info = self.request('/object_info/MocapProcessVideo')
        if 'MocapProcessVideo' not in info:
            raise RuntimeError('This server does not have the MocapProcessVideo custom node')
        return info

    def upload(self, video, on_progress=None):
        path = Path(video)
        if path.suffix.lower() not in ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'):
            raise ValueError('Choose an MP4, MOV, MKV, AVI, WebM, or M4V video')
        boundary = 'mocap-'+uuid.uuid4().hex
        filename = uuid.uuid4().hex+path.suffix.lower()
        prefix = (f'--{boundary}\r\nContent-Disposition: form-data; name="type"\r\n\r\ninput\r\n'
                  f'--{boundary}\r\nContent-Disposition: form-data; name="subfolder"\r\n\r\nmocap\r\n'
                  f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
                  'Content-Type: application/octet-stream\r\n\r\n').encode()
        suffix = f'\r\n--{boundary}--\r\n'.encode()
        with path.open('rb') as source:
            def chunks():
                last = 0.
                yield prefix
                for block in iter(lambda: source.read(1024*1024), b''):
                    yield block
                    now = time.monotonic()
                    if on_progress and now-last > .5:
                        on_progress(dict(stage='Uploading video', done=source.tell(), total=path.stat().st_size, unit='bytes'))
                        last = now
                yield suffix
            headers = dict(self.headers, **{'Content-Type': 'multipart/form-data; boundary='+boundary,
                                            'Content-Length': str(len(prefix)+path.stat().st_size+len(suffix))})
            with urlopen(Request(self.endpoint+'/upload/image', data=chunks(), headers=headers), timeout=300) as response:
                result = json.load(response)
        return '/'.join(x for x in (result.get('subfolder', ''), result['name']) if x)

    def submit(self, uploaded_video, options):
        self.progress_id = uuid.uuid4().hex
        self.log_offset = 0
        graph = {'1': {'class_type': 'MocapProcessVideo', 'inputs': dict(options, video=uploaded_video, progress_id=self.progress_id)}}
        response = self.request('/prompt', {'prompt': graph, 'client_id': uuid.uuid4().hex})
        if response.get('node_errors'):
            raise RuntimeError(json.dumps(response['node_errors']))
        return str(uuid.UUID(response['prompt_id']))

    def job_progress(self):
        if not self.progress_id:
            return None
        token = uuid.UUID(self.progress_id).hex
        request = Request(self.endpoint+f'/mocap/status/{token}?offset={self.log_offset}', headers=self.headers)
        try:
            with urlopen(request, timeout=10) as response:
                result = json.load(response)
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        self.log_offset = result.get('log_offset', self.log_offset)
        return result

    def progress_from_graph(self, graph):
        for node in graph.values():
            if node.get('class_type') == 'MocapProcessVideo':
                self.progress_id = node.get('inputs', {}).get('progress_id', '')
                return

    def wait(self, prompt_id, report=lambda message: None, stop=None, timeout=21600, on_progress=None):
        prompt_id = str(uuid.UUID(prompt_id))
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                raise RuntimeError('Monitoring stopped; use Resume Job to retrieve this job later')
            history = self.request('/history/'+prompt_id)
            if prompt_id in history:
                record = history[prompt_id]
                if not self.progress_id:
                    self.progress_from_graph(record.get('prompt', [None, None, {}])[2])
                packet = self.job_progress() if on_progress else None
                while packet:
                    on_progress(packet)
                    if packet.get('log_offset', 0) >= packet.get('log_size', 0) or not packet.get('log_chunk'):
                        break
                    packet = self.job_progress()
                state = record.get('status', {})
                if state.get('status_str') == 'error':
                    errors = [v for k, v in state.get('messages', []) if k in ('execution_error', 'execution_interrupted')]
                    detail = errors[-1].get('exception_message', 'Job interrupted') if errors else 'See server job log'
                    raise RuntimeError('ComfyUI execution failed: '+detail)
                outputs = record.get('outputs', {})
                bundles = [item for node in outputs.values() for item in node.get('mocap_bundle', [])]
                if bundles:
                    return bundles[0]
                if state.get('completed'):
                    raise RuntimeError('Job completed without a mocap bundle')
            queue = self.request('/queue')
            running = any(item[1] == prompt_id for item in queue.get('queue_running', []))
            queued = any(item[1] == prompt_id for item in queue.get('queue_pending', []))
            if not history and not running and not queued:
                raise RuntimeError('Job is absent from queue/history (server restarted, job removed, or wrong server)')
            if running:
                report('Running inference on ComfyUI')
            if running and on_progress:
                if not self.progress_id:
                    item = next(item for item in queue['queue_running'] if item[1] == prompt_id)
                    self.progress_from_graph(item[2])
                packet = self.job_progress()
                if packet:
                    on_progress(packet)
            elif queued:
                pending = queue.get('queue_pending', [])
                position = next(i for i, item in enumerate(pending) if item[1] == prompt_id)+1
                report(f'Queued on ComfyUI — position {position}')
            if stop is not None:
                stop.wait(2)
            else:
                time.sleep(2)
        raise TimeoutError('Monitoring timed out; job may still run. Use Resume Job.')

    def cancel(self, prompt_id):
        prompt_id = str(uuid.UUID(prompt_id))
        # This fork implements atomic, job-specific cancellation; never /interrupt globally.
        return self.request('/api/jobs/'+prompt_id+'/cancel', {})

    def download(self, output, destination, on_progress=None):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        query = urlencode({key: output[key] for key in ('filename', 'subfolder', 'type')})
        partial = destination.with_suffix('.part')
        digest = hashlib.sha256()
        try:
            with urlopen(Request(self.endpoint+'/view?'+query, headers=self.headers), timeout=120) as response, partial.open('wb') as target:
                size = 0
                last = 0.
                total = int(response.headers.get('Content-Length', '0')) or None
                for block in iter(lambda: response.read(1024*1024), b''):
                    size += len(block)
                    if size > 2*1024**3:
                        raise ValueError('Bundle exceeds the prototype 2 GiB download limit; use a shorter frame range')
                    digest.update(block)
                    target.write(block)
                    now = time.monotonic()
                    if on_progress and now-last > .5:
                        on_progress(dict(stage='Downloading result', done=size, total=total, unit='bytes'))
                        last = now
            if digest.hexdigest() != output['sha256']:
                raise ValueError('Downloaded bundle checksum does not match the server result')
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
        return str(destination)
