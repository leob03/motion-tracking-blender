"""Build explicitly selected artist and worker-source packages; no model/media walks."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'production_bridge/mocap_comfy/__init__.py'
info = next(ast.literal_eval(node.value) for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'bl_info' for t in node.targets))
version = '.'.join(map(str, info['version']))
out = ROOT/'dist'
out.mkdir(exist_ok=True)
artist = ['__init__.py', 'client.py', 'importer.py', 'rig.py', 'log_view.py', 'face.py']
snapshot = json.loads((ROOT/'provenance/source-snapshot.json').read_text())
worker = [row['path'] for row in snapshot['files'] if '/tests/' not in row['path'] and '/mocap_comfy/' not in row['path'] and not row['path'].endswith('package_addon.py')]


def archive(name, members):
    target = out/name
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, member in sorted(members):
            data = (ROOT/path).read_bytes()
            item = zipfile.ZipInfo(member, date_time=(2026, 10, 15, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            bundle.writestr(item, data)
    print(target)
    return hashlib.sha256(target.read_bytes()).hexdigest()+'  '+name


checksums = [archive('mocap_comfy-'+version+'.zip', [('production_bridge/mocap_comfy/'+f, 'mocap_comfy/'+f) for f in artist]),
             archive('mocap_worker_snapshot-'+version+'.zip', [(f, f) for f in worker])]
(out/'SHA256SUMS').write_text('\n'.join(checksums)+'\n')
