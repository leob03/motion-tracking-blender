"""Verify only manifest-listed files; never traverse model storage."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--study-root', type=Path, required=True)
parser.add_argument('--nightly-models', type=Path, required=True)
parser.add_argument('--gvhmr-node', type=Path, required=True)
parser.add_argument('--mode', choices=['gvhmr', 'hybrid', 'face'], default='face')
parser.add_argument('--sha256', action='store_true', help='Sequentially read/hash each selected file; otherwise check existence and size')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'provenance/model-manifest.json').read_text())
roots = dict(study=args.study_root, nightly_models=args.nightly_models, gvhmr_node=args.gvhmr_node)
features = {'all'} | ({'hybrid'} if args.mode != 'gvhmr' else set()) | ({'realdenseface_flame'} if args.mode == 'face' else set())
failed = []
count = 0
for item in manifest['files']:
    if not features.intersection(item['features']):
        continue
    count += 1
    path = roots[item['root']]/item['path']
    if not path.is_file():
        failed.append(item['id']+': missing '+str(path))
    elif path.stat().st_size != item['bytes']:
        failed.append(item['id']+': byte count differs')
    elif args.sha256:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            failed.append(item['id']+': SHA256 differs')
if failed:
    raise SystemExit('\n'.join(failed))
print(f'Verified {count} selected assets: '+('size and SHA256' if args.sha256 else 'existence and size only'))
