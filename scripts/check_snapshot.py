"""Check preserved source bytes and the explicit model manifest."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/'provenance/source-snapshot.json').read_text())
for row in manifest['files']:
    path = ROOT/row['path']
    assert path.stat().st_size == row['bytes'], row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['path']
    if path.suffix == '.py': ast.parse(path.read_text())
# This manifest describes the frozen baseline; update deliberately for future versions.
assets = json.loads((ROOT/'provenance/model-manifest.json').read_text())
assert len({row['id'] for row in assets['files']}) == len(assets['files'])
for row in assets['files']:
    assert len(row['sha256']) == 64 and row['bytes'] > 0
    path = Path(row['path'])
    assert not path.is_absolute() and '..' not in path.parts
print('Preserved source hashes and model manifest verified.')
