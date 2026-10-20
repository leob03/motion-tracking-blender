"""Package only artist-side source, without model code or server paths."""
from pathlib import Path
import zipfile
ROOT = Path(__file__).resolve().parent
output = ROOT/'dist/mocap_comfy-0.5.0.zip'
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
    for filename in ('__init__.py', 'client.py', 'importer.py', 'rig.py', 'log_view.py', 'face.py', 'face_controls.py'):
        bundle.write(ROOT/'mocap_comfy'/filename, 'mocap_comfy/'+filename)
print(output)
