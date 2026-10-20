"""Re-export retained FLAME fits without rerunning any inference (worker only)."""
import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'production_bridge'))
from flame_export import add_flame_controls
from rig_export import attach_flame_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--rebuild', action='store_true', help='Rebuild an existing editable export from its retained fit and native MHR body rig')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output path; the original capture is preserved')
    with zipfile.ZipFile(args.input) as z:
        if sum(i.file_size for i in z.infolist()) > 2*1024**3:
            raise ValueError('Bundle too large')
        members = {name: z.read(name) for name in z.namelist()}
    manifest = json.loads(members['manifest.json'])
    if manifest.get('face_capture') != 'realdenseface_flame':
        raise ValueError('Only retained RealDenseFace + FLAME results can be upgraded')
    def npz(raw):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if sum(i.file_size for i in z.infolist()) > 2*1024**3:
                raise ValueError('Array archive too large')
        with np.load(io.BytesIO(raw), allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    arrays = npz(members['animation.npz'])
    rig = {k: v for k, v in arrays.items() if k.startswith('rig_')}
    if manifest.get('face_editing'):
        if not args.rebuild:
            raise ValueError('Already editable; use --rebuild to update its deformation export')
        offset = int(arrays['flame_vertex_offset'])
        original_bones = rig['rig_names'].tolist().index('FLAME_root')
        for k in ('rig_names', 'rig_parents', 'rig_rest'):
            rig[k] = rig[k][:original_bones]
        rig['rig_pose'] = rig['rig_pose'][:, :original_bones]
        for k in ('rig_indices', 'rig_weights'):
            rig[k] = rig[k][:offset]
        fit = arrays['actor_camera_to_world']
        camera_vertices = np.einsum('fji,fvj->fvi', fit[:, :3, :3], arrays['vertices_world']-fit[:, None, :3, 3])
        attach_flame_weights(rig, camera_vertices[0, :offset], camera_vertices[0])
        arrays = {k: v for k, v in arrays.items() if not k.startswith('flame_')}
    report = add_flame_controls(arrays, rig, npz(members['face_parameters.npz']), args.study)
    arrays.update(rig)
    buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays)
    members['animation.npz'] = buffer.getvalue()
    manifest['face_editing'] = report
    manifest['version'] = 2
    manifest['rig']['bones'] = len(rig['rig_names'])
    manifest['body_note'] = 'Native body/finger rig, FLAME expression keys and jaw/eye bones with retained surface residuals.'
    members['manifest.json'] = json.dumps(manifest, indent=2).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, content in members.items():
            z.writestr(name, content)
    print(json.dumps(report, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
