import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mocap_comfy.importer import import_bundle
fixture=ROOT/'test_results/synthetic/result.mocap.zip'
with zipfile.ZipFile(fixture) as z:
    manifest=json.loads(z.read('manifest.json'))
    animation=z.read('animation.npz')
with tempfile.TemporaryDirectory(prefix='mocap-invalid-test-') as scratch:
    before=set(bpy.data.objects)
    bad=Path(scratch)/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:
        z.writestr('manifest.json',json.dumps(dict(manifest,version=99)))
        z.writestr('animation.npz',animation)
    try:
        import_bundle(bad)
    except ValueError as exc:
        assert 'schema' in str(exc)
    else:
        raise AssertionError('Unknown schema accepted')
    with np.load(io.BytesIO(animation),allow_pickle=False) as src:
        arrays={k:src[k] for k in src.files}
    arrays['faces']=np.array([[0,1,999999]])
    stream=io.BytesIO();np.savez_compressed(stream,**arrays)
    with zipfile.ZipFile(bad,'w') as z:
        z.writestr('manifest.json',json.dumps(manifest))
        z.writestr('animation.npz',stream.getvalue())
    try:
        import_bundle(bad)
    except ValueError as exc:
        assert 'triangles' in str(exc)
    else:
        raise AssertionError('Invalid topology accepted')
    assert set(bpy.data.objects)==before
    with np.load(io.BytesIO(animation),allow_pickle=False) as src:
        arrays={k:src[k] for k in src.files}
    n=manifest['frames']
    arrays.update(rig_names=np.array(['root','a','b']),rig_parents=np.array([-1,2,1]),
                  rig_rest=np.tile(np.eye(4),(3,1,1)),rig_pose=np.tile(np.eye(4),(n,3,1,1)),
                  rig_indices=np.zeros((3,1),np.int32),rig_weights=np.ones((3,1)),rig_vertices=arrays['vertices_world'])
    for field, reason in [('rig_parents','hierarchy'),('rig_weights','weights')]:
        if field == 'rig_weights':
            arrays['rig_parents']=np.array([-1,0,1]);arrays['rig_weights'][0,0]=2.
        stream=io.BytesIO();np.savez_compressed(stream,**arrays)
        with zipfile.ZipFile(bad,'w') as z:
            z.writestr('manifest.json',json.dumps(manifest));z.writestr('animation.npz',stream.getvalue())
        try: import_bundle(bad)
        except ValueError as exc: assert reason in str(exc),str(exc)
        else: raise AssertionError('Invalid rig accepted: '+field)
        assert set(bpy.data.objects)==before
(ROOT/'test_results/blender_invalid.json').write_text(json.dumps(dict(unknown_schema_rejected=True,invalid_topology_rejected=True,invalid_rig_hierarchy_rejected=True,invalid_skin_weights_rejected=True,no_scene_changes_on_validation_failure=True),indent=2))
print('Invalid-bundle checks passed')
