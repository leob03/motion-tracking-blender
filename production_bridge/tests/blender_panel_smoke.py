"""Call the mocap panels' draw() with a stand-in layout for each camera mode / method / lens case (no GUI needed).

blender -b --factory-startup --python-exit-code 1 --python production_bridge/tests/blender_panel_smoke.py
"""
import json
import sys
from types import SimpleNamespace

import bpy

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import mocap_comfy as mc  # noqa: E402
mc.register()
mc.wrap_width = lambda context: 40
calls = []


class Fake:
    alert = active = enabled = False

    def __getattr__(self, name):
        def call(*args, **kwargs):
            calls.append((name, args[1] if name == 'prop' and len(args) > 1 else kwargs.get('text', '')))
            return Fake()
        return call


settings = bpy.context.scene.mocap_comfy
plates = {
    'movie': dict(width=1920, height=1080, frames=100, fps=24.),
    'exr_lens': dict(sequence=True, width=4608, height=3164, frames=116, fps=24., first=977, last=1092, pattern='a.####.exr', gaps=0,
                     pixel_aspect=1., lens=dict(focal_mm=32.77, sensor_width_mm=27.99, focal_px=5394.2, plate_width=4608, camera='ARRI ALEXA 35',
                                                source='EXR header: FocalLength 32.77 mm', warning='')),
    'exr_focal_only': dict(sequence=True, width=2048, height=1080, frames=50, fps=24., first=1, last=50, pattern='b.####.exr', gaps=0,
                           pixel_aspect=1., lens=dict(focal_mm=50., sensor_width_mm=None, focal_px=None, plate_width=2048, camera='', source='x', warning='Filmback unknown')),
}
failures = []
for plate_name, info in plates.items():
    for method in ('sam_camera', 'hybrid', 'gvhmr'):
        for mode in ('SCENE', 'LOCKED', 'ESTIMATE', 'FOCAL'):
            settings.plate_info = json.dumps(info)
            settings.method = method
            settings.camera_mode = mode
            for panel in (mc.MOCAP_PT_camera, mc.MOCAP_PT_process, mc.MOCAP_PT_import):
                try:
                    panel.draw(SimpleNamespace(layout=Fake()), bpy.context)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f'{plate_name}/{method}/{mode}/{panel.__name__}: {exc!r}')
            if mode == 'LOCKED':
                lens = mc.locked_lens(settings)
                print('LENS', plate_name, method, None if lens is None else (round(lens['ratio'], 4), lens['source'][:60]))
            issue = mc.method_camera_issue(settings)
            if issue and not (method == 'sam_camera' and mode in ('ESTIMATE', 'FOCAL')):
                failures.append(f'unexpected issue {plate_name}/{method}/{mode}')
# Changing to sam_camera while on an estimate mode switches to locked-off.
settings.camera_mode, settings.method = 'ESTIMATE', 'gvhmr'
settings.method = 'sam_camera'
print('AUTO_SWITCH', settings.camera_mode)
settings.locked_focal_mm, settings.sensor_mm = 35., 36.
settings.plate_info = json.dumps(plates['movie'])
print('LENS typed', mc.locked_lens(settings))
print('PANEL_SMOKE', 'FAIL' if failures else 'OK', failures[:5], len(calls))
if failures:
    raise SystemExit(1)
