"""Graphical Blender test: real screen splits need redraws, not background mode."""
import json
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Quaternion

ROOT = Path(__file__).resolve().parents[1]
if '--installed' not in sys.argv:
    sys.path.insert(0, str(ROOT))
    import mocap_comfy
    mocap_comfy.register()
else:
    import mocap_comfy
from mocap_comfy.log_view import find_log_area, refresh_log_views

OUT = ROOT/'test_results/v031'
OUT.mkdir(parents=True, exist_ok=True)
state = {}


def next_step(fn):
    def checked():
        try:
            fn()
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            (OUT/'log_view_error.txt').write_text(error)
            bpy.ops.wm.quit_blender()
    bpy.app.timers.register(checked, first_interval=.6)


def open_log():
    c = bpy.context
    w = c.window_manager.windows[0]
    s = w.screen
    area = next(a for a in s.areas if a.type == 'VIEW_3D')
    space = area.spaces.active
    space.region_3d.view_rotation = Quaternion((1, 0, 0), .3)
    space.region_3d.view_distance = 7.5
    space.shading.type = 'SOLID'
    state.update(window=w, screen=s, view=area, before=len(s.areas), width=area.width,
                 rotation=tuple(space.region_3d.view_rotation), others={a.as_pointer(): (a.type,a.width,a.height) for a in s.areas if a != area})
    assert c.scene.mocap_comfy.max_frames == 0
    with c.temp_override(window=w,area=area):
        assert bpy.ops.mocap_comfy.live_log() == {'CANCELLED'}
        assert len(s.areas) == state['before']
        text = bpy.data.texts.new('Test Mocap job log')
        text.write('Preparing footage\nRunning GVHMR\nRunning SAM\n')
        c.scene.mocap_comfy.log_text = text.name
        assert bpy.ops.mocap_comfy.live_log() == {'FINISHED'}
    next_step(check_open)


def check_open():
    s, view = state['screen'], state['view']
    log = find_log_area(s)
    assert len(s.areas) == state['before']+1
    assert log and log.type == 'TEXT_EDITOR' and view.type == 'VIEW_3D'
    assert log.x > view.x and log.height == view.height
    assert tuple(view.spaces.active.region_3d.view_rotation) == state['rotation']
    assert view.spaces.active.region_3d.view_distance == 7.5
    assert view.spaces.active.shading.type == 'SOLID'
    state['log_pointer'] = log.as_pointer()
    with bpy.context.temp_override(window=state['window'],area=view):
        assert bpy.ops.mocap_comfy.live_log() == {'FINISHED'}
        assert len(s.areas) == state['before']+1
    # Persist the ownership flag and support re-enable without duplicating a pane.
    mocap_comfy.unregister()
    mocap_comfy.register()
    assert find_log_area(s).as_pointer() == state['log_pointer']
    text = bpy.data.texts.new('Next Mocap job log')
    text.write('Second job\nNew live log follows in the same pane.\n')
    bpy.context.scene.mocap_comfy.log_text = text.name
    refresh_log_views(bpy.context.scene, text)
    assert log.spaces.active.text == text
    assert len([h for h in bpy.app.handlers.load_pre if h.__module__ == 'mocap_comfy']) == 1
    next_step(close_header)


def close_header():
    log = find_log_area(state['screen'])
    with bpy.context.temp_override(window=state['window'],area=log,region=next(r for r in log.regions if r.type == 'HEADER')):
        assert bpy.ops.mocap_comfy.close_log() == {'FINISHED'}
    next_step(check_closed)


def check_closed():
    s, view = state['screen'], state['view']
    assert find_log_area(s) is None
    assert len(s.areas) == state['before']
    assert view.width == state['width']
    assert tuple(view.spaces.active.region_3d.view_rotation) == state['rotation']
    for area in s.areas:
        if area != view:
            assert (area.type,area.width,area.height) == state['others'][area.as_pointer()]
    with bpy.context.temp_override(window=state['window'],area=view):
        assert bpy.ops.mocap_comfy.live_log() == {'FINISHED'}
    next_step(close_sidebar)


def close_sidebar():
    with bpy.context.temp_override(window=state['window'],area=state['view']):
        assert bpy.ops.mocap_comfy.close_log() == {'FINISHED'}
    next_step(finish)


def finish():
    assert len(state['screen'].areas) == state['before']
    result = dict(blender=bpy.app.version_string, version=mocap_comfy.bl_info['version'],
                  addon=mocap_comfy.__file__, opens_right_of_viewport=True, repeated_open_reuses_pane=True,
                  close_from_header_and_sidebar=True, original_view_and_other_editors_preserved=True,
                  next_job_reuses_pane=True, disable_reenable_keeps_log_recognizable=True,
                  no_log_no_split=True, default_frame_count=0)
    (OUT/'log_view.json').write_text(json.dumps(result,indent=2))
    print('LOG_VIEW_VALIDATION',json.dumps(result),flush=True)
    bpy.ops.wm.quit_blender()

next_step(open_log)
