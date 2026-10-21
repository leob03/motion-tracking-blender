"""Export an imported mocap result for Nuke: node script, camera .chan, and an Alembic of body + camera.

Conventions follow the ViPE Remote Tracking exporter (docs/NUKE.md there): Blender (x, y, z) ->
Nuke (x, z, -y), Camera2 rot_order ZXY, filmback from Blender's sensor fit, win_translate from shift.
"""
from functools import partial
import math
from pathlib import Path
import re

import bpy
from mathutils import Matrix

TO_NUKE = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))


def _num(value):
    return f'{value:.10g}'


def _quote(text):
    for old, new in (('\\', '\\\\'), ('"', '\\"'), ('[', '\\['), (']', '\\]'), ('$', '\\$'), ('{', '\\{'), ('}', '\\}'), ('\n', '\\n')):
        text = text.replace(old, new)
    return f'"{text}"'


def _path(path):
    return _quote(str(path).replace('\\', '/'))


def _curve(values, start):
    return '{curve x%d %s}' % (start, ' '.join(map(_num, values)))


def _value(values, start):
    if max(values)-min(values) <= 1e-12*max(1.0, abs(values[0])):
        return _num(values[0])
    return _curve(values, start)


def _node(kind, *knobs):
    return '\n'.join([kind+' {']+[f' {k} {v}' for k, v in knobs if v is not None]+['}'])


def _filmback(data, width, height):
    """Horizontal aperture (mm) and the sensor size Blender's shift is measured in."""
    fit = data.sensor_fit if data.sensor_fit != 'AUTO' else ('HORIZONTAL' if width >= height else 'VERTICAL')
    if fit == 'HORIZONTAL':
        return data.sensor_width, data.sensor_width
    size = data.sensor_width if data.sensor_fit == 'AUTO' else data.sensor_height
    return size*width/height, size


def _camera_track(context, camera, frames, width, height, warnings):
    scene, rows, previous, notes = context.scene, [], None, set()
    for frame in frames:
        scene.frame_set(frame)
        evaluated = camera.evaluated_get(context.evaluated_depsgraph_get())
        matrix = TO_NUKE@evaluated.matrix_world
        scale = matrix.to_scale()
        rotation = matrix.to_3x3().normalized()
        if max(scale)-min(scale) > 1e-6*max(scale):
            notes.add('The camera has non-uniform scale; it was removed and the rotation may be approximate.')
        euler = rotation.to_euler('ZXY', previous) if previous is not None else rotation.to_euler('ZXY')
        previous = euler
        data = evaluated.data
        aperture, fit = _filmback(data, width, height)
        rows.append(dict(frame=frame, translate=tuple(matrix.to_translation()), rotate=tuple(math.degrees(a) for a in euler),
                         focal=data.lens, haperture=aperture, vaperture=aperture*height/width,
                         win=(2*data.shift_x*fit/aperture, 2*data.shift_y*fit/aperture), near=data.clip_start, far=data.clip_end))
    warnings.extend(sorted(notes))
    return rows


def _layer_path(layer_collection, collection):
    if layer_collection.collection == collection:
        return [layer_collection]
    for child in layer_collection.children:
        path = _layer_path(child, collection)
        if path:
            return [layer_collection]+path
    return []


def _alembic(context, path, objects, active, start, end, width, height, warnings):
    """Selected-only, world-flattened Alembic of the evaluated (skinned) body and the camera."""
    scene, layer = context.scene, context.view_layer
    selected, previous_active, undo = [o for o in layer.objects if o.select_get()], layer.objects.active, []
    render = scene.render
    try:
        for obj in objects:
            for collection in obj.users_collection:
                for item in _layer_path(layer.layer_collection, collection):
                    if item.hide_viewport:
                        undo.append(partial(setattr, item, 'hide_viewport', True))
                        item.hide_viewport = False
            if obj.hide_get(view_layer=layer):
                undo.append(partial(obj.hide_set, True, view_layer=layer))
                obj.hide_set(False, view_layer=layer)
        for key, value in (('resolution_x', width), ('resolution_y', height), ('resolution_percentage', 100), ('pixel_aspect_x', 1.0), ('pixel_aspect_y', 1.0)):
            undo.append(partial(setattr, render, key, getattr(render, key)))
            setattr(render, key, value)
        layer.update()
        objects = [o for o in objects if o.visible_get(view_layer=layer)]
        for obj in selected:
            obj.select_set(False)
        for obj in objects:
            obj.select_set(True)
        layer.objects.active = active
        allowed = bpy.ops.wm.alembic_export.get_rna_type().properties.keys()
        options = dict(filepath=str(path), start=start, end=end, selected=True, flatten=True, uvs=False, orcos=False, normals=True,
                       export_custom_properties=False, evaluation_mode='VIEWPORT', as_background_job=False, init_scene_frame_range=False)
        with context.temp_override(selected_objects=objects, selected_editable_objects=objects, active_object=active, object=active):
            result = bpy.ops.wm.alembic_export(**{k: v for k, v in options.items() if k in allowed})
        if 'FINISHED' not in result or not Path(path).is_file():
            raise RuntimeError(f'exporter returned {sorted(result)}')
        return True
    except Exception as error:
        warnings.append(f'Alembic export failed ({error}); the .nk and .chan files are unaffected.')
        return False
    finally:
        for obj in layer.objects:
            if obj.select_get():
                obj.select_set(False)
        for restore in reversed(undo):
            restore()
        for obj in selected:
            if obj.name in layer.objects:
                obj.select_set(True)
        layer.objects.active = previous_active


def _frame_file(pattern, frame):
    pattern = re.sub(r'#+', lambda m: f'{frame:0{len(m.group())}d}', pattern)
    return re.sub(r'%0?(\d*)d', lambda m: f'{frame:0{m.group(1) or 1}d}', pattern)


def _plate(collection, start, count, fmt, warnings):
    pattern, movie = collection.get('plate_pattern'), collection.get('plate_movie')
    if pattern:
        first = int(collection.get('plate_first_frame', start))
        last = first+count-1
        knobs = [('inputs', 0), ('file', _path(pattern)), ('format', fmt), ('first', first), ('last', last),
                 ('origfirst', first), ('origlast', last), ('origset', 'true')]
        if first != start:
            knobs += [('frame_mode', 'offset'), ('frame', start-first)]
        check = _frame_file(pattern, first)
    elif movie:
        # Movie frame 1 is source frame 0; place the processed range at the job's start frame.
        first_source = int(collection.get('source_start_frame', 0))
        last = first_source+count
        knobs = [('inputs', 0), ('file', _path(movie)), ('format', fmt), ('first', 1), ('last', last), ('origfirst', 1), ('origlast', last),
                 ('origset', 'true'), ('frame_mode', '"start at"'), ('frame', start-first_source)]
        check = movie
    else:
        warnings.append('No plate path is stored on this result; a black Constant stands in for the plate.')
        return _node('Constant', ('inputs', 0), ('channels', 'rgb'), ('format', fmt), ('name', 'Mocap_Plate_missing'), ('xpos', 0), ('ypos', -300))
    if not Path(check).is_file():
        warnings.append(f'The plate file was not found on this machine: {check}')
    return _node('Read', *knobs, ('name', 'Mocap_Plate'), ('xpos', 0), ('ypos', -300))


def export_nuke(context, collection, directory):
    """Write the Nuke package for a mocap result collection; returns {'nk', 'files', 'warnings'}."""
    if collection is None or 'bundle' not in collection:
        raise ValueError('Select an object of an imported mocap result first.')
    if not str(directory or '').strip():
        raise ValueError('Choose an output folder for the Nuke export.')
    scene = context.scene
    camera = bpy.data.objects.get(collection.get('scene_camera', ''))
    if camera is None:
        cameras = [o for o in collection.all_objects if o.type == 'CAMERA']
        camera = scene.camera if scene.camera in cameras else next((o for o in cameras if o.get('mocap_camera_role') == 'VIDEO'), cameras[0] if cameras else None)
    if camera is None:
        raise ValueError('The result has no camera.')
    body = next((o for o in collection.all_objects if o.type == 'MESH'), None)
    if body is None:
        raise ValueError('The result has no body mesh; re-import it with "Animated body mesh" enabled.')
    start = int(collection.get('start_frame', scene.frame_start))
    count = int(collection.get('frames', scene.frame_end-start+1))
    width, height = scene.render.resolution_x, scene.render.resolution_y
    folder = Path(bpy.path.abspath(str(directory))).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    label = re.sub(r'[^A-Za-z0-9._-]+', '_', f"{collection.name.split('|')[1].strip() if '|' in collection.name else 'mocap'}_{collection.get('track_id', '')}").strip('_.')
    warnings = []
    old_frame = scene.frame_current
    try:
        rows = _camera_track(context, camera, range(start, start+count), width, height, warnings)
    finally:
        scene.frame_set(old_frame)
    fmt = f'"{width} {height} 0 0 {width} {height} 1 "'
    abc = folder/f'{label}_body.abc'
    have_abc = _alembic(context, abc, [body, camera], body, start, start+count-1, width, height, warnings)
    win = [r['win'] for r in rows]
    has_win = max(abs(c) for w in win for c in w) > 1e-9
    info = (f"Mocap {collection.get('method', '')} · {collection.get('track_id', '')}\nFrames {start}-{start+count-1}, {width}x{height}\n"
            f"Camera {camera.name}: focal {_num(rows[0]['focal'])} mm, filmback {_num(rows[0]['haperture'])} x {_num(rows[0]['vaperture'])} mm")
    nodes = ['set cut_paste_input [stack 0]',
             _node('StickyNote', ('inputs', 0), ('name', 'Mocap_Info'), ('label', _quote(info)), ('note_font_size', 16), ('xpos', -520), ('ypos', -320)),
             _plate(collection, start, count, fmt, warnings), 'set NMocapPlate [stack 0]',
             _node('Camera2', ('inputs', 0),
                   ('translate', '{%s}' % ' '.join(_curve([r['translate'][i] for r in rows], start) for i in range(3))),
                   ('rotate', '{%s}' % ' '.join(_curve([r['rotate'][i] for r in rows], start) for i in range(3))),
                   ('rot_order', 'ZXY'), ('focal', _value([r['focal'] for r in rows], start)),
                   ('haperture', _num(rows[0]['haperture'])), ('vaperture', _num(rows[0]['vaperture'])),
                   ('near', _num(rows[0]['near'])), ('far', _num(rows[0]['far'])),
                   ('win_translate', '{%s %s}' % (_value([w[0] for w in win], start), _value([w[1] for w in win], start)) if has_win else None),
                   ('name', 'Mocap_Camera'), ('xpos', -320), ('ypos', -60)),
             'set NMocapCamera [stack 0]']
    if have_abc:
        nodes += [_node('ReadGeo2', ('inputs', 0), ('file', _path(abc)), ('read_on_each_frame', 'true'), ('name', 'Mocap_Body'), ('xpos', -160), ('ypos', -160)),
                  _node('Scene', ('inputs', 1), ('name', 'Mocap_Scene'), ('xpos', -150), ('ypos', -60))]
    else:
        nodes.append(_node('Scene', ('inputs', 0), ('name', 'Mocap_Scene'), ('xpos', -150), ('ypos', -60)))
    nodes += ['set NMocapScene [stack 0]', 'push $NMocapCamera', 'push $NMocapScene', 'push $NMocapPlate',
              _node('ScanlineRender', ('inputs', 3), ('name', 'Mocap_Render'), ('label', _quote('bg: plate, obj/scn: body, cam: Mocap_Camera')), ('xpos', 0), ('ypos', 0))]
    nk = folder/f'{label}_mocap.nk'
    nk.write_text('\n'.join(nodes)+'\n')
    chan = folder/f'{label}_camera.chan'
    chan.write_text(''.join('%d %s\n' % (r['frame'], ' '.join(map(_num, (*r['translate'], *r['rotate'], math.degrees(2*math.atan(r['vaperture']/(2*r['focal']))))))) for r in rows))
    readme = folder/'README.txt'
    readme.write_text('\n'.join([
        f"Mocap export for Nuke: {collection.name}, frames {start}-{start+count-1}, {width}x{height}.",
        f"{nk.name}: File > Import Script. Plate Read, Mocap_Camera (Camera2, rot_order ZXY), Mocap_Body (ReadGeo2 of the Alembic), ScanlineRender check.",
        f"{chan.name}: frame tx ty tz rx ry rz vfov; import onto a Camera with rot_order ZXY, haperture {_num(rows[0]['haperture'])} mm, vaperture {_num(rows[0]['vaperture'])} mm.",
        f"{abc.name}: animated body mesh (skinned, world space, metres) and the camera, Y-up." + ('' if have_abc else ' NOT WRITTEN - see warnings.'),
        'ReadGeo2/ScanlineRender wiring was generated without Nuke; check input order if the render is empty.',
        'Paths are absolute; repath the Read nodes if this folder moves.'])+'\n')
    files = [str(nk), str(chan)]+([str(abc)] if have_abc else [])+[str(readme)]
    return {'nk': str(nk), 'files': files, 'warnings': warnings}
