"""Run: blender -b --factory-startup --python blender_relative_plate_path.py -- PLATE SAVE_AS.blend (or - to keep the opened .blend unsaved)
Pick a plate the way Blender's file browser does (relative to the .blend) and probe it."""
import sys, json, bpy
args = sys.argv[sys.argv.index('--')+1:]
plate, save_as = args[0], args[1]
import addon_utils; addon_utils.enable('mocap_comfy', default_set=True)
if save_as != '-':
    bpy.ops.wm.save_as_mainfile(filepath=save_as)
relative = bpy.path.relpath(plate)
s = bpy.context.scene.mocap_comfy
s.video = relative
info = json.loads(s.plate_info)
print('RESULT', json.dumps(dict(blend=bpy.data.filepath, picked=relative, stored=s.video, info={k: info.get(k) for k in ('error', 'sequence', 'width', 'height', 'first', 'last', 'pattern', 'declared_colorspace')}, colour=s.plate_colorspace)))
