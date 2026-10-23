"""Run: blender -b IMPORTED.blend --python blender_plate_mapping.py -- OUT_DIR
Render frames through the plate compositor; each must match its own proxy frame best (not a neighbour)."""
import sys, json, bpy, numpy as np, glob, os
from pathlib import Path
out = Path(sys.argv[sys.argv.index('--')+1])
s = bpy.context.scene
col = next(c for c in bpy.data.collections if c.get('method'))
print('COMPOSITED', col.get('plate_composited'), 'film', s.render.film_transparent, s.view_settings.view_transform)
for o in bpy.data.objects:
    o.hide_render = True  # plate only
s.render.engine = 'BLENDER_WORKBENCH'; s.render.resolution_percentage = 12
clip = next(n for n in s.node_tree.nodes if n.type == 'MOVIECLIP').clip
proxy_dir = Path(bpy.path.abspath(clip.filepath)).parent
def load(path, size):
    img = bpy.data.images.load(str(path)); img.scale(*size); px = np.array(img.pixels[:]).reshape(size[1], size[0], 4)[..., :3]; bpy.data.images.remove(img); return px
results = {}
first, last = s.frame_start, s.frame_end
for f in (first, (first+last)//2, last):
    s.frame_set(f); s.render.filepath = str(out/f'plate_{f}.png'); bpy.ops.render.render(write_still=True)
    size = (s.render.resolution_x*s.render.resolution_percentage//100, s.render.resolution_y*s.render.resolution_percentage//100)
    rendered = load(out/f'plate_{f}.png', size)
    scores = {g: float(np.abs(rendered-load(proxy_dir/f'proxy.{g:04d}.jpg', size)).mean()) for g in (f-1, f, f+1) if (proxy_dir/f'proxy.{g:04d}.jpg').exists()}
    results[f] = dict(best=min(scores, key=scores.get), scores={k: round(v, 4) for k, v in scores.items()})
print('MAPPING', json.dumps(results))
assert all(r['best'] == f for f, r in results.items())
