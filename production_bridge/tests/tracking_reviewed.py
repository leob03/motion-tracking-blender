"""Run with the worker Python: tracking_reviewed.py STUDY_INPUTS_DIR
Seeded tracker against the study's reviewed two-person Karoshi090 tracks (same criteria as character_selection)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tracking
S = Path(sys.argv[1])
m = json.loads((S/'karoshi090/clip.json').read_text()); ref = json.loads((S/'karoshi090_dual/clip.json').read_text())
known = np.array(ref['boxes_xyxy_tracks']); detections = [d['candidates'] for d in m['tracking_decisions']]
size = (m['width'], m['height'])
boxes, observed = tracking.track(known[0], detections, size)
agreement = np.array([[tracking._iou(a, b) for a, b in zip(x, y)] for x, y in zip(boxes, known)])
reverse, reverse_observed = tracking.track(known[0], [d[::-1] for d in detections], size)
gapped = [list(d) for d in detections]
for f in range(20, 25):
    gapped[f] = [d for d in gapped[f] if tracking._iou(d, known[f, 0]) < .8]
gap_boxes, gap_observed = tracking.track(known[0], gapped, size)
result = dict(min_iou=float(agreement.min()), all_observed=bool(observed.all()),
              reversed_identical=bool(np.array_equal(observed, reverse_observed) and np.allclose(boxes, reverse)),
              gap_recovered=bool(not gap_observed[20:25, 0].any() and gap_observed[25:, 0].all()),
              gap_interpolated_min_iou=float(min(tracking._iou(a, b) for a, b in zip(gap_boxes[20:25, 0], known[20:25, 0]))),
              visible_span=tracking.visible_span(gap_observed[:, 0], 0))
print(json.dumps(result))
assert result['min_iou'] > .95 and result['all_observed'] and result['reversed_identical'] and result['gap_recovered']
assert result['gap_interpolated_min_iou'] > .9 and result['visible_span'] == (0, len(detections)-1)
