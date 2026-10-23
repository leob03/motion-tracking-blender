"""Seeded multi-person box tracking for crowded plates.

Each seeded person keeps a centre with a damped velocity and a smoothed box size. Detections are
assigned per frame with the Hungarian method; a match must be near the predicted centre and of a
plausible size. Size never extrapolates, so a partial detection (an arm, a half-hidden body) cannot
collapse the prediction. While a person is unmatched, the search radius grows a little and the velocity
decays, so they are picked up again near where they disappeared; the boxes in a gap are interpolated
afterwards. A person whose predicted box has left the image is not searched for any more: in a crowd,
anyone appearing elsewhere would otherwise be taken for them.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _center_size(box):
    box = np.asarray(box, float)
    return (box[:2]+box[2:])/2, np.maximum(box[2:]-box[:2], 1.)


def _iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    inter = np.prod(np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])))
    return float(inter/max(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter, 1e-9))


def track(seeds, detections, frame_size=None, max_area_ratio=2.5, radius=.6, radius_growth=.05, max_radius=1.2, max_gap=24):
    """Boxes (frames, people, 4) and observed flags (frames, people) for people seeded on frame 0.

    frame_size (width, height) enables the left-the-image rule; max_gap stops searching after that many frames.
    """
    centers, sizes = zip(*(_center_size(box) for box in seeds)) if len(seeds) else ((), ())
    states = [dict(center=c, size=s, velocity=np.zeros(2), gap=0, gone=False) for c, s in zip(centers, sizes)]
    boxes, observed = [], []
    for candidates in detections:
        candidates = np.asarray(candidates, float).reshape(-1, 4)
        cand_center, cand_size = (candidates[:, :2]+candidates[:, 2:])/2, np.maximum(candidates[:, 2:]-candidates[:, :2], 1.)
        cost = np.full((len(states), len(candidates)), 1e6)
        for i, state in enumerate(states):
            if state['gone']:
                continue
            predicted = state['center']+state['velocity']
            box = np.concatenate([predicted-state['size']/2, predicted+state['size']/2])
            scale = max(state['size'])
            reach = min(radius+radius_growth*state['gap'], max_radius)
            for j in range(len(candidates)):
                distance = np.linalg.norm(cand_center[j]-predicted)/scale
                area = np.prod(cand_size[j])/np.prod(state['size'])
                if distance < reach and 1/max_area_ratio < area < max_area_ratio:
                    cost[i, j] = (1-_iou(box, candidates[j]))+.5*distance+.3*abs(np.log(area))
        matches = {i: j for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e5} if len(candidates) and states else {}
        frame_boxes, frame_seen = [], []
        for i, state in enumerate(states):
            if i in matches:
                center, size = cand_center[matches[i]], cand_size[matches[i]]
                step = (center-state['center'])/(state['gap']+1)
                state['velocity'] = .6*state['velocity']+.4*step
                state['center'], state['size'], state['gap'] = center, .7*state['size']+.3*size, 0
                frame_boxes.append(candidates[matches[i]].copy())
                frame_seen.append(True)
            else:
                state['center'] = state['center']+state['velocity']
                state['velocity'] = .7*state['velocity']
                state['gap'] += 1
                if state['gap'] > max_gap or (frame_size is not None and outside(state['center'], state['size'], frame_size)):
                    state['gone'] = True
                frame_boxes.append(np.concatenate([state['center']-state['size']/2, state['center']+state['size']/2]))
                frame_seen.append(False)
        boxes.append(frame_boxes)
        observed.append(frame_seen)
    boxes = np.array(boxes, float).reshape(len(detections), len(states), 4)
    observed = np.array(observed, bool).reshape(len(detections), len(states))
    for person in range(len(states)):
        seen = np.flatnonzero(observed[:, person])
        for start, end in zip(seen[:-1], seen[1:]):
            if end-start > 1:
                weights = np.linspace(0, 1, end-start+1)[1:-1, None]
                boxes[start+1:end, person] = (1-weights)*boxes[start, person]+weights*boxes[end, person]
    return boxes, observed


def outside(center, size, frame_size):
    """True when at least half of the box lies outside the image."""
    lo, hi = np.maximum(center-size/2, 0), np.minimum(center+size/2, frame_size)
    return np.prod(np.maximum(hi-lo, 0)) < .5*np.prod(size)


def visible_span(observed, seed, max_gap=12):
    """(first, last) frame indices around seed where the person is followed with gaps of at most max_gap."""
    seen = np.flatnonzero(observed)
    if not len(seen):
        return seed, seed
    keep = [seed]
    for direction in (1, -1):
        last = seed
        for index in seen[::direction]:
            if (index-seed)*direction <= 0:
                continue
            if abs(index-last)-1 > max_gap:
                break
            last = index
        keep.append(last)
    return int(min(keep)), int(max(keep))


def gaps(observed):
    """(first, last) frame index pairs of each run without a matching detection."""
    runs, start = [], None
    for index, seen in enumerate(list(observed)+[True]):
        if not seen and start is None:
            start = index
        elif seen and start is not None:
            runs.append((start, index-1))
            start = None
    return runs
