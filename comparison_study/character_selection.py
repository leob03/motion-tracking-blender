"""Stable seeded track IDs and a selection manifest shared by both body backends.
Test footage: Karoshi090. Association: Hungarian IoU + predicted center distance.
Short gaps retain IDs; long/crossing occlusions still require an identity review.
"""
import json,hashlib,argparse,os
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
R=Path(__file__).resolve().parent

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter;return float(inter/max(union,1e-9))

def track(seeds,detections,max_gap=12):
    states=[dict(box=np.array(b,dtype=float),velocity=np.zeros(4),gap=0) for b in seeds];boxes=[];observed=[]
    for frame,candidates in enumerate(detections):
        det=np.asarray(candidates,dtype=float).reshape(-1,4);pred=np.array([s['box']+s['velocity'] for s in states]);cost=np.full((len(states),len(det)),1e6)
        for i,s in enumerate(states):
            if s['gap']>max_gap:continue
            for j,b in enumerate(det):
                distance=np.linalg.norm((pred[i,:2]+pred[i,2:]-b[:2]-b[2:])/2)/max(np.linalg.norm(pred[i,2:]-pred[i,:2]),1)
                overlap=iou(pred[i],b)
                if overlap>.05 and distance<.65:cost[i,j]=1-overlap+.2*distance
        matches={i:j for i,j in zip(*linear_sum_assignment(cost)) if cost[i,j]<1e5} if len(det) else {}
        flags=[]
        for i,s in enumerate(states):
            if i in matches:
                b=det[matches[i]];s['velocity']=.6*s['velocity']+.4*(b-s['box']);s['box']=b;s['gap']=0;flags.append(True)
            else:s['box']=pred[i];s['gap']+=1;flags.append(False)
        boxes.append([s['box'].copy() for s in states]);observed.append(flags)
    return np.array(boxes),np.array(observed)

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2));os.replace(temp,path)

def prepare():
    m=json.loads((R/'inputs/karoshi090/clip.json').read_text());ref=json.loads((R/'inputs/karoshi090_dual/clip.json').read_text());known=np.array(ref['boxes_xyxy_tracks']);detections=[d['candidates'] for d in m['tracking_decisions']];boxes,obs=track(known[0],detections);reverse,revobs=track(known[0],[d[::-1] for d in detections]);assert np.array_equal(obs,revobs) and np.allclose(boxes,reverse)
    agreement=np.array([[iou(a,b) for a,b in zip(x,y)] for x,y in zip(boxes,known)]);assert agreement.min()>.95 and obs.all(),(agreement.min(),obs.sum())
    gaps=[list(d) for d in detections]
    for f in range(20,25):gaps[f]=[d for d in gaps[f] if iou(d,known[f,0])<.8]
    gapbox,gapobs=track(known[0],gaps);assert not gapobs[20:25,0].any() and gapobs[25:,0].all();assert min(iou(a,b) for a,b in zip(gapbox[25:,0],known[25:,0]))>.95
    rev=hashlib.sha256(boxes.astype('<f4').tobytes()).hexdigest()[:16]
    registry=dict(clip='karoshi090',revision=rev,width=m['width'],height=m['height'],fps=m['fps'],frames=m['frames'],video='inputs/karoshi090/input.mp4',poster='inputs/karoshi090_selection/preview.jpg',tracks=[dict(id='person_001',label='Blue shirt',color='#57b9ff',cache_index=0,gvhmr_clip='karoshi090'),dict(id='person_002',label='Dark jacket',color='#ffb45b',cache_index=1,gvhmr_clip='karoshi090_dark')],boxes_xyxy=boxes.tolist(),observed=obs.tolist(),tracking='Two first-frame person seeds; Hungarian association using IoU and velocity. IDs are never assigned by per-frame confidence order.',limitations='This two-person shot is validated. Re-entry after long occlusion, crossings, camera cuts and changing costumes need a stronger tracker or manual correction. Mask initialization is not implemented; use click selection of tracked boxes.')
    out=R/'inputs/karoshi090_selection';atomic_json(out/'tracks.json',registry)
    import cv2
    cv2.setNumThreads(2);cap=cv2.VideoCapture(str(R/'inputs/karoshi090/input.mp4'));ok,im=cap.read();cap.release();assert ok;cv2.imwrite(str(out/'preview.jpg'),im)
    validation=dict(frames=len(boxes),tracks=2,minimum_iou_against_reviewed_tracks=float(agreement.min()),reversed_detection_order_identical=True,five_frame_gap_recovered_same_id=True,no_inferred_boxes_in_original_shot=bool(obs.all()),scope='Seeded two-actor Karoshi090, not a general crowd/re-identification benchmark.')
    atomic_json(out/'tracking_validation.json',validation);print(validation)

def select(payload,output_root=None):
    registry=json.loads((R/'inputs/karoshi090_selection/tracks.json').read_text());ids=payload.get('selected_ids');tracks=registry['tracks'];valid={t['id'] for t in tracks}
    if payload.get('revision')!=registry['revision']:raise ValueError('Tracking data changed; reload the selector.')
    if not isinstance(ids,list) or not ids or any(not isinstance(x,str) for x in ids) or len(ids)!=len(set(ids)) or not set(ids)<=valid:raise ValueError('Select one or both available track IDs.')
    selected=[t for t in tracks if t['id'] in ids];root=Path(output_root) if output_root else R/'results/karoshi090_selection';root.mkdir(parents=True,exist_ok=True)
    indices=[t['cache_index'] for t in selected];boxes=np.asarray(registry['boxes_xyxy'])[:,indices]
    manifest=dict(clip='karoshi090',revision=registry['revision'],selected_ids=[t['id'] for t in selected],selected_count=len(selected),frames=registry['frames'],fps=registry['fps'],coordinate_space='Independent camera-space predictions; no world placement changes.',inference_status='Existing validated per-person predictions filtered by persistent ID; no new inference.',source_video='inputs/karoshi090/input.mp4')
    # Every backend consumes the same selected IDs and boxes, with stable slot order.
    plan=dict(manifest,boxes_xyxy_tracks=boxes.tolist(),backends=dict(sam=dict(mode='multi-person boxes',track_ids=manifest['selected_ids']),gvhmr=dict(mode='one temporal sequence per track ID',tracks=[dict(id=t['id'],boxes_xyxy=boxes[:,i].tolist()) for i,t in enumerate(selected)])))
    for backend in ['sam_smooth','gvhmr']:
        if backend=='sam_smooth':
            with np.load(R/'results/karoshi090_dual/sam/smooth.npz') as d:data={k:(d[k][:,indices] if k in ['vertices_camera','joints_camera'] else d[k]) for k in d.files}
        else:
            arrays=[]
            for t in selected:
                with np.load(R/'results'/t['gvhmr_clip']/'gvhmr/native.npz') as d:arrays.append({k:d[k] for k in ['vertices_camera','joints_camera','faces','fps','K']})
            data={k:np.stack([d[k] for d in arrays],axis=1) if k in ['vertices_camera','joints_camera'] else arrays[0][k] for k in arrays[0]}
        data['track_ids']=np.array(manifest['selected_ids']);tmp=root/f'{backend}.tmp.npz';np.savez_compressed(tmp,**data);os.replace(tmp,root/f'{backend}.npz')
    atomic_json(root/'processing_plan.json',plan);atomic_json(root/'selection.json',manifest)
    return dict(**manifest,downloads=[f'results/karoshi090_selection/{x}' for x in ['selection.json','processing_plan.json','sam_smooth.npz','gvhmr.npz']])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--ids',nargs='+');a=p.parse_args()
    if a.prepare:prepare()
    if a.ids:
        reg=json.loads((R/'inputs/karoshi090_selection/tracks.json').read_text());print(select(dict(selected_ids=a.ids,revision=reg['revision'])))
