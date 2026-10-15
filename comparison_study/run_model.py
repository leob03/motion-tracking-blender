"""Run the installed GVHMR port or pinned ComfyUI SAM port on shared inputs.
All generated files stay within this study. No remote inference is used.
"""
import argparse
import importlib
import json
import pickle
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INSTALLED = Path('/home/innovation/dev-lbringer/ComfyUI_nightly')
p = argparse.ArgumentParser()
p.add_argument('method', choices=['gvhmr','sam'])
p.add_argument('clip')
p.add_argument('--face',action='store_true')
p.add_argument('--multi',action='store_true')
p.add_argument('--force-rerun',action='store_true')
p.add_argument('--work-root',type=Path,help='Isolated inputs/results directory for a ComfyUI job')
p.add_argument('--sam-smoothing',type=float,default=1.)
p.add_argument('--no-hand-refinement',action='store_true')
a = p.parse_args()
ASSETS = ROOT
sys.path.insert(0, str(ASSETS.parent/'production_bridge'))
from progress import report
if a.work_root:
    ROOT = a.work_root.resolve()
# Serialize GPU inference across the render driver and the precompute queue.
import fcntl
_inference_lock=open(ASSETS/'inference.lock','w')
fcntl.flock(_inference_lock,fcntl.LOCK_EX)
_completion=ROOT/'results'/a.clip/a.method/'run.json'
if _completion.exists() and not a.force_rerun:
    _done=json.loads(_completion.read_text())
    if not a.face or _done.get('face_estimation'):
        print('Reuse completed output:',_completion,flush=True)
        raise SystemExit(0)
sys.argv = [sys.argv[0], "--force-fp32"]  # Consistent fp32 compute on this Turing GPU.
sys.path.insert(0,str(INSTALLED if a.method=='gvhmr' else ASSETS/'vendor/ComfyUI'))
import comfy.options
comfy.options.enable_args_parsing()
import cv2
import numpy as np
import torch
import folder_paths

out = ROOT/'results'/a.clip/a.method
out.mkdir(parents=True,exist_ok=True)
folder_paths.set_output_directory(str(out))
folder_paths.models_dir = str(INSTALLED/'models')
meta=json.loads((ROOT/'inputs'/a.clip/'clip.json').read_text())
K=np.array(meta['K'],dtype=np.float32)
JOINT_NAMES=['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
             'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
start=time.perf_counter()
torch.cuda.init()
torch.cuda.reset_peak_memory_stats()

def save_common(name, verts, joints, faces, **extra):
    assert np.isfinite(verts).all() and np.isfinite(joints).all(), 'Nonfinite predictions'
    proj=joints@K.T
    xy=proj[...,:2]/proj[...,2:3]
    np.savez_compressed(out/f'{name}.npz',vertices_camera=verts,joints_camera=joints,
                        joints_2d=xy,faces=faces,joint_names=JOINT_NAMES,
                        fps=meta['fps'],K=K,**extra)

with torch.inference_mode():
    if a.method=='sam':
        report('sam', 'Loading SAM body model')
        from comfy_extras.nodes_sam3d_body import SAM3DBody_Loader,SAM3DBody_Predict,SAM3DBody_Smooth,SAM3DBody_FaceExpression
        from comfy_extras.sam3d_body.export.glb_skeletal import build_glb_skeletal
        folder_paths.add_model_folder_path('detection',str(ASSETS/'models'))
        model=SAM3DBody_Loader.execute('sam_3d_body_dinov3_bf16.safetensors')[0]
        loaded=time.perf_counter()
        cap=cv2.VideoCapture(str(ROOT/'inputs'/a.clip/'input.mp4')); imgs=[]
        while True:
            ok,frame=cap.read()
            if not ok: break
            imgs.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        cap.release()
        tracks=meta['boxes_xyxy_tracks'] if a.multi else [[b] for b in meta['boxes_xyxy']]
        boxes=[[dict(x=x1,y=y1,width=x2-x1,height=y2-y1) for x1,y1,x2,y2 in frame] for frame in tracks]
        fov=float(np.degrees(2*np.arctan(meta['height']/(2*K[1,1]))))
        pose=None
        report('sam', 'SAM body and hands', 0, len(imgs), frames=len(imgs))
        for offset in range(0,len(imgs),16):
            images=torch.from_numpy(np.stack(imgs[offset:offset+16])).float()/255
            part=SAM3DBody_Predict.execute(model,images,bboxes=boxes[offset:offset+16],run_hand_refinement=not a.no_hand_refinement,fov=fov,batch_size=4)[0]
            if pose is None: pose=part
            else: pose['frames'].extend(part['frames'])
            del images
            report('sam', 'SAM body and hands', min(offset+16, len(imgs)), len(imgs), frames=len(imgs))
        del imgs
        torch.cuda.synchronize(); inferred=time.perf_counter()
        with open(out/'pose_raw.pkl','wb') as f: pickle.dump(pose,f)
        smoothed=SAM3DBody_Smooth.execute(pose,a.sam_smoothing,'savgol',7,30.)[0]
        variants=[('raw',pose),('smooth',smoothed)]
        if a.face:
            report('sam', 'SAM native facial expressions')
            cap=cv2.VideoCapture(str(ROOT/'inputs'/a.clip/'input.mp4')); face_images=[]
            while True:
                ok,frame=cap.read()
                if not ok:break
                face_images.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            cap.release()
            # uint8 staging + small float16 input reduces peak host memory; detector casts to uint8.
            images=torch.from_numpy(np.stack(face_images)).to(torch.float16)/255
            del face_images
            calls=[]
            detector=model.model.face_landmarker
            original_detect=detector.detect_batch
            def traced_detect(*args,**kwargs):
                result=original_detect(*args,**kwargs)
                calls.extend([len(f) for f in result])
                return result
            detector.detect_batch=traced_detect
            try: face=SAM3DBody_FaceExpression.execute(model,smoothed,images)[0]
            finally: detector.detect_batch=original_detect
            del images
            with open(out/'pose_face.pkl','wb') as f:pickle.dump(face,f)
            expr=np.stack([f[0]['expr_params'] for f in face['frames']])
            np.save(out/'expressions.npy',expr)
            (out/'face_diagnostics.json').write_text(json.dumps(dict(detector_calls=len(calls),detections_per_call=calls,frames_with_nonzero_expression=int(np.any(np.abs(expr)>1e-6,axis=-1).sum()),coefficient_min=float(expr.min()),coefficient_max=float(expr.max()),note='Detection calls precede smoothing and gap interpolation; nonzero output is not independent accuracy evidence.'),indent=2))
            variants.append(('face',face))
        for label,data in variants:
            report('sam', 'Exporting SAM '+label+' surface and skeleton')
            frames=data['frames']
            assert len(frames)==meta['frames'] and all(len(f)==(2 if a.multi else 1) for f in frames)
            verts=np.stack([np.stack([p['pred_vertices']+p['pred_cam_t'][None,:] for p in f]) for f in frames])
            if not a.multi:verts=verts[:,0]
            ids=[5,6,7,8,62,41,9,10,11,12,13,14]
            joints=np.stack([np.stack([p['pred_keypoints_3d'][ids]+p['pred_cam_t'][None,:] for p in f]) for f in frames])
            if not a.multi:joints=joints[:,0]
            save_common(label,verts,joints,data['faces'],coordinate_system='opencv_camera_meters',
                        world_recovery=False)
            (out/f'{label}.glb').write_bytes(build_glb_skeletal(data,model=model.model,fps=meta['fps'],
                        camera_translation='absolute',include_face_morphs=True,bone_smooth_window=0))
        details=dict(persons_per_frame=2 if a.multi else 1,hand_refinement=not a.no_hand_refinement,face_estimation=a.face,batch_size=4,precision="fp32",smoother=f'savgol 7, strength {a.sam_smoothing}, rotation threshold 30 deg/frame',
                     integration='cloned ComfyUI core',world_recovery=False)
    else:
        report('gvhmr', 'Loading GVHMR and solving body/world motion')
        # Import only the inference module, avoiding unrelated plugin UI initializers.
        package=types.ModuleType('study_gvhmr')
        package.__path__=[str(INSTALLED/'custom_nodes/ComfyUI-MotionCapture/nodes')]
        sys.modules['study_gvhmr']=package
        node=importlib.import_module('study_gvhmr.inference_node').GVHMRInference()
        from comfy_api.latest._input_impl.video_types import VideoFromFile
        models=INSTALLED/'models/motion_capture'
        config=dict(models_dir=str(models),gvhmr_path=str(models/'gvhmr.safetensors'),
                    vitpose_path=str(models/'vitpose.safetensors'),hmr2_path=str(models/'hmr2.safetensors'),
                    body_models_path=str(models/'body_models'),precision='fp32',attention='auto',dpvo_dir='')
        loaded=time.perf_counter()
        result=node.run_inference(VideoFromFile(str(ROOT/'inputs'/a.clip/'input.mp4')),
                                  VideoFromFile(str(ROOT/'inputs'/a.clip/'boxes.mp4')),config,
                                  moving_camera=meta['moving_camera'],focal_length_mm=float(np.hypot(36,24)),
                                  intrinsics=torch.tensor(K),chunk_size=4)
        if not result[0]: raise RuntimeError(result[2])
        torch.cuda.synchronize(); inferred=time.perf_counter()
        params=np.load(result[0])
        from study_gvhmr.body_model.smplx_utils import make_smplx
        body=make_smplx('supermotion',batch_size=1).cpu()
        from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform
        from smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES
        from rig_export import compact_weights
        rig_poses=[]
        rig_rest=None
        verts,joints,wverts,wjoints=[],[],[],[]
        for i in range(meta['frames']):
            common={k:torch.tensor(params[k][i:i+1]).float() for k in ['body_pose','betas']}
            for space,vs,js in [('incam',verts,joints),('world',wverts,wjoints)]:
                suffix='_incam' if space=='incam' else ''
                result=body(**common,global_orient=torch.tensor(params['global_orient'+suffix][i:i+1]).float(),
                            transl=torch.tensor(params['transl'+suffix][i:i+1]).float(), return_full_pose=True)
                vs.append(result.vertices[0].numpy()); js.append(result.joints[0,[16,17,18,19,20,21,1,2,4,5,7,8]].numpy())
                if space == 'world':
                    shaped = body.bm.v_template[None]+blend_shapes(common['betas'], body.bm.shapedirs)
                    rest_joints = vertices2joints(body.bm.J_regressor, shaped)
                    rotations = batch_rodrigues(result.full_pose.reshape(-1,3)).reshape(1,-1,3,3)
                    _, transforms = batch_rigid_transform(rotations, rest_joints, body.bm.parents)
                    bind = torch.eye(4).repeat(len(body.bm.parents),1,1)
                    bind[:,:3,3] = rest_joints[0]
                    pose_matrices = transforms[0]@bind
                    pose_matrices[:,:3,3] += torch.tensor(params['transl'][i]).float()
                    if rig_rest is None: rig_rest=bind.numpy()
                    rig_poses.append(pose_matrices.numpy())
            report('gvhmr', 'Building GVHMR mesh and native skeleton', i+1, meta['frames'], frames=meta['frames'])
        save_common('native',np.stack(verts),np.stack(joints),body.faces,
                    vertices_world=np.stack(wverts),joints_world=np.stack(wjoints),
                    coordinate_system='opencv_camera_meters; world Y up',world_recovery=True)
        indices, weights = compact_weights(body.bm.lbs_weights.numpy())
        np.savez_compressed(out/'rig_native.npz', rig_names=np.array(SMPLX_JOINT_NAMES[:len(body.bm.parents)]),
                            rig_parents=body.bm.parents.numpy(),rig_rest=rig_rest,rig_pose=np.stack(rig_poses),
                            rig_indices=indices,rig_weights=weights)
        details=dict(integration='existing modified ComfyUI-MotionCapture port',precision='fp32',
                     moving_camera=meta['moving_camera'],vo_focal_mm=float(np.hypot(36,24)),vo='simple_vo' if meta['moving_camera'] else 'static',
                     model_load_included_in_inference_time=True)
    torch.cuda.synchronize()
    timing=dict(clip=a.clip,method=a.method,frames=meta['frames'],fps=meta['fps'],
                setup_seconds=loaded-start,inference_seconds=inferred-loaded,
                total_seconds=time.perf_counter()-start,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,**details)
    (out/'run.json').write_text(json.dumps(timing,indent=2))
    print(json.dumps(timing,indent=2))
