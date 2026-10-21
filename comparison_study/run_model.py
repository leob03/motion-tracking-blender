"""Run the installed GVHMR port or pinned ComfyUI SAM port on shared inputs.
All generated files stay within this study. No remote inference is used.

One-shot:  run_model.py gvhmr|sam CLIP [--work-root DIR] ...
Warm mode: run_model.py serve gvhmr|sam --socket PATH [--idle-timeout S]
           Keeps the model in host RAM between jobs (GPU memory is released after each job).
"""
import argparse
import contextlib
import importlib
import json
import os
import pickle
import socket
import sys
import time
import types
from pathlib import Path

ASSETS = Path(__file__).resolve().parent
INSTALLED = Path('/home/innovation/dev-lbringer/ComfyUI_nightly')
sys.path.insert(0, str(ASSETS.parent/'production_bridge'))
JOINT_NAMES=['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
             'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
FULLFRAME_DIAGONAL_MM = (36**2+24**2)**.5
_cache = {}


def parse(argv):
    p = argparse.ArgumentParser()
    p.add_argument('method', choices=['gvhmr','sam'])
    p.add_argument('clip')
    p.add_argument('--face',action='store_true')
    p.add_argument('--multi',action='store_true')
    p.add_argument('--force-rerun',action='store_true')
    p.add_argument('--work-root',type=Path,help='Isolated inputs/results directory for a ComfyUI job')
    p.add_argument('--sam-smoothing',type=float,default=1.)
    p.add_argument('--no-hand-refinement',action='store_true')
    return p.parse_args(argv)


def boot(method):
    """Import the ComfyUI tree matching this method; once per process."""
    sys.argv = [sys.argv[0], "--force-fp32"]  # Consistent fp32 compute on this Turing GPU.
    sys.path.insert(0,str(INSTALLED if method=='gvhmr' else ASSETS/'vendor/ComfyUI'))
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import folder_paths
    folder_paths.models_dir = str(INSTALLED/'models')
    torch.cuda.init()


class KnownCameraVO:
    """Stands in for SimpleVO: GVHMR only consumes the camera rotations (angular velocity)."""
    poses = None

    def __init__(self, *args, **kwargs):
        pass

    def compute(self):
        import numpy as np
        return [np.linalg.inv(c2w) for c2w in self.poses]


def gvhmr_node():
    if 'gvhmr' not in _cache:
        # Import only the inference module, avoiding unrelated plugin UI initializers.
        package=types.ModuleType('study_gvhmr')
        package.__path__=[str(INSTALLED/'custom_nodes/ComfyUI-MotionCapture/nodes')]
        sys.modules['study_gvhmr']=package
        module=importlib.import_module('study_gvhmr.inference_node')
        if _cache.get('warm'):
            loaded={}
            original=module.GVHMRInference._load_models
            def cached(cls,config):
                key=json.dumps(config,sort_keys=True)
                if key not in loaded:
                    loaded[key]=original(config)
                return loaded[key]
            module.GVHMRInference._get_or_load_model=classmethod(cached)
        _cache['gvhmr']=(module,module.GVHMRInference(),module.SimpleVO)
    return _cache['gvhmr']


def sam_model():
    if 'sam' not in _cache:
        import folder_paths
        from comfy_extras.nodes_sam3d_body import SAM3DBody_Loader
        folder_paths.add_model_folder_path('detection',str(ASSETS/'models'))
        _cache['sam']=SAM3DBody_Loader.execute('sam_3d_body_dinov3_bf16.safetensors')[0]
    return _cache['sam']


def run(a):
    from progress import report
    import cv2
    import numpy as np
    import torch
    import folder_paths
    root = a.work_root.resolve() if a.work_root else ASSETS
    _completion=root/'results'/a.clip/a.method/'run.json'
    if _completion.exists() and not a.force_rerun:
        _done=json.loads(_completion.read_text())
        if not a.face or _done.get('face_estimation'):
            print('Reuse completed output:',_completion,flush=True)
            return
    out = root/'results'/a.clip/a.method
    out.mkdir(parents=True,exist_ok=True)
    folder_paths.set_output_directory(str(out))
    meta=json.loads((root/'inputs'/a.clip/'clip.json').read_text())
    K=np.array(meta['K'],dtype=np.float32)
    known_path=root/'inputs'/a.clip/'known_camera.npz'
    known=dict(np.load(known_path)) if known_path.exists() else None
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    warm_reused = a.method in _cache

    def save_common(name, verts, joints, faces, **extra):
        assert np.isfinite(verts).all() and np.isfinite(joints).all(), 'Nonfinite predictions'
        proj=joints@K.T
        xy=proj[...,:2]/proj[...,2:3]
        np.savez_compressed(out/f'{name}.npz',vertices_camera=verts,joints_camera=joints,
                            joints_2d=xy,faces=faces,joint_names=JOINT_NAMES,
                            fps=meta['fps'],K=K,**extra)

    with torch.inference_mode():
        if a.method=='sam':
            report('sam', 'Loading SAM body model' if not warm_reused else 'Reusing warm SAM body model')
            from comfy_extras.nodes_sam3d_body import SAM3DBody_Predict,SAM3DBody_Smooth,SAM3DBody_FaceExpression
            from comfy_extras.sam3d_body.export.glb_skeletal import build_glb_skeletal
            model=sam_model()
            loaded=time.perf_counter()
            cap=cv2.VideoCapture(str(root/'inputs'/a.clip/'input.mp4')); imgs=[]
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
            torch.cuda.synchronize(); inferred=time.perf_counter()
            with open(out/'pose_raw.pkl','wb') as f: pickle.dump(pose,f)
            smoothed=SAM3DBody_Smooth.execute(pose,a.sam_smoothing,'savgol',7,30.)[0]
            variants=[('raw',pose),('smooth',smoothed)]
            if a.face:
                report('sam', 'SAM native facial expressions')
                # uint8 staging + small float16 input reduces peak host memory; detector casts to uint8.
                # The decoded frames are reused instead of decoding the clip a second time.
                images=torch.from_numpy(np.stack(imgs)).to(torch.float16)/255
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
            del imgs
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
            report('gvhmr', 'Loading GVHMR and solving body/world motion' if not warm_reused else 'Reusing warm GVHMR models; solving body/world motion')
            module,node,simple_vo=gvhmr_node()
            from comfy_api.latest._input_impl.video_types import VideoFromFile
            models=INSTALLED/'models/motion_capture'
            config=dict(models_dir=str(models),gvhmr_path=str(models/'gvhmr.safetensors'),
                        vitpose_path=str(models/'vitpose.safetensors'),hmr2_path=str(models/'hmr2.safetensors'),
                        body_models_path=str(models/'body_models'),precision='fp32',attention='auto',dpvo_dir='')
            # SimpleVO converts full-frame millimetres back to pixels by the image diagonal;
            # derive them from the job focal so odometry and GVHMR share one lens.
            vo_focal_mm=float(K[0,0]*FULLFRAME_DIAGONAL_MM/np.hypot(meta['width'],meta['height']))
            module.SimpleVO=simple_vo
            if known is not None:
                KnownCameraVO.poses=known['c2w']
                module.SimpleVO=KnownCameraVO
            loaded=time.perf_counter()
            result=node.run_inference(VideoFromFile(str(root/'inputs'/a.clip/'input.mp4')),
                                      VideoFromFile(str(root/'inputs'/a.clip/'boxes.mp4')),config,
                                      moving_camera=meta['moving_camera'] or known is not None,focal_length_mm=vo_focal_mm,
                                      intrinsics=torch.tensor(known['K'] if known is not None else K),chunk_size=4)
            if not result[0]: raise RuntimeError(result[2])
            torch.cuda.synchronize(); inferred=time.perf_counter()
            params=np.load(result[0])
            if 'body' not in _cache:
                from study_gvhmr.body_model.smplx_utils import make_smplx
                _cache['body']=make_smplx('supermotion',batch_size=1).cpu()
            body=_cache['body']
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
                         moving_camera=meta['moving_camera'] or known is not None,vo_focal_mm=vo_focal_mm,
                         vo='known scene camera rotations' if known is not None else 'simple_vo' if meta['moving_camera'] else 'static',
                         model_load_included_in_inference_time=not warm_reused)
        torch.cuda.synchronize()
        timing=dict(clip=a.clip,method=a.method,frames=meta['frames'],fps=meta['fps'],
                    setup_seconds=loaded-start,inference_seconds=inferred-loaded,
                    total_seconds=time.perf_counter()-start,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                    gpu=torch.cuda.get_device_name(),torch=torch.__version__,warm_model=warm_reused,**details)
        (out/'run.json').write_text(json.dumps(timing,indent=2))
        print(json.dumps(timing,indent=2))


@contextlib.contextmanager
def inference_lock():
    # Serialize GPU inference across one-shot runs, warm servers and the precompute queue.
    import fcntl
    with open(ASSETS/'inference.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield


def release_gpu():
    import comfy.model_management
    import torch
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    torch.cuda.empty_cache()


def _stream_handlers():
    import logging
    loggers=[logging.root]+[l for l in logging.root.manager.loggerDict.values() if isinstance(l,logging.Logger)]
    return [h for logger in loggers for h in logger.handlers if type(h) is logging.StreamHandler]


@contextlib.contextmanager
def job_output(log):
    """Send prints, progress bars and logging handlers to this job's log.

    Streams are swapped without flushing: a handler created during an earlier job may still
    hold that job's (now closed) log, possibly wrapped, and is repaired here instead.
    """
    handlers=_stream_handlers()
    streams=[sys.__stderr__ if getattr(h.stream,'closed',False) else h.stream for h in handlers]
    for h in handlers:
        h.stream=log
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            yield
    finally:
        for h,s in zip(handlers,streams):
            h.stream=s
        for h in _stream_handlers():
            if h.stream is log or getattr(h.stream,'closed',False):
                h.stream=sys.__stderr__


def serve(method, path, idle_timeout):
    """Answer one job per connection: {argv, status_file, log_file} -> {ok, error}."""
    _cache['warm']=True
    boot(method)
    path=Path(path)
    path.unlink(missing_ok=True)
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.bind(str(path))
    os.chmod(path,0o600)
    inode=path.stat().st_ino
    server.listen(1)
    server.settimeout(idle_timeout)
    print(f'[warm {method}] ready on {path}',flush=True)
    try:
        while True:
            try:
                connection,_=server.accept()
            except socket.timeout:
                print(f'[warm {method}] idle for {idle_timeout:.0f}s; exiting',flush=True)
                return
            with connection:
                connection.settimeout(None)
                request=json.loads(connection.makefile('r').readline())
                if request.get('ping'):
                    connection.sendall(b'{"ok": true}\n')
                    continue
                os.environ['MOCAP_STATUS_FILE']=request['status_file']
                reply=dict(ok=True)
                try:
                    with open(request['log_file'],'a',buffering=1) as log, job_output(log):
                        try:
                            with inference_lock():
                                run(parse(request['argv']))
                        finally:
                            release_gpu()
                except (Exception,SystemExit) as exc:
                    import traceback
                    traceback.print_exc()
                    reply=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
                try:
                    connection.sendall((json.dumps(reply)+'\n').encode())
                except OSError:
                    pass  # The job was cancelled; keep serving.
    finally:
        # A newer server may own the path by now; remove only our own socket.
        with contextlib.suppress(OSError):
            if path.stat().st_ino==inode:
                path.unlink()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='serve':
        p=argparse.ArgumentParser()
        p.add_argument('serve'); p.add_argument('method',choices=['gvhmr','sam'])
        p.add_argument('--socket',required=True); p.add_argument('--idle-timeout',type=float,default=900.)
        s=p.parse_args()
        serve(s.method,s.socket,s.idle_timeout)
        return
    a=parse(sys.argv[1:])
    with inference_lock():
        boot(a.method)
        run(a)


if __name__=='__main__':
    main()
