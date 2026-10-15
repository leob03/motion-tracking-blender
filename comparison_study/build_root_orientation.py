"""Rigidly replace SAM root orientation + hip XYZ, leaving its articulation intact.
Reads the actual smoothed bone_001 rotation from the existing skeletal GLB.
MHR and SMPL-X neutral body axes are X-left, Y-up, Z-forward. MHR->OpenCV
is diag(1,-1,-1). The world hip anchor remains the prior XYZ experiment's anchor.
"""
import json, struct, sys
import numpy as np
from scipy.spatial.transform import Rotation
from build_hybrid_candidates import R, load, project, rms_acceleration

def root_rotation(path, frames):
    with open(path,'rb') as f:
        assert f.read(4)==b'glTF'; version,total=struct.unpack('<II',f.read(8));assert version==2
        n,kind=struct.unpack('<II',f.read(8));g=json.loads(f.read(n))
        n,kind=struct.unpack('<II',f.read(8));binary=f.read(n)
    def accessor(index):
        a=g['accessors'][index];v=g['bufferViews'][a['bufferView']]
        assert a['componentType']==5126 and 'byteStride' not in v
        k={'VEC4':4,'MAT4':16,'SCALAR':1}[a['type']]
        return np.frombuffer(binary,'<f4',count=a['count']*k,offset=v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(a['count'],k)
    nodes=g['nodes'];root=next(i for i,n in enumerate(nodes) if n.get('name')=='bone_001')
    parents={c:i for i,n in enumerate(nodes) for c in n.get('children',[])}
    chain=[];i=root
    while i in parents:chain.append(i);i=parents[i]
    chain.append(i);rot=np.tile(np.eye(3),(frames,1,1))
    for node in reversed(chain):
        q=np.tile(nodes[node].get('rotation',[0,0,0,1]),(frames,1))
        for anim in g.get('animations',[]):
            for ch in anim['channels']:
                if ch['target']==dict(node=node,path='rotation'):
                    sampler=anim['samplers'][ch['sampler']];q=accessor(sampler['output']);assert len(q)==frames
        rot=rot@Rotation.from_quat(q).as_matrix()
    bind=np.linalg.inv(accessor(g['skins'][0]['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1))
    assert np.max(np.abs(bind[1,:3,:3]-np.eye(3)))<1e-5
    return np.diag([1.,-1.,-1.])@rot

def build(clip):
    b=R/'results'/clip;out=b/'hybrid_candidates';s=load(b/'sam/smooth.npz');g=load(b/'gvhmr/native.npz');p=load(b/'gvhmr/smpl_params_0001.npz');place=load(out/'gvhmr_xyz_placement.npz')
    F=len(s['vertices_camera']);source=root_rotation(b/'sam/smooth.glb',F);target=Rotation.from_rotvec(p['global_orient']).as_matrix()
    rotation=target@source.transpose(0,2,1);hip=s['joints_camera'][:,[6,7]].mean(1);ghip=g['joints_world'][:,[6,7]].mean(1)
    t=ghip-np.einsum('fij,fj->fi',rotation,hip);T=np.tile(np.eye(4),(F,1,1));T[:,:3,:3]=rotation;T[:,:3,3]=t
    apply=lambda x:np.einsum('fij,fvj->fvi',rotation,x)+t[:,None]
    v=apply(s['vertices_camera']);j=apply(s['joints_camera']);inverse=lambda x:np.einsum('fji,fvj->fvi',rotation,x-t[:,None])
    rc=place['original_camera_to_world_R'];tc=place['original_camera_to_world_t'];vc=np.einsum('fji,fvj->fvi',rc,v-tc[:,None]);jc=np.einsum('fji,fvj->fvi',rc,j-tc[:,None])
    rooterr=float(np.max(np.abs(rotation@source-target)));hiperr=float(np.max(np.abs(j[:,[6,7]].mean(1)-ghip)))
    roundtrip=float(np.max(np.abs(inverse(v.astype(np.float32))-s['vertices_camera'])))
    projectionerr=float(np.max(np.abs(project(inverse(v.astype(np.float32)),g['K'])-project(s['vertices_camera'],g['K']))))
    assert rooterr<1e-6 and hiperr<1e-6 and roundtrip<2e-6 and projectionerr<.01
    np.savez_compressed(out/'gvhmr_xyz_rotation.npz',vertices_world=v.astype(np.float32),vertices_camera=vc.astype(np.float32),joints_world=j.astype(np.float32),faces=s['faces'],fps=g['fps'])
    np.savez_compressed(out/'gvhmr_xyz_rotation_placement.npz',world_from_camera=T,source_root_rotation_camera=source,target_root_rotation_world=target,original_camera_to_world_R=rc,original_camera_to_world_t=tc,compensating_camera_to_world=T,world_from_glb_yup_meters=T@np.diag([1.,-1.,-1.,1.]),input_coordinates='OpenCV camera-space meters, including pred_cam_t',output_coordinates='GVHMR inferred Y-up world meters',fps=g['fps'])
    m=dict(clip=clip,frames=F,root_rotation_max_matrix_error=rooterr,hip_max_error_m=hiperr,rigid_roundtrip_max_m=roundtrip,compensating_camera_reprojection_max_px=projectionerr,fixed_camera_joint_projection_change_mean_px=float(np.linalg.norm(project(jc,g['K'])-project(s['joints_camera'],g['K']),axis=-1).mean()),rotation_correction_mean_deg=float(np.degrees(Rotation.from_matrix(target@(rc@source).transpose(0,2,1)).magnitude()).mean()),depth_acceleration_rms_m_s2=rms_acceleration(ghip@rc[0,:,2],float(g['fps'])),anchor='Midpoint of the same two hip landmarks used in XYZ, not raw rig translation.',root_source='Smoothed SAM GLB bone_001 global rotation, corrected for OpenCV axes; GVHMR SMPL-X global_orient in world.',camera_note='The supplied compensating camera preserves this subject projection exactly. It is actor-specific, not a new independently estimated shared scene camera. Other actors/background are not preserved.',limitations='No ground truth, foot contact, shared-camera optimization or scale fitting. Unreliable GVHMR root orientation can worsen partial-body footage.')
    (out/'orientation_metrics.json').write_text(json.dumps(m,indent=2));print(json.dumps(m),flush=True)
if __name__=='__main__':
    for clip in sys.argv[1:]:build(clip)
