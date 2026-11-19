"""Make assets/flame_open_mhr_landmarks.npz: SAM 3D Body's face keypoints embedded on the FLAME 2023 Open surface.

Run once (CPU, about a minute); the worker only reads the result.
    python tools/make_flame_landmarks.py --study "$MOCAP_STUDY_ROOT"
Inputs: FLAME 2023 Open (CC-BY-4.0), MHR's TorchScript model + head mask (Apache-2.0), SAM 3D Body's keypoint map and
mesh faces (SAM License). FLAME Open is fitted (similarity + 100 identity components) to MHR's neutral head; each SAM
face keypoint is then projected onto the fitted FLAME skin, giving a face index and barycentric weights that hold for
any FLAME identity. Replaces the SMPL-X landmark set and SMIRK's landmark_embedding.npy in the neck join.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from safetensors import safe_open
from scipy import sparse
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import face_assets  # noqa: E402

# SAM face keypoints that barely move with expressions in either model, front of face only (ears are poorly observed):
# nose bridge 4, 3, 5; nose tip 2, 108; nostrils 113, 114, 116, 117; inner eye corners 50, 91, 26, 74; outer 92, 51, 75, 27.
STABLE = [4, 3, 5, 2, 108, 113, 114, 116, 117, 50, 91, 26, 74, 92, 51, 75, 27]
JAW_PROTECT = .8  # FLAME jaw skin weight above which the chin edge stays as fitted during neck smoothing

parser = argparse.ArgumentParser()
parser.add_argument('--study', type=Path, required=True)
parser.add_argument('--output', type=Path, default=face_assets.LANDMARKS)
args = parser.parse_args()
study = args.study.resolve()

with face_assets.flame_model_path(study).open('rb') as stream:  # trusted server asset (numpy/scipy-only pickle)
    flame = pickle.load(stream, encoding='latin1')
T = np.asarray(flame['v_template'], float)
S = np.asarray(flame['shapedirs'], float)[:, :, :100]
F = np.asarray(flame['f'], np.int64)
jaw_weight = np.asarray(flame['weights'], float)[:, 2]

mhr = torch.jit.load(str(study/'vendor/MHR/assets/mhr_model.pt'), map_location='cpu')
with torch.no_grad():
    vertices, _ = mhr(torch.zeros(1, 45), torch.zeros(1, 204), torch.zeros(1, 72))
V = vertices[0].double().numpy()/100  # neutral MHR, metres, y up
with safe_open(str(study/face_assets.SAM_WEIGHTS), 'pt') as weights:
    MF = weights.get_tensor('head_pose.faces').numpy().astype(np.int64)
face_map = face_assets.face_keypoint_map(study).astype(float)
head_mask = np.load(study/'vendor/MHR/tools/mhr_smpl_conversion/assets/head_hand_mask.npz')['head_mask'] > .5

# FLAME eyeballs are separate connected components; fit and embed on the skin only.
n = len(T)
adjacency = sparse.coo_matrix((np.ones(3*len(F)), (F.ravel(), np.roll(F, 1, 1).ravel())), shape=(n, n))
_, label = connected_components(adjacency, directed=False)
skin = label == np.bincount(label).argmax()
skin_faces = F[skin[F].all(1)]
target = trimesh.Trimesh(V, MF[head_mask[MF].all(1)], process=False)
keypoints = face_map@V


def anchors(points):
    top = points[points[:, 1].argmax()]
    upper = points[points[:, 1] > top[1]-.25]
    return top, upper[upper[:, 2].argmax()]  # head top, nose tip (both models face +z)


ft, fn = anchors(T[skin])
mt, mn = anchors(V[head_mask])
s = np.linalg.norm(mt-mn)/np.linalg.norm(ft-fn)
R = np.eye(3)
t = mn-s*fn
beta = np.zeros(S.shape[2])
use = skin & (T[:, 1] > np.percentile(T[skin, 1], 25))  # upper head; MHR replaces FLAME's lower neck anyway
for iteration in range(40):
    shaped = T+S@beta
    closest, distance, tri = target.nearest.on_surface(s*shaped[use]@R.T+t)
    keep = distance < (np.percentile(distance, 90) if iteration > 5 else .05)
    index = np.flatnonzero(use)[keep]
    Y, N = closest[keep], target.face_normals[tri][keep]
    P = shaped[index]
    mp, my = P.mean(0), Y.mean(0)
    u, sv, vt = np.linalg.svd((P-mp).T@(Y-my))
    D = np.diag([1, 1, np.sign(np.linalg.det(vt.T@u.T))])
    R = vt.T@D@u.T
    s = float((sv*np.diag(D)).sum()/((P-mp)**2).sum())
    t = my-s*R@mp
    if iteration >= 3:
        Yl, Nl = (Y-t)@R/s, N@R
        A = np.einsum('kd,kdc->kc', Nl, S[index])
        b = np.einsum('kd,kd->k', Nl, Yl-T[index])
        beta = np.linalg.solve(A.T@A+1e-4*np.eye(A.shape[1]), A.T@b)
fitted = s*(T+S@beta)@R.T+t
_, fit_distance, _ = target.nearest.on_surface(fitted[use])
closest, keypoint_distance, tri = trimesh.Trimesh(fitted, skin_faces, process=False).nearest.on_surface(keypoints)
bary = trimesh.triangles.points_to_barycentric(fitted[skin_faces[tri]], closest)
face_index = np.array([np.flatnonzero((F == skin_faces[k]).all(1))[0] for k in tri])
assert np.allclose(np.einsum('kc,kcd->kd', bary, fitted[F[face_index]]), closest, atol=1e-6)
assert keypoint_distance.max() < .004, 'a SAM face keypoint is more than 4 mm from the fitted FLAME surface'
carrying = np.unique(F[face_index])
protected = np.union1d(np.unique(F[np.isin(F, carrying).any(1)]), np.flatnonzero(jaw_weight > JAW_PROTECT))
report = dict(flame_scale_to_mhr=s, surface_gap_mm_median=float(np.median(fit_distance)*1000),
              surface_gap_mm_p90=float(np.percentile(fit_distance, 90)*1000),
              keypoint_gap_mm_median=float(np.median(keypoint_distance)*1000), keypoint_gap_mm_max=float(keypoint_distance.max()*1000),
              keypoints=len(keypoints), stable=STABLE, protected_vertices=len(protected), jaw_protect_weight=JAW_PROTECT)
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output, lmk_faces_idx=face_index.astype(np.int32), lmk_bary_coords=bary.astype(np.float64),
                    stable=np.array(STABLE, np.int32), protected_vertices=protected.astype(np.int32),
                    keypoint_gap_m=keypoint_distance.astype(np.float32), report=np.array(json.dumps(report)))
print(args.output, json.dumps(report))
