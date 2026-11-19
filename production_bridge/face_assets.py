"""Commercially usable FLAME face assets shared by face_worker.py (fit + neck join) and flame_export.py (Blender keys).

FLAME 2023 Open (MPI-IS, CC-BY-4.0) is FLAME 2023 with a different expression basis: same topology, identity basis,
joints and skin weights. The head is placed on SAM's body with SAM 3D Body's own face keypoints (a fixed linear map of
MHR vertices) matched to the same keypoints embedded on the FLAME Open surface (assets/flame_open_mhr_landmarks.npz,
made by tools/make_flame_landmarks.py). No SMPL-X, SMIRK or FLAME 2023 asset is read.
"""
from pathlib import Path

import numpy as np

FLAME_MODEL = 'models/face/flame_downloads/FLAME2023_Open/flame2023_Open.pkl'
FLAME_NAME = 'FLAME 2023 Open'
FLAME_CREDIT = ('Face model: FLAME 2023 Open, Max Planck Institute for Intelligent Systems, CC-BY-4.0 '
                '(T. Li, T. Bolkart, M. J. Black, H. Li, J. Romero: Learning a model of facial shape and expression '
                'from 4D scans, SIGGRAPH Asia 2017). https://flame.is.tue.mpg.de')
LANDMARKS = Path(__file__).resolve().with_name('assets')/'flame_open_mhr_landmarks.npz'
SAM_WEIGHTS = 'models/sam_3d_body_dinov3_bf16.safetensors'
MHR_VERTICES = 18439
BODY_KEYPOINTS = 70  # SAM's 308 keypoints: 70 body/hand, then 238 face


# Captures fitted before 0.8.0 used non-commercial FLAME 2023. Their parameters carry no 'flame_model' entry; the
# export (scripts/upgrade_flame_bundle.py) must replay them with the model they were fitted with.
LEGACY_MODELS = {'FLAME2023': 'models/face/flame_downloads/FLAME2023/flame2023.pkl'}


def flame_model_path(study, name=FLAME_NAME):
    if name == FLAME_NAME:
        return Path(study)/FLAME_MODEL
    if name in LEGACY_MODELS:
        return Path(study)/LEGACY_MODELS[name]
    raise ValueError(f'Unknown FLAME model {name!r}')


def fitted_model(params):
    """Name of the FLAME model a fit used (face_worker.py parameters)."""
    return str(params['flame_model']) if 'flame_model' in params else 'FLAME2023'


def landmarks():
    """Embedding of SAM's face keypoints on FLAME Open: faces, barycentrics, the stable subset and protected vertices."""
    with np.load(LANDMARKS, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def face_keypoint_map(study, rows=None):
    """SAM 3D Body's face keypoints as weights over MHR vertices (rows sum to 1; no joint terms)."""
    from safetensors import safe_open
    with safe_open(str(Path(study)/SAM_WEIGHTS), 'pt') as weights:
        mapping = weights.get_slice('head_pose.keypoint_mapping')
        full = mapping[BODY_KEYPOINTS:, :].float().numpy()
    face, joints = full[:, :MHR_VERTICES], full[:, MHR_VERTICES:]
    if np.abs(joints).max() > 0 or not np.allclose(face.sum(1), 1, atol=1e-4):
        raise ValueError('Unexpected SAM 3D Body face keypoint map; the FLAME placement needs vertex-only keypoints')
    return face if rows is None else face[rows]
