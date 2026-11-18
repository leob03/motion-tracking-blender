"""Lens and gravity of a locked-off plate from the image itself (GeoCalib, vendored in third_party/geocalib).

Used only when the plate carries no lens metadata and the artist typed no focal length. GeoCalib is the
single-image calibration ViPE uses to initialise its own intrinsics; here it solves several frames of the
shot together with one shared focal length, and returns the camera's gravity direction (tilt and roll), so a
locked-off scene comes in level.
"""
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = 'models/geocalib/pinhole.tar'


def read_frames(video, count=8):
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(np.linspace(0, max(total-1, 0), min(count, max(total, 1))).round().astype(int).tolist())
    frames = []
    try:
        for index in range(total):
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError('No frames to estimate the lens from')
    return frames


def estimate(video, study, count=8, device=None):
    """Return dict(focal_px, focal_sigma_px, up_camera, frames, method) at the video's own resolution.

    up_camera is the world up direction expressed in OpenCV camera axes (x right, y down, z forward).
    """
    import torch
    from third_party.geocalib import GeoCalib
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = GeoCalib(weights=str(Path(study)/WEIGHTS)).to(device).eval()
    frames = read_frames(video, count)
    images = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255).to(device)
    try:
        with torch.no_grad():
            result = model.calibrate(images, camera_model='pinhole', shared_intrinsics=True)
        focal = float(result['camera'].f[0, 1].item())
        sigma = float(result['focal_uncertainty'][0].item()) if 'focal_uncertainty' in result else float('nan')
        # GeoCalib's "gravity" vector is the world UP direction in OpenCV camera axes: from_rp(0, 0) = (0, -1, 0),
        # i.e. a level camera; pitch adds +z when the camera looks up.
        up = result['gravity'].vec3d.float().cpu().numpy()
    finally:
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    up = np.median(up/np.linalg.norm(up, axis=1, keepdims=True), axis=0)
    up /= np.linalg.norm(up)
    tilt = float(np.degrees(np.arcsin(np.clip(up[2], -1, 1))))
    roll = float(np.degrees(np.arctan2(-up[0], -up[1])))
    return dict(focal_px=focal, focal_sigma_px=sigma, up_camera=up.tolist(), tilt_up_degrees=tilt, roll_degrees=roll,
                frames=len(frames), method='GeoCalib (shared focal over evenly spaced frames)')


def level_camera(up_camera):
    """Camera-to-world rotation (Y-up world) of a camera whose world up is up_camera; no yaw (looks along -Z)."""
    up = np.asarray(up_camera, float)
    up /= np.linalg.norm(up)
    forward = np.array([0., 0., 1.])-up[2]*up
    if np.linalg.norm(forward) < 1e-3:  # looking straight up or down: any horizontal axis will do
        forward = np.array([1., 0., 0.])-up[0]*up
    forward /= np.linalg.norm(forward)
    x_world = np.cross(up, -forward)
    return np.stack([x_world, up, -forward])  # rows: world axes in camera coordinates -> camera-to-world rotation
