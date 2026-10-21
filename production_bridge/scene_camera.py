"""Place GVHMR's world-space body in a known (solved) scene camera's world.

The known camera fixes where the plate is seen from; GVHMR supplies a temporally consistent
world trajectory. Fitting each frame to the camera independently would follow the per-frame
depth estimates and reintroduce the depth jitter GVHMR avoids, so the body world is placed
with ONE rigid transform for the whole shot (robustly fitted). The remaining plate mismatch is
removed by a small, smoothed sideways correction parallel to the image plane ("plate lock"):
depth from the camera is never changed by it.

Conventions: Y-up metric world, OpenCV camera-to-world matrices, (F, J, 3) joint arrays.
"""
import numpy as np


def to_world(c2w, points):
    return np.einsum('fij,f...j->f...i', c2w[:, :3, :3], points)+c2w[:, None, :3, 3]


def to_camera(c2w, points):
    return np.einsum('fji,f...j->f...i', c2w[:, :3, :3], points-c2w[:, None, :3, 3])


def project(K, points):
    p = np.einsum('fij,f...j->f...i', K, points)
    return p[..., :2]/p[..., 2:3]


def umeyama(source, target, weights, with_scale=False):
    """Weighted similarity (s, R, t) minimizing sum w ||s R x + t - y||^2."""
    w = weights/weights.sum()
    mu_s, mu_t = w@source, w@target
    xs, xt = source-mu_s, target-mu_t
    covariance = (xt*w[:, None]).T@xs
    U, S, Vt = np.linalg.svd(covariance)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U@Vt))
    R = U@D@Vt
    scale = float(np.trace(np.diag(S)@D)/(w@(xs**2).sum(1))) if with_scale else 1.
    return scale, R, mu_t-scale*R@mu_s


def register(world_joints, camera_joints, c2w, iterations=5):
    """Robust rigid transform from GVHMR world to the known camera's world.

    Targets are GVHMR's camera-space joints carried through the known camera. Frames whose
    camera-space depth disagrees with the shot's consistent placement are down-weighted.
    """
    frames, joints = world_joints.shape[:2]
    target = to_world(c2w, camera_joints)
    weights = np.ones(frames)
    for _ in range(iterations):
        _, R, t = umeyama(world_joints.reshape(-1, 3), target.reshape(-1, 3), np.repeat(weights, joints))
        residual = np.linalg.norm(world_joints@R.T+t-target, axis=-1).mean(1)
        spread = max(1.4826*np.median(residual), 1e-3)
        weights = 1/(1+(residual/(2*spread))**2)
    scale = umeyama(world_joints.reshape(-1, 3), target.reshape(-1, 3), np.repeat(weights, joints), with_scale=True)[0]
    return R, t, dict(residual_median_m=float(np.median(residual)), residual_p95_m=float(np.percentile(residual, 95)),
                      downweighted_frames=int((weights < .5).sum()), scale_estimate=scale)


def gaussian_smooth(values, sigma):
    if sigma <= 0 or len(values) < 2:
        return values
    radius = int(3*sigma+.5)
    kernel = np.exp(-.5*(np.arange(-radius, radius+1)/sigma)**2)
    padded = np.pad(values, ((radius, radius), (0, 0)), mode='edge')
    smoothed = np.stack([np.convolve(padded[:, axis], kernel/kernel.sum(), mode='valid') for axis in range(values.shape[1])], 1)
    return smoothed


def plate_lock(anchor_world, reference_camera, c2w, strength=1., sigma=3.):
    """World offsets moving the anchor onto the reference viewing ray at unchanged camera depth."""
    current = to_camera(c2w, anchor_world[:, None])[:, 0]
    reference = reference_camera*(current[:, 2:3]/np.maximum(reference_camera[:, 2:3], 1e-6))
    offset = np.einsum('fij,fj->fi', c2w[:, :3, :3], reference-current)
    return strength*gaussian_smooth(offset, sigma)


def orientation_lock(world_rotations, camera_rotations, c2w, strength=1., sigma=3.):
    """Rotations (F, 3, 3) turning world root orientations toward the plate-consistent ones, smoothed.

    camera_rotations are the network's camera-space root orientations; carried through the known
    camera they give the orientation that reproduces the plate. Only the smoothed difference is applied.
    """
    from scipy.spatial.transform import Rotation
    plate = np.einsum('fij,fjk->fik', c2w[:, :3, :3], camera_rotations)
    delta = Rotation.from_matrix(plate@world_rotations.transpose(0, 2, 1)).as_rotvec()
    return Rotation.from_rotvec(strength*gaussian_smooth(delta, sigma)).as_matrix()


def plate_error(K, c2w, world_joints, reference_camera_joints):
    """Pixel distance between the placed joints and the network's own camera-space projection."""
    placed = project(K, to_camera(c2w, world_joints))
    reference = project(K, reference_camera_joints)
    distance = np.linalg.norm(placed-reference, axis=-1)
    return dict(mean_px=float(distance.mean()), p95_px=float(np.percentile(distance, 95)), max_px=float(distance.max()))


def tilt_degrees(R, scene_up=(0., 1., 0.)):
    """Angle between GVHMR's registered gravity axis and the scene's up direction."""
    up = np.asarray(scene_up, float)
    return float(np.degrees(np.arccos(np.clip(R[:, 1]@(up/np.linalg.norm(up)), -1, 1))))
