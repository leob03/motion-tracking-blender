"""SAM 3D Body on a known camera: lock the actor's proportions, then place the body in depth.

SAM 3D Body predicts a good per-frame pose that matches the plate, but it guesses the person's distance from
the camera independently on every frame (body size and distance trade off), so in a scene the body jitters
back and forth. With a known camera (Camera Track / ViPE solve, or a locked-off camera) two steps fix that:

1. lock_identity: one body shape and one skeleton scale for the whole shot (per-frame medians), re-run
   through MHR so the mesh, joints and rig agree and bone lengths stay constant.
2. solve_placement: one number per frame, q_i, scaling SAM's camera-space root translation, i.e. moving the
   body along its viewing ray (its position on the plate does not move), plus one global factor s mapping
   SAM metres to scene units (scaling about the camera centre, which leaves the plate unchanged). Chosen so
   that, in the scene:
     * the body's apparent size stays what SAM saw (prior: q_i = locked size / SAM's size on that frame),
     * the root moves smoothly along the viewing ray (robust, so genuinely fast moves are kept),
     * a planted, visible foot does not slide,
     * s stays 1 unless planted feet and camera travel make it measurable.
   Ankles outside the frame never get contacts, so close-ups fall back to size + smoothness.

Conventions: OpenCV camera space (x right, y down, z forward), camera-to-world (F, 4, 4) matrices, metres for
SAM. The 12 study joints: 0 lsh 1 rsh 2 lel 3 rel 4 lwr 5 rwr 6 lhip 7 rhip 8 lknee 9 rknee 10 lank 11 rank.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

STUDY_KEYPOINTS = [5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]  # MHR70 keypoints of the 12 study joints
BONES = [(0, 2), (2, 4), (1, 3), (3, 5), (6, 8), (8, 10), (7, 9), (9, 11), (0, 1), (6, 7)]
ANKLES = (10, 11)
DEFAULTS = dict(size_sigma=0.03, plate_budget=0.004, depth_acc_sigma=4.0, acc_robust=2.0, contact_sigma=0.08, scale_sigma=0.25,
                contact_speed=0.35, min_scale=0.8, max_scale=1.25, min_scale_contacts=12, min_travel_ratio=0.15)


def body_size(joints):
    """Sum of limb and torso lengths of the 12 study joints (F, 12, 3) -> (F,); insensitive to pose."""
    size = sum(np.linalg.norm(joints[:, a]-joints[:, b], axis=-1) for a, b in BONES)
    torso = np.linalg.norm((joints[:, 0]+joints[:, 1])/2-(joints[:, 6]+joints[:, 7])/2, axis=-1)
    return size+2*torso


def lock_identity(pose, head, person=0, chunk=64):
    """Return (pose with one shape/scale for the actor, re-run through MHR; report).

    `head` is SAM's body MHR head (model.head_pose). Pose, hands and expressions keep their per-frame values.
    Frames keep pred_cam_t; mesh, keypoints, joint coordinates/rotations and mhr_model_params are recomputed.
    """
    import torch
    frames = pose['frames']
    people = [f[person] for f in frames]
    shapes = np.stack([np.asarray(p['shape_params'], np.float32) for p in people])
    scales = np.stack([np.asarray(p['scale_params'], np.float32) for p in people])
    shape, scale = np.median(shapes, 0), np.median(scales, 0)
    device = head.scale_mean.device
    out = []
    for start in range(0, len(people), chunk):
        part = people[start:start+chunk]
        n = len(part)

        def stack(key):
            return torch.from_numpy(np.stack([np.asarray(p[key], np.float32) for p in part])).to(device)
        rot = stack('global_rot')
        expr = stack('expr_params') if all(p.get('expr_params') is not None for p in part) else None
        verts, kp, joints, model_params, rots = head.mhr_forward(
            global_trans=torch.zeros_like(rot), global_rot=rot, body_pose_params=stack('body_pose_params'),
            hand_pose_params=stack('hand_pose_params'),
            scale_params=torch.from_numpy(np.repeat(scale[None], n, 0)).to(device),
            shape_params=torch.from_numpy(np.repeat(shape[None], n, 0)).to(device), expr_params=expr,
            return_keypoints=True, return_joint_coords=True, return_model_params=True, return_joint_rotations=True)
        flip = lambda x: (x.float()*x.new_tensor([1., -1., -1.])).cpu().numpy()  # MHR y-up -> camera y-down
        verts, kp, joints = flip(verts), flip(kp), flip(joints)
        model_params, rots = model_params.float().cpu().numpy(), rots.float().cpu().numpy()
        for i, p in enumerate(part):
            q = dict(p)
            q.update(shape_params=shape.copy(), scale_params=scale.copy(), pred_vertices=verts[i],
                     pred_keypoints_3d=kp[i, :70], pred_joint_coords=joints[i], pred_global_rots=rots[i],
                     mhr_model_params=model_params[i])
            if 'pred_face_keypoints_3d' in p:
                q['pred_face_keypoints_3d'] = kp[i, 70:]
            out.append(q)
    locked = dict(pose, frames=[[q] for q in out])
    relative = lambda values, centre: float(np.median(np.linalg.norm(values-centre, axis=1))/max(np.linalg.norm(centre), 1e-6))
    report = dict(frames=len(out), shape_spread=relative(shapes, shape), scale_spread=relative(scales, scale),
                  note='Per-actor median MHR shape and skeleton scale; pose, hands and expressions unchanged.')
    return locked, report


def trainable_copy(module, device=None):
    """Copy of a module usable with autograd even if its weights were loaded under torch.inference_mode."""
    import copy
    import torch
    def normal(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone() if value.is_inference() else value
        if isinstance(value, list):
            return [normal(v) for v in value]
        if isinstance(value, tuple):
            return tuple(normal(v) for v in value)
        if isinstance(value, dict):
            return {k: normal(v) for k, v in value.items()}
        return value

    with torch.inference_mode(False):
        clone = copy.deepcopy(module)
        for m in clone.modules():
            for key, value in list(m._parameters.items()):
                if value is not None:
                    m._parameters[key] = torch.nn.Parameter(value.detach().clone(), requires_grad=False)
            for key, value in list(m._buffers.items()):
                if value is not None:
                    m._buffers[key] = value.detach().clone()
            for key, value in list(vars(m).items()):  # plain tensor attributes (caches, index maps)
                if not key.startswith('_') and isinstance(value, (torch.Tensor, list, tuple, dict)):
                    setattr(m, key, normal(value))
        if device is not None:
            clone = clone.to(device)  # inside the non-inference block, so the moved tensors stay usable
    return clone


def refit_pose(locked, reference, head, K, depth_factor, steps=150, chunk=256, prior=0.02, hand_weight=0.25, lr=0.02, person=0):
    """Re-aim the locked body's joints at SAM's own 2D keypoints (constant bones, plate kept).

    Locking the proportions changes limb lengths, so a wrist or ankle can leave the plate by a few percent of the
    image. Per frame, the global rotation and body pose (Euler parameters) are adjusted so the locked body's 70
    keypoints (body, hands, feet) reproject where SAM's own body did, with `prior` pulling toward SAM's angles.
    Hands, shape, scale and expressions stay fixed. depth_factor (F,) scales pred_cam_t to the depth used for the
    fit (the projection-preserving depth for the locked body). Returns (pose, report).
    """
    import torch
    frames = locked['frames']
    ref = [f[person] for f in reference['frames']]
    # The SAM weights may have been offloaded to the CPU after prediction; fit on the GPU when there is one.
    head = trainable_copy(head, 'cuda' if torch.cuda.is_available() else None)
    device = head.scale_mean.device
    out, before, after = [], [], []
    with torch.inference_mode(False), torch.enable_grad():
        Kt = torch.as_tensor(np.asarray(K, np.float32), device=device)  # created outside inference mode
        for start in range(0, len(frames), chunk):
            part = [f[person] for f in frames[start:start+chunk]]
            rp = ref[start:start+chunk]
            n = len(part)

            def stack(rows, key):
                return torch.tensor(np.stack([np.asarray(p[key], np.float32) for p in rows]), device=device)
            T = stack(rp, 'pred_cam_t')*torch.tensor(np.asarray(depth_factor[start:start+n], np.float32), device=device)[:, None]
            target3d = stack(rp, 'pred_keypoints_3d')+stack(rp, 'pred_cam_t')[:, None]
            target = target3d@Kt.T
            target = target[..., :2]/target[..., 2:3]
            rot0, pose0 = stack(part, 'global_rot'), stack(part, 'body_pose_params')
            rot = rot0.clone().requires_grad_(True)
            pose = pose0.clone().requires_grad_(True)
            fixed = dict(hand_pose_params=stack(part, 'hand_pose_params'), scale_params=stack(part, 'scale_params'),
                         shape_params=stack(part, 'shape_params'),
                         expr_params=stack(part, 'expr_params') if all(p.get('expr_params') is not None for p in part) else None)
            sign = torch.tensor([1., -1., -1.], device=device)
            scale_px = float(K[0][0])/100  # residuals in ~1% of focal length units

            def project(r, b):
                kp = head.mhr_forward(global_trans=torch.zeros_like(r), global_rot=r, body_pose_params=b, return_keypoints=True,
                                      **fixed)[1][:, :70].float()*sign
                p = (kp+T[:, None])@Kt.T
                return p[..., :2]/p[..., 2:3]
            # MHR70: 21-40 right hand and 42-61 left hand (wrists 41/62 count as body), so hands do not outvote the body.
            weight = torch.ones(70, device=device)
            weight[21:41] = hand_weight
            weight[42:62] = hand_weight
            weight = weight/weight.sum()
            optimizer = torch.optim.Adam([rot, pose], lr=lr)
            for step in range(steps):
                optimizer.zero_grad()
                error = ((project(rot, pose)-target)/scale_px).pow(2).sum(-1)
                loss = (error*weight).sum(-1).mean()+prior*((pose-pose0).pow(2).mean()+(rot-rot0).pow(2).mean())*100
                loss.backward()
                optimizer.step()
                if step == 0:
                    before.append(error.detach().sqrt().cpu().numpy()*scale_px)
            with torch.no_grad():
                after.append(((project(rot, pose)-target).pow(2).sum(-1)).sqrt().cpu().numpy())
                verts, kp, joints, model_params, rots = head.mhr_forward(
                    global_trans=torch.zeros_like(rot), global_rot=rot, body_pose_params=pose, return_keypoints=True,
                    return_joint_coords=True, return_model_params=True, return_joint_rotations=True, **fixed)
            flip = lambda x: (x.detach().float()*x.new_tensor([1., -1., -1.])).cpu().numpy()
            verts, kp, joints = flip(verts), flip(kp), flip(joints)
            model_params, rots = model_params.detach().float().cpu().numpy(), rots.detach().float().cpu().numpy()
            rot_np, pose_np = rot.detach().cpu().numpy(), pose.detach().cpu().numpy()
            for i, p in enumerate(part):
                q = dict(p)
                q.update(global_rot=rot_np[i], body_pose_params=pose_np[i], pred_vertices=verts[i], pred_keypoints_3d=kp[i, :70],
                         pred_joint_coords=joints[i], pred_global_rots=rots[i], mhr_model_params=model_params[i])
                if 'pred_face_keypoints_3d' in p:
                    q['pred_face_keypoints_3d'] = kp[i, 70:]
                out.append([q])
    before, after = np.concatenate(before), np.concatenate(after)
    report = dict(steps=steps, prior=prior, hand_weight=hand_weight, lr=lr, keypoint_error_px_before=dict(median=float(np.median(before)), p95=float(np.percentile(before, 95))),
                  keypoint_error_px_after=dict(median=float(np.median(after)), p95=float(np.percentile(after, 95))),
                  note='Locked-proportion pose re-aimed at SAM 2D keypoints (body, hands, feet); hands/shape/scale fixed.')
    return dict(locked, frames=out), report


def _visible(points_2d, width, height, margin=.02):
    m = margin*max(width, height)
    return (points_2d[..., 0] > m) & (points_2d[..., 0] < width-m) & (points_2d[..., 1] > m) & (points_2d[..., 1] < height-m)


def solve_placement(body, translation, sam_size, c2w, fps, ankles_2d, width, height, scale_free=True, cfg=None, extent_px=None):
    """Depth placement of a locked body on a known camera.

    body:        (F, 12, 3) locked joints in camera axes WITHOUT the root translation (pred_keypoints_3d subset)
    translation: (F, 3) SAM's camera-space root translation (pred_cam_t), metres
    sam_size:    (F,) body_size of SAM's own (unlocked) camera-space joints, frame by frame
    c2w:         (F, 4, 4) camera-to-world in scene units
    ankles_2d:   (F, 2, 2) SAM's 2D ankles (pixels), for contact visibility
    scale_free:  allow the global SAM-metres -> scene-units factor to move (needs planted feet; off for a static camera)
    extent_px:   (F,) how far the body reaches on screen from its root (pixels). Moving the body in depth changes its
                 apparent size, which shifts the extremities by about extent x size change; the size prior per frame is
                 therefore plate_budget x width / extent (capped by size_sigma): close-ups stay on the plate, distant
                 bodies get smoothed.
    Returns q (F,), s, contacts (F, 2), report. Camera-space placed points: s*(body + q[:, None, None]*translation[:, None]).
    """
    cfg = dict(DEFAULTS, **(cfg or {}))
    F = len(body)
    R, t = c2w[:, :3, :3], c2w[:, :3, 3]
    hip_rel = (body[:, 6]+body[:, 7])/2
    ankle_rel = body[:, list(ANKLES)]
    u0 = np.log(body_size(body)/sam_size)
    size_sigma = np.full(F, cfg['size_sigma'])
    if extent_px is not None:
        size_sigma = np.minimum(size_sigma, cfg['plate_budget']*width/np.maximum(np.asarray(extent_px, float), 1.))
    ray = translation/np.linalg.norm(translation, axis=1, keepdims=True)
    ray_world = np.einsum('fij,fj->fi', R, ray)
    visible = _visible(ankles_2d, width, height)
    dt = 1.0/fps
    # The global scale is only measurable when the camera travels a fair fraction of the subject distance
    # (planted feet then fix the parallax); for pans, tilts and locked-off shots it stays 1.
    travel = float(np.linalg.norm(np.ptp(t, axis=0)))/max(float(np.median(np.linalg.norm(translation, axis=1))), 1e-6)
    scale_free = scale_free and travel >= cfg['min_travel_ratio']

    def world(rel, q, s):  # rel (F, ..., 3) -> world (F, ..., 3)
        shape = (F,)+(1,)*(rel.ndim-1)
        cam = s*(rel+q.reshape(shape)*translation.reshape((F,)+(1,)*(rel.ndim-2)+(3,)))
        return np.einsum('fij,f...j->f...i', R, cam)+t.reshape((F,)+(1,)*(rel.ndim-2)+(3,))

    def residuals(x, contacts, acc_w, scale_sigma):
        u, ls = x[:F], x[F]
        q, s = np.exp(u), np.exp(ls)
        res = [(u-u0)/size_sigma]
        hip = world(hip_rel, q, s)
        acc = (hip[2:]-2*hip[1:-1]+hip[:-2])/dt**2
        # Only the depth component is free; lateral motion is fixed by the plate.
        res.append(acc_w*np.einsum('fi,fi->f', acc, ray_world[1:-1])/s/cfg['depth_acc_sigma'])
        if contacts.any():
            a = world(ankle_rel, q, s)
            vel = (a[1:]-a[:-1])/dt
            both = contacts[1:] & contacts[:-1]
            res.append((vel*both[..., None]).ravel()/s/cfg['contact_sigma'])
        res.append(np.array([ls/scale_sigma]))
        return np.concatenate(res)

    def sparsity(contacts):
        rows = F+(F-2)+(6*(F-1) if contacts.any() else 0)+1
        sp = lil_matrix((rows, F+1), dtype=int)
        sp[np.arange(F), np.arange(F)] = 1
        for i in range(F-2):
            sp[F+i, i:i+3] = 1
        sp[F:F+F-2, F] = 1
        if contacts.any():
            o = 2*F-2
            for i in range(F-1):
                sp[o+6*i:o+6*i+6, i:i+2] = 1
            sp[o:o+6*(F-1), F] = 1
        sp[rows-1, F] = 1
        return sp

    contacts = np.zeros((F, 2), bool)
    acc_w = np.ones(F-2)
    x = np.concatenate([u0, [0.]])
    rounds = 0
    for rounds in range(1, 6):
        measurable = scale_free and contacts.sum() >= cfg['min_scale_contacts']
        scale_sigma = cfg['scale_sigma'] if measurable else 1e-3
        # Optimise z = x + 1: SciPy's first trust region is |x0|, which is ~0 when the start is the identity (u = 0).
        result = least_squares(lambda z, *args: residuals(z-1., *args), x+1., jac_sparsity=sparsity(contacts), method='trf',
                               x_scale='jac', max_nfev=200, args=(contacts, acc_w, scale_sigma))
        x = result.x-1.
        x[F] = np.clip(x[F], np.log(cfg['min_scale']), np.log(cfg['max_scale']))
        q, s = np.exp(x[:F]), float(np.exp(x[F]))
        # Robust smoothness: frames whose depth acceleration stays large are real fast motion, not jitter.
        hip = world(hip_rel, q, s)
        acc = np.einsum('fi,fi->f', (hip[2:]-2*hip[1:-1]+hip[:-2])/dt**2, ray_world[1:-1])/s/cfg['depth_acc_sigma']
        new_w = 1/np.sqrt(1+(acc/cfg['acc_robust'])**2)
        a = world(ankle_rel, q, s)
        speed = np.zeros((F, 2))
        speed[1:-1] = np.linalg.norm(a[2:]-a[:-2], axis=-1)*fps/2/s
        speed[0], speed[-1] = speed[1], speed[-2]
        new_contacts = visible & (speed < cfg['contact_speed'])
        if (new_contacts == contacts).all() and np.allclose(new_w, acc_w, atol=.05):
            break
        contacts, acc_w = new_contacts, new_w

    def jitter_cm(q_, s_):
        """Hip distance from its own ~0.3 s smoothed path, in SAM centimetres."""
        from scipy.ndimage import gaussian_filter1d
        hip_ = world(hip_rel, q_, s_)
        return float(np.sqrt((np.linalg.norm(hip_-gaussian_filter1d(hip_, .15*fps, axis=0, mode='nearest'), axis=1)**2).mean())/s_*100)

    def skate(q_, s_):
        a_ = world(ankle_rel, q_, s_)
        v = np.linalg.norm(a_[1:]-a_[:-1], axis=-1)*fps/s_
        m = contacts[1:] & contacts[:-1]
        return float(v[m].mean()) if m.any() else None

    sam_q = np.exp(u0)  # what SAM itself implies for the locked body
    report = dict(global_scale=s, scale_free=bool(measurable), camera_travel_ratio=travel, rounds=rounds,
                  size_sigma_median=float(np.median(size_sigma)),
                  depth_factor_range=[float(q.min()), float(q.max())],
                  apparent_size_change_p95=float(np.percentile(np.abs(q/sam_q-1), 95)),
                  contact_frames=contacts.sum(0).tolist(), ankle_visible_frames=visible.sum(0).tolist(),
                  fast_motion_frames=int((acc_w < .5).sum()),
                  jitter_cm_before=jitter_cm(sam_q, 1.), jitter_cm_after=jitter_cm(q, s),
                  planted_foot_speed_before=skate(sam_q, 1.), planted_foot_speed_after=skate(q, s),
                  optimizer=dict(cost=float(result.cost), status=int(result.status)), settings=cfg)
    return q, s, contacts, report
