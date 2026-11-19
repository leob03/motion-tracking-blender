# Licences by pipeline path (0.8.0)

Summary of the licence review of 19 November 2026, read from each project's own licence files and model cards. It is a working summary for production planning, not legal advice; have the studio confirm before delivery work.

## Default path: SAM 3D Body on camera, body and fingers

| Component | Role | Licence | Commercial use |
| --- | --- | --- | --- |
| SAM 3D Body weights (Comfy-Org conversion) | Body, fingers, face keypoints | SAM License (Meta) | Yes, royalty-free; Meta claims nothing in outputs |
| MHR | Body model, rig, head mask | Apache-2.0 | Yes |
| D-FINE-L (`ustc-community/dfine-large-coco`) | Actor detection boxes | Apache-2.0 | Yes |
| GeoCalib | Locked-off lens estimate | Apache-2.0 | Yes |

SAM License conditions: include the licence text when redistributing the weights; no military, ITAR, nuclear, espionage or weapons use; no reverse engineering; acknowledge SAM in publications. The licence ends if you sue Meta over it, and Meta may amend it.

## Face options

| Option | Components | Status |
| --- | --- | --- |
| SAM native face | SAM 3D Body expressions (MediaPipe face components inside the SAM checkpoint) | Commercially usable, as above |
| RealDenseFace + FLAME head | FLAME 2023 Open (CC-BY-4.0), SAM face keypoints, own landmark file, RealDenseFace | Head model and placement are commercially usable since 0.8.0. **Open question:** RealDenseFace code and weights are MIT, but the weights were trained on Ava-256 (CC-BY-NC-4.0) and FLAME 2023 fits, and its `flame_assets.npz` (FLAME UV layout) ships with them |

FLAME 2023 Open conditions: credit FLAME (see [THIRD_PARTY.md](../THIRD_PARTY.md)); results carry the credit line in `face_details.credit` and `face_editing.credit`. Its licence forbids pornographic content and fake, libellous, misleading or defamatory content.

FLAME 2023 Open is FLAME 2023 with a different expression basis: identical mesh, identity basis, joints and skin weights. Results look the same as with FLAME 2023 (head placement within 2.4 mm, mouth opening correlation 0.99 on the two test shots). Captures fitted before 0.8.0 used non-commercial FLAME 2023 and still replay with it when upgraded.

## Fallback: GVHMR

GVHMR (Zhejiang University) is licensed for research and non-profit use only; commercial use needs a licence from the authors. It also uses SMPL-X (non-commercial; commercial licences through Meshcapade). The `hybrid` and `gvhmr` methods are therefore not commercially usable as they stand.

## Removed in 0.8.0

| Component | Licence | Was used for |
| --- | --- | --- |
| Ultralytics YOLOv8m-seg | AGPL-3.0 | Actor detection (replaced by D-FINE-L) |
| FLAME 2023 | MPI non-commercial | Face head model (replaced by FLAME 2023 Open) |
| SMPL-X landmark set | MPI non-commercial | Placing the FLAME head on SAM's body (replaced by SAM's own face keypoints) |
| SMIRK `landmark_embedding.npy` | Derived from FLAME | Landmarks on the FLAME head (replaced by our own file) |

D-FINE-L was checked against YOLO on five shots through the full pipeline: plate error within 0.2 px, depth jitter within 0.3 cm, and meshes within YOLO's own run-to-run differences.
