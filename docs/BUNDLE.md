# Result bundle v1 / v2

The downloaded `.mocap.zip` is data. The artist importer never runs downloaded Python or loads uploaded pickles. `read_bundle` validates schema, array shapes, finite values, topology, skin weights and hierarchy before creating scene objects.

Members:

- `manifest.json`: `schema="mocap-blender-bundle"`, `version=1` for legacy/body results or `version=2` for editable FLAME, method, FPS/dimensions/frame count, source start, track ID, coordinate notes, inference options/provenance and timings.
- `animation.npz`: arrays listed below; loaded with `allow_pickle=False`.
- `actor_preview.jpg`, `tracking.json`: selected first-frame actor and tracked boxes.
- `face_parameters.npz`: optional SAM expression parameters or RealDenseFace fitted FLAME parameters. The worker uses retained FLAME parameters to construct editable controls; the artist importer uses the validated numeric editing arrays.

The synthetic test omits preview/tracking. Older v1 bundles can lack native rig fields and import as caches. Unexpected ZIP members, duplicate names and archives/arrays over the import size limits are rejected.

## Main arrays

Let F=frames, V=vertices, J=body landmarks, B=bones, I=skin influences per vertex.

| Array | Shape / meaning |
| --- | --- |
| `camera_to_world` | F×4×4; native GVHMR-derived mapping from OpenCV camera to inferred world |
| `actor_camera_to_world` | optional F×4×4; compensating camera preserving this hybrid actor's projection |
| `K` | F×3×3 camera intrinsics in processing-image pixels |
| `source_frames` | F zero-based original-video frame indices |
| `joints_world` | F×J×3; original body landmarks, retained for validation; no separate Blender marker objects |
| `vertices_world` | F×V×3; captured surface in inferred world meters |
| `faces` | triangles with valid vertex indices |
| `joints_projected_pixels` | landmark pixels through the native camera |
| `actor_joints_projected_pixels` | hybrid landmark pixels through the actor-fit camera |
| `rig_names`, `rig_parents` | B native names and parent indices, root parent -1; parents precede children |
| `rig_rest` | B×4×4 rest transforms |
| `rig_pose` | F×B×4×4 captured global bone transforms |
| `rig_indices`, `rig_weights` | V×I skin indices and normalized weights |
| `rig_vertices` | F×V×3 surface before skinning, inverse-fitted to reproduce the captured mesh |

World is right-handed **Y-up, meters**. Camera coordinates are OpenCV (X right, Y down, Z forward). Blender maps world `(x,y,z)` to `(x,-z,y)`, and flips the local camera Y/Z axes. Intrinsics include principal-point offsets and fixed pixel-aspect ratio. No image-undistortion/lens-distortion solve is encoded.

Hybrid placement replaces the SAM hip midpoint XYZ and root orientation with GVHMR's; local SAM articulation stays. The native camera and actor-fit camera have different meanings. Combining independently processed actors into a shared recovered scene is **not** guaranteed by this bundle.

Blender timeline frame `start+i` corresponds to source frame `source_start_frame+i`. The original movie background starts at `start-source_start_frame`. Frame rate uses Blender FPS/FPS-base to preserve fractional rates. Focal estimates are in the processed image; user focal input is scaled from original-video pixels.

The editable armature applies to a captured pre-skin surface cache, preserving original body correctives and facial shape in the unedited result. Manual bone edits do not rerun the learned model. Version 2 adds four FLAME bones, blends their influence into the original head/neck weights, and subtracts predicted expressions from the pre-skin cache. Animated expression shape keys add them back. This preserves the captured fit while allowing native edits. The joined lower neck has zero facial-edit influence.

Limits: the prototype uses a 2 GiB client download cap and importer expanded-size checks. Long sequences need a different cache/streaming design before relying on this format for arbitrary shot lengths.

## Editable FLAME extension (v2, add-on 0.4.0+)

Old add-ons reject version 2 rather than silently importing an incomplete face. The new importer still accepts legacy version 1. All five arrays below and `manifest.face_editing.version=1` are required and validated before creating objects.

| Array | Shape / meaning |
| --- | --- |
| `flame_expression_dirs` | E×H×3 identity-matched expression directions in rig rest space, with neck fade |
| `flame_expressions` | F×E captured coefficients |
| `flame_vertex_offset` | scalar integer; H FLAME vertices occupy the end of the joined mesh |
| `flame_neck_mask` | H correction influences, 0 at the protected lower neck and 1 on the upper face |
| `flame_bone_quaternions` | F×3×4 captured jaw/left-eye/right-eye local quaternions, WXYZ |

`rig_vertices` in v2 is the residual surface: add expression directions times coefficients before skinning. Blender uses relative, linearly keyed frame residuals and expression keys. `vertices_world` remains the complete unedited captured surface, including when importing without an armature. Captured face controls are also stored in the Blender mesh for the restore-frame operator. The resulting animation evaluates without add-on handlers or model files.
