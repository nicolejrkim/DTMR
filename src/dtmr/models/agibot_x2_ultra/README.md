# AgiBot X2 Ultra (31 dof) tracking model

Source: AgibotTech/agibot_x2_urdf, `X2_URDF-v1.3.0/x2_ultra.xml` @ 575cc6b (official MuJoCo description; the
URDF twin `x2_ultra.urdf` carries the same joints, ranges and effort limits). License: Mulan Permissive Software
License v2 (`LICENSE`, copied from the source repository).

`x2_ultra_mjx_torque.xml` keeps the official names, kinematics, inertials, joint ranges and per-joint armature (0.03) verbatim;
the mesh collisions are replaced by named primitives (torso, head, upper arm, forearm+hand, thigh, shin, foot box) with
contype/conaffinity 0 and an explicit `<pair>` list in `scene_mjx_torque_tracking.xml`; torque motors with ctrlrange = the
official effort limits. Joint order (qpos): left leg (6), right leg (6), waist yaw/pitch/roll, left arm (7), right arm (7), head (2).
The visual meshes under `assets/` are decimated to ~10 % of the official triangle counts (visual only; the physics uses the primitives).
