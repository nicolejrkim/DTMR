# AgiBot A3 T3.0 (31 dof) tracking model

Source: AgibotTech/A3-A3U-robot-model, `a3_t3d0/mjcf/a3_t3d0.xml` @ 589f508 (official MuJoCo description of the A3 T3.0 base
version; the URDF twin `a3_t3d0/urdf/model.urdf` carries the same 31 joints, ranges and effort limits). License: Mulan Permissive
Software License v2 (`LICENSE`, copied from the source repository).

`a3_t3d0_mjx_torque.xml` keeps the official names, kinematics, inertials and joint ranges verbatim (armature 0.005, the T1 class
default, since the official model publishes none); the mesh collisions are replaced by named primitives (torso, head, upper arm,
forearm+hand, thigh, shin, foot box) with contype/conaffinity 0 and an explicit `<pair>` list in `scene_mjx_torque_tracking.xml`;
torque motors with ctrlrange = the official effort limits. Joint order (qpos): waist yaw/roll/pitch, head (2), left arm (7),
right arm (7), left leg (6), right leg (6) - legs LAST. Link names are capitalised `*_Link` except `pelvis_link`.
