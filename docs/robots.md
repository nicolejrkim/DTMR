# Supported robots

A robot is a descriptor file in [`src/dtmr/robots/`](../src/dtmr/robots): the model, PD gains,
which bodies are feet, knees and hands, and named joint groups. A run config names one
with `robot:` and holds only tuning, so switching embodiment is one line. Each robot
ships one release config, its recommended recipe:

| Robot | `robot:` | DoF | Release config |
|---|---|---|---|
| Unitree G1 | `unitree_g1` | 29 | [`unitree_g1_release_warp.yaml`](../src/dtmr/examples/unitree_g1_release_warp.yaml) |
| Unitree H1-2 | `unitree_h1_2` | 27 | [`unitree_h1_2_release_warp.yaml`](../src/dtmr/examples/unitree_h1_2_release_warp.yaml) |
| Unitree R1 | `unitree_r1` | 24 | [`unitree_r1_release_warp.yaml`](../src/dtmr/examples/unitree_r1_release_warp.yaml) |
| Booster T1 | `booster_t1` | 23 | [`booster_t1_release_warp.yaml`](../src/dtmr/examples/booster_t1_release_warp.yaml) |
| Booster K1 | `booster_k1` | 22 | [`booster_k1_release_warp.yaml`](../src/dtmr/examples/booster_k1_release_warp.yaml) |
| AgiBot X2 Ultra | `agibot_x2_ultra` | 31 | [`agibot_x2_ultra_release_warp.yaml`](../src/dtmr/examples/agibot_x2_ultra_release_warp.yaml) |
| AgiBot A3 T3.0 | `agibot_a3` | 31 | [`agibot_a3_release_warp.yaml`](../src/dtmr/examples/agibot_a3_release_warp.yaml) |

A release config runs any clip of its robot with `--motion_path`;
[Configuration](configuration.md) lists the keys.

## Model licences

Robot descriptions keep their vendors' licences, inside each `src/dtmr/models/<robot>/`:
Unitree G1, H1-2 and R1 (BSD-3-Clause, Unitree Robotics), Booster T1 (MuJoCo
Menagerie, Apache-2.0), Booster K1 (BSD-3-Clause, Booster Robotics), AgiBot X2 Ultra
and A3 T3.0 (Mulan PSL v2, AgiBot; visual meshes of the X2 decimated to 10 % for the
package).

A robot that is not listed is a model directory and one descriptor file away:
[Adding a robot](ADDING_A_ROBOT.md).
