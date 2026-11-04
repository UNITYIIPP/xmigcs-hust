# SUPERMIMIC EVT2 motion tracking

The default configuration loads `model/actor_0100000.onnx`, its matching
`model/actor_0100000.json`, and `motion/run1_subject5.npz`.
No training repository import is required. The keyboard `v` command selects
SUPERMIMIC from STANDZERO, SWR, or AMPMLP. From STOP, press `z`, then `v`.

## Format-2 actor exports (default)

The actor takes float32 `obs_history [B,990]` and `goal [B,730]`, and returns
raw `action [B,29]`. Normalizers are embedded in the ONNX graph. Joint names,
defaults, action scales, PD gains, action gains and reference masks are read
from the matching `.json` file (replace `.onnx` with `.json`). Missing or
incompatible metadata is rejected during state construction.

The 10-frame history is oldest first. Each 99-value observation contains:
joint position minus default (29), joint velocity (29), pelvis gyro (3),
projected gravity (3), previous raw action (29), and
`R_robot(t).T @ R_reference(t)` in column-major 6D (6). The first observation
fills all history slots; later slots retain their original reference timestamp.
The optional heading alignment rotates reference orientations to the robot's
initial heading before computing this feedback.

Goals use future frames t+1..t+10. Each 73-value frame contains:

1. Future reference position minus current reference position, in current reference heading (3).
2. Future reference rotation in current reference heading, column-major 6D (6).
3. Reference joint positions (29).
4. Reference joint velocities, without the legacy 0.1 factor (29).
5. Root-local linear and angular velocities (3+3).

Column-major 6D is `[R00,R10,R20,R01,R11,R21]`. Future indices beyond the clip
end repeat the entire final frame, including its recorded velocities, matching
the actor training library. The target equation follows the sidecar:

```text
q_target = q0 + action_scale * action_gain * action
           + reference_mask * (q_ref(t+1) - q0)
```

For this checkpoint the 12 leg joints use absolute targets around q0; the
17 waist/arm joints use the next reference pose plus the policy residual.
Desired joint velocity and feedforward torque are zero. The observation always
stores the previous raw actor output, before reference offsets are applied.

`goal_encoding: auto` selects `current_ref` for this interface and
`reference_relative` for the legacy interface below. An explicit encoding that
does not match the selected model is rejected.

## Legacy embedded SuperMimic exports

To use the previous model, set `model_path: policydexevt.onnx` and keep
`goal_encoding: auto`. The legacy parameter and feature conventions are preserved.

Joint order, default posture, action scales, PD gains and the policy control period
are read exclusively from `supermimic_interface` metadata embedded inside the ONNX.
PD gains are looked up by joint name in `robot_config.joints` and then scattered into
motor order. Missing/inconsistent metadata is rejected; re-export older checkpoints
with `tools/export_onnx.py`. YAML values and JSON sidecars cannot override model parameters.
The YAML retains model/reference paths, FSM scheduling, entry/IMU settings and fixed
head-joint settings for the two motors outside the 29-joint policy.
ONNX input names `observation_history` and `obs_history` are both supported.
History length and future horizon are read from the graph. The current model uses
`[B,10,99]` history and `[B,730]` goals; input normalizers remain embedded.
The control node calls this state every configured `dt` seconds. Inference,
observation history and reference advancement run once per `control.decimation`
calls, with joint targets held between inferences. Keep
`dt * control.decimation = 0.02` seconds to match the 50 Hz checkpoint and clip:
for example, `0.02 * 1`, `0.005 * 4` (the default), or `0.0025 * 8`.
These settings do not change the simulator's physics integration step.
The global `dex_config.yaml` `dt` separately controls motor-command transmission.
Warm-start interpolation runs every FSM tick for the configured duration;
entry and reentry start inference immediately after warmup, independent of
the global control clock.

The 99 observation entries are joint position minus default (29), joint velocity
(29), pelvis-frame angular velocity (3), projected gravity (3), previous raw
policy action (29), and current reference orientation relative to the pelvis (6).
History is oldest first; the first observation repeats across the initial history.
There is no extra normalization, scaling or finite-range clipping.

Goals use the next 1..K reference frames, with row-major 6D rotations
`[R00,R01,R10,R11,R20,R21]` and feature order:

1. Future reference root position minus current reference root in current reference heading (3).
2. Future reference root orientation in that heading frame (6).
3. Future root-local linear/angular velocities (3+3).
4. Future joint positions (29).
5. Future joint velocities multiplied by 0.1 (29).

Only frames beyond the clip end have their velocities set to zero. Actual
last-frame recorded velocities remain unchanged. Compact exports/libraries and
prepared `root_p/root_quat/root_v/root_w/q/qd` clips are supported. Joints are
reordered by name; select `motion_clip` explicitly for a multi-clip library.

Joint targets are `default_joint_positions + action_scales * action`; desired
velocity and feedforward torque are zero. The previous observation stores the
raw action. Targets/gains are scattered into the motor order by joint name.
At clip completion the state requests STOP and holds the last target while the
FSM processes the transition, without repeating the final inference.

## Reference and entry settings (both interfaces)

Set `motion_path` to one of the copied clips:

- `walk1_subject1.npz`
- `run1_subject5.npz`
- `walk_forward_amateur_001__A001.npz`

The keyboard `v` command continues to select SUPERMIMIC through the existing FSM.
`control.start_frame` selects the initial frame. `warm_start_time` optionally
interpolates joint pose before advancing the reference (currently 0.0).
Reentry clears history, actions, phase and completion state.

`control.imu_frame: pelvis` means incoming orientation and gyro are already
expressed for the pelvis. Use `imu_waist_link` only if the driver supplies raw
sensor-frame measurements; the existing URDF mounting rotation is then applied.
`align_reference_heading: true` rotates the reference heading to the initial
robot heading, without changing the reference-relative future goals.

## Offline verification

No hardware interfaces or motor publishers are started by these tests. They
exercise the real default actor, independently constructed observations/goals,
residual targets, history/reset, decimation, clip completion, metadata rejection,
and the original embedded model:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s test -p test_supermimic_deploy.py -v
```

For the legacy embedded model's full MuJoCo parity testing, from
`/home/mig/dou/supermimic_vibed` (select `policydexevt.onnx` first):

```bash
bash tools/python.sh tools/check_deploy_parity.py \
  --deploy-root /home/mig/dou/xmigcs-hust \
  --report logs/run/xmigcs_deploy_parity.json
```

The parity test feeds native MuJoCo sensor states into the deployment FSM and
compares every observation, future goal, action and joint target against the
standalone runner. MuJoCo is actually driven by the FSM actions. Tests disable
pose warmup because MuJoCo is already reset to the reference pose and velocities.
The default XML for this comparison is the original `evt2.xml`.

On 2026-10-29, all three complete clips passed the height/tilt/numerical criteria.
Maximum goal difference was zero, maximum observation difference was below
`5e-7`, and maximum action difference was below `3e-6`. Native simulation still
showed global position drift; completing a clip is not a zero-error trajectory.

Actual ROS simulation or hardware additionally depends on its robot geometry,
contact model, servo/friction settings, sensor coordinate conventions and timing.
Pose interpolation does not reproduce the nonzero reference velocities installed
by native sim2sim reset. These offline tests validate the policy/control contract;
they do not establish identical real-world startup or closed-loop dynamics.
