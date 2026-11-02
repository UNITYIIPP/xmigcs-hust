"""ALLMIMIC actor: current-reference goals and hybrid residual joint targets."""

import json
import os
from pathlib import Path

import numpy as np

from xmigcs.common.joint_mapping import MJ_JOINTS_29
from xmigcs.policy.whole_body_control.fsm_whole_body_control import FSMStateWHOLE_BODY_CONTROL
from xmigcs.policy.whole_body_control.reference_window import joint_permutation, rotation
from xmigcs.utils.logging_utils import get_logger

_logger = get_logger(__name__)


def encode_reference_goal(current, future):
    """Current-reference-heading position/rotation, column-major 6D, raw qd."""
    yaw = np.arctan2(current['rot'][1, 0], current['rot'][0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    heading_rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    delta = (future['pos'] - current['pos']) @ heading_rot
    relative = heading_rot.T @ future['rot']
    # Training mat_to_6d concatenates first column, then second column.
    rot6 = relative[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    v = np.einsum('kji,kj->ki', future['rot'], future['v'])
    w = np.einsum('kji,kj->ki', future['rot'], future['w'])
    return np.concatenate((delta, rot6, future['q'], future['qd'], v, w), axis=-1).reshape(-1).astype(np.float32)


class FSMStateALLMIMIC(FSMStateWHOLE_BODY_CONTROL):
    """Reuse U reference sources/timing; actor inputs and setpoints follow export JSON."""

    def __init__(self, robot_data, config_path=None):
        config_path = config_path or os.path.join(os.path.dirname(__file__), 'config', 'allmimic.yaml')
        self._policy_dir = os.path.dirname(os.path.abspath(__file__))
        super().__init__(robot_data, config_path)
        self.num_obs = 93
        self.obs = np.zeros(93, dtype=np.float32)
        self.history = np.zeros((self.history_length, 93), dtype=np.float32)
        self._reset_tracking()

    def _configure_policy(self):
        inputs = {i.name: i for i in self.ort_session.get_inputs()}
        outputs = self.ort_session.get_outputs()
        metadata_path = Path(self.onnx_path).with_suffix('.json')
        with metadata_path.open() as f:
            meta = json.load(f)
        task = meta['task_config']
        offsets = meta['future_frame_offsets']
        self.history_length = int(meta['history_frames'])
        self.goal_frames = len(offsets)
        if (meta.get('model_file') != Path(self.onnx_path).name or meta.get('robot') != 'evt'
                or task.get('goal_pos') != 'current_ref' or task.get('goal_rot') != 'current_ref'
                or task.get('heading') != 'xaxis'
                or task.get('rot_repr') != '6d' or task.get('action') != 'res'
                or task.get('res_legs') != 'abs' or self.history_length < 1
                or offsets != list(range(1, self.goal_frames + 1)) or not self.goal_frames
                or not np.isclose(float(task['dt']), .02) or not np.isclose(float(meta['control_frequency_hz']), 50.)):
            raise ValueError('Unsupported or mismatched ALLMIMIC export metadata')
        shapes = {name: x.shape for name, x in inputs.items()}
        if (set(inputs) != {'obs_history', 'goal'}
                or shapes['obs_history'][1:] != [self.history_length * 93]
                or shapes['goal'][1:] != [self.goal_frames * 73]
                or any(len(x.shape) != 2 or x.type != 'tensor(float)' or
                       (isinstance(x.shape[0], int) and x.shape[0] != 1) for x in inputs.values())
                or len(outputs) != 1 or outputs[0].name != 'action' or len(outputs[0].shape) != 2
                or outputs[0].shape[1] != 29 or outputs[0].type != 'tensor(float)'):
            raise ValueError(f'Unexpected ALLMIMIC ONNX inputs or output: {shapes}')
        control = meta['control']
        self.joint_seq = list(control['joint_names'])
        self.joint_xml = list(MJ_JOINTS_29)
        self.mj2lab = joint_permutation(self.joint_xml, self.joint_seq)
        self._motor_indices = self.locked_joint_map[self.mj2lab]
        rows = control['actuators']
        by_name = {r['joint']: r for r in rows}
        if len(rows) != 29 or len(by_name) != 29 or set(by_name) != set(self.joint_seq):
            raise ValueError('ALLMIMIC actuator names must match all 29 policy joints')
        self.default_angles_lab = np.asarray(control['default_joint_positions_rad'], dtype=np.float32)
        self.action_scale_lab = np.asarray(control['action_scale_rad'], dtype=np.float32)
        self.action_gain = np.asarray(control['policy_action_gain'], dtype=np.float32)
        self.reference_mask = np.asarray(control['reference_mask'], dtype=np.float32)
        leg_mask = np.asarray(control['leg_mask'], dtype=bool)
        self.stiffness_array_seq = np.asarray([by_name[n]['stiffness'] for n in self.joint_seq], dtype=np.float32)
        self.damping_array_seq = np.asarray([by_name[n]['damping'] for n in self.joint_seq], dtype=np.float32)
        effort = np.asarray([by_name[n]['effort_limit'] for n in self.joint_seq], dtype=np.float32)
        arrays = (self.default_angles_lab, self.action_scale_lab, self.action_gain, self.reference_mask,
                  self.stiffness_array_seq, self.damping_array_seq, effort)
        if (any(a.shape != (29,) or not np.isfinite(a).all() for a in arrays) or leg_mask.shape != (29,)
                or np.any(self.action_scale_lab <= 0) or np.any(effort <= 0)
                or np.any(self.stiffness_array_seq < 0) or np.any(self.damping_array_seq < 0)
                or not np.array_equal(self.reference_mask, (~leg_mask).astype(np.float32))
                or not np.allclose(self.action_gain[leg_mask], 1.)
                or not np.allclose(self.action_gain[~leg_mask], float(task['res_scale']))):
            raise ValueError('Invalid or inconsistent ALLMIMIC control parameters')
        def motor_order(values):
            result = np.empty(29, dtype=np.float32)
            result[self._motor_indices] = values
            return result
        self.joint_pos_array = motor_order(self.default_angles_lab)
        self.stiffness_array = motor_order(self.stiffness_array_seq)
        self.damping_array = motor_order(self.damping_array_seq)
        self.effort_limits_motor = motor_order(effort)

    def on_enter(self):
        self._reset_tracking()
        if self.ref_source == 'gmr':
            self._collect_gmr_reference()
        _logger.info('[ALLMIMIC] %s, ref=%s, history=%d, future=%d, policy=50 Hz',
                     os.path.basename(self.onnx_path), self.ref_source, self.history_length, self.goal_frames)

    def inner_run(self):
        window = self._reference_window()
        if self.ref_source == 'gmr':
            self._maybe_warn_missing()
        if window is None:
            return
        current, future, _ = window
        self.current_reference = current
        if self._alignment_ticks < 2:
            # Retain the existing two reference ticks on entry; no spatial alignment is needed.
            self._alignment_ticks += 1
        else:
            sensor_rot = rotation(self.robot_data_.get_robot_quat())
            robot_rot = sensor_rot @ self._sensor_in_pelvis.T
            q = self.robot_data_.get_joint_pos()[self._motor_indices]
            qd = self.robot_data_.get_joint_vel()[self._motor_indices]
            gyro = self._sensor_in_pelvis @ self.robot_data_.get_angular_velocity()
            self.obs = np.concatenate((q-self.default_angles_lab, qd, gyro,
                                       robot_rot.T @ [0, 0, -1], self.action)).astype(np.float32)
            if not np.isfinite(self.obs).all():
                raise ValueError('Non-finite ALLMIMIC observation')
            if not self._history_initialized:
                self.history[:] = self.obs
                self._history_initialized = True
            else:
                self.history[:-1] = self.history[1:]
                self.history[-1] = self.obs
            self.future_goal = encode_reference_goal(current, future)
            action = np.asarray(self.ort_session.run(['action'], {'obs_history': self.history.reshape(1, -1),
                                                                 'goal': self.future_goal[None]})[0], dtype=np.float32)
            if action.shape != (1, 29) or not np.isfinite(action).all():
                raise ValueError('ALLMIMIC must produce finite action [1,29]')
            self.action = action[0].copy()
            target = (self.default_angles_lab + self.action_scale_lab * self.action_gain * self.action
                      + self.reference_mask * (future['q'][0] - self.default_angles_lab))
            if self.tau_limit_enabled:
                target = self._apply_tau_limit_lab(target)
            self.robot_data_.q_d_[6 + self._motor_indices] = target
            self.robot_data_.q_dot_d_[6 + self._motor_indices] = 0.
            self.robot_data_.tau_d_[6 + self._motor_indices] = 0.
        self.counter_step += 1
        if self.ref_source == 'npz':
            self._npz_frame += 1
