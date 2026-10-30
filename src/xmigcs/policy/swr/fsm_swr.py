"""
FSM State: SWR (Whole-Body Walking)

Observation layout (47-dim):
  ang_vel(3) + projected_gravity(3) + command(3) + amp_condition_onehot(6) +
  qj(12) + dqj(12) + last_actions(12) + [1]
"""
import numpy as np
import os
from typing import Optional

from xmigcs.policy.onnx_policy_base import ONNXPolicyBase
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData
from xmigcs.common.rotation import euler_to_quaternion_scipy, quat_rotate_inverse_numpy
from xmigcs.common.joint_mapping import MJ_JOINTS_31


class FSMStateSWR(ONNXPolicyBase):
    """MLP策略状态实现 - 与C++版本完全一致"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = self._get_config_path()
        cfg = self._load_config(config_path)
        self._init_extra_config(cfg)
        self._init_buffers()
        self._init_onnx_and_joints(cfg)

    def _get_config_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "config", "swr.yaml")

    def _init_extra_config(self, cfg: dict) -> None:
        """Initialize SWR-specific config."""
        ctrl_cfg = cfg.get('control', {})
        self.action_scale_ = ctrl_cfg.get('action_scale')
        self.warm_start_time_ = ctrl_cfg.get('warm_start_time', 0.3)
        self._warm_start_pose = np.zeros(self.motor_num_, dtype=np.float32)

        self.is_first_step_ = True
        step = (self.decimation_ if self.decimation_ else 1) * self.dt_
        if self.warm_start_time_ > 0 and step > 0:
            self._warm_start_steps = max(1, int(self.warm_start_time_ / step))
        else:
            self._warm_start_steps = 0
        self._warmup_inference_counter = 0

    def _init_onnx_and_joints(self, cfg: dict) -> None:
        """Load ONNX model and build joint mapping."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "model")
        self._init_onnx_session(model_dir, cfg["model_path"])
        self._parse_onnx_metadata()
        self._build_joint_mapping(MJ_JOINTS_31, cfg.get('fixed_joints', []))

    # ── State lifecycle ──

    def _reset_internal_state(self) -> None:
        super()._reset_internal_state()
        self.is_first_step_ = True
        self._warmup_inference_counter = 0

    def on_enter(self):
        self._reset_internal_state()
        self._warmup_inference_counter = 0
        if self.robot_data_ is not None:
            try:
                self._warm_start_pose = self.robot_data_.get_joint_pos().copy()
            except Exception:
                self._warm_start_pose.fill(0.0)
        else:
            self._warm_start_pose.fill(0.0)

    # ── Main run loop ──

    def run(self, flag: ControlFlag):
        """运行MLP状态 - 与C++版本完全一致"""
        if int(self.robot_data_.time_now_ / self.dt_) % self.decimation_ == 0:
            self.compute_observation(flag)
            self.compute_actions()

            target_dof_pos_lab = self.actions_ * self.action_scale_lab + self.default_angles_lab
            target_dof_pos_mj = self.robot_data_.get_joint_pos().copy()
            target_dof_pos_mj[self.lab2mj] = target_dof_pos_lab
            commanded_pos = target_dof_pos_mj

            base = self.robot_data_.q_d_.shape[0] - self.motor_num_
            self.robot_data_.q_d_[base:base + len(self.joint_xml)] = commanded_pos
            self.robot_data_.q_dot_d_[base:base + len(self.joint_xml)] = 0.0
            self.robot_data_.tau_d_[base:base + len(self.joint_xml)] = 0.0

            self.last_actions_[:] = self.actions_

        self._apply_pd_gains()
        self._apply_fixed_joints()

    def _apply_pd_gains(self):
        """Set PD gains for all joints (fixed joints overridden by _apply_fixed_joints)."""
        self.robot_data_.joint_kp_p_[:len(self.joint_xml)] = self.stiffness_array
        self.robot_data_.joint_kd_p_[:len(self.joint_xml)] = self.damping_array

    def _apply_action_to_robot(self) -> None:
        """Applied inline in run() due to SWR-specific logic."""
        pass

    # ── Observation ──

    def compute_observation(self, flag: ControlFlag):
        roll, pitch, yaw = (
            float(self.robot_data_.imu_data_[2]),
            float(self.robot_data_.imu_data_[1]),
            float(self.robot_data_.imu_data_[0]),
        )
        quat_wxyz = euler_to_quaternion_scipy(roll, pitch, yaw)
        q_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float32)
        gravity_init = quat_rotate_inverse_numpy(
            q_xyzw, np.array([0., 0., -1.], dtype=np.float32))

        x_speed_command, y_speed_command, yaw_speed_command = self.robot_data_.get_walk_cmd()
        command = np.array([
            x_speed_command, y_speed_command, yaw_speed_command
        ], dtype=np.float32)

        gyro = np.array([
            self.robot_data_.imu_data_[3], self.robot_data_.imu_data_[4],
            self.robot_data_.imu_data_[5]
        ], dtype=np.float32) * self.ang_vel_scale_

        q_mj = self.robot_data_.get_joint_pos()
        dq_mj = self.robot_data_.get_joint_vel()
        ang_vel = self.robot_data_.get_angular_velocity()

        qj = q_mj[self.lab2mj]
        dqj = dq_mj[self.lab2mj]
        qj = qj - self.default_angles_lab

        # AMP condition one-hot (priority rules)
        amp_condition_onehot = self._compute_amp_condition(command)

        proprio = np.concatenate([
            ang_vel,
            gravity_init,
            command,
            amp_condition_onehot,
            qj,
            dqj,
            self.last_actions_,
            [1],
        ])

        self._update_proprio_history(proprio)

    @staticmethod
    def _compute_amp_condition(command: np.ndarray) -> np.ndarray:
        """Compute AMP condition one-hot vector from command."""
        vx = float(command[0])
        vy = float(command[1])
        vyaw = float(command[2])
        v_thresh = 0.2
        w_thresh = 0.2
        run_thresh = 1.4

        abs_vx = abs(vx)
        abs_vy = abs(vy)
        abs_vyaw = abs(vyaw)

        cond_6_run = vx > run_thresh
        cond_1_fwd = (vx > v_thresh) and (not cond_6_run)
        cond_2_bwd = vx < -v_thresh
        cond_3_lat = (abs_vx < v_thresh) and (abs_vy > v_thresh)
        cond_4_yaw = (abs_vx < v_thresh) and (abs_vy < v_thresh) and (abs_vyaw > w_thresh)
        cond_5_idle = not (cond_1_fwd or cond_2_bwd or cond_3_lat or cond_4_yaw or cond_6_run)

        onehot = np.zeros(6, dtype=np.float32)
        onehot[0] = 1.0 if cond_1_fwd else 0.0
        onehot[1] = 1.0 if cond_2_bwd else 0.0
        onehot[2] = 1.0 if cond_3_lat else 0.0
        onehot[3] = 1.0 if cond_4_yaw else 0.0
        onehot[4] = 1.0 if cond_5_idle else 0.0
        onehot[5] = 1.0 if cond_6_run else 0.0
        return onehot

    def on_exit(self):
        """退出MLP状态"""
        if getattr(self, "obs_log_file", None) is not None:
            try:
                self.obs_log_file.flush()
                self.obs_log_file.close()
            except Exception as e:
                pass
            self.obs_log_file = None
