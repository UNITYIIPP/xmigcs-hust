"""FSM state for the two-input SuperMimic EVT2 ONNX motion tracker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from xmigcs.common.joint_mapping import MJ_JOINTS_29, MJ_JOINTS_31
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData
from xmigcs.common.rotation import matrix_from_quat
from xmigcs.policy.onnx_policy_base import ONNXPolicyBase
from xmigcs.policy.supermimic.reference_motion import ReferenceMotion, _rotation_6d, _yaw_matrix


class FSMStateSUPERMIMIC(ONNXPolicyBase):
    """Run a 50 Hz reference clip with the policy's 10-frame observation history."""

    def __init__(self, robot_data: RobotData, config_path: str | None = None,
                 model_path: str | None = None, motion_path: str | None = None):
        super().__init__(robot_data)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = config_path or os.path.join(current_dir, "config", "supermimic.yaml")
        cfg = self._load_config(config_path)
        model_path = model_path or cfg["model_path"]
        motion_path = motion_path or cfg["motion_path"]
        if not os.path.isabs(model_path):
            model_path = os.path.join(current_dir, "model", model_path)
        if not os.path.isabs(motion_path):
            motion_path = os.path.join(current_dir, "motion", motion_path)

        self.model_path = model_path
        self._init_supermimic_session(model_path)
        self._configure_policy(cfg)
        self.motion_ = ReferenceMotion(motion_path, self.joint_seq, self.goal_frames_, self.goal_encoding_,
                                       clip_name=cfg.get("motion_clip"))
        self._init_buffers()
        self.future_goal_ = np.zeros(self.goal_frames_ * 73, dtype=np.float32)
        self._reference_alignment = np.eye(3, dtype=np.float32)
        self._sensor_in_pelvis = self._load_sensor_rotation(cfg["control"].get("imu_frame", "pelvis"))
        self._warm_start_steps = max(0, round(float(cfg["control"].get("warm_start_time", 0.0)) / self.dt_))
        self._warm_start_pose = np.zeros(self.action_num_, dtype=np.float32)
        self._warm_start_count = 0
        self._start_frame = int(cfg["control"].get("start_frame", 0))
        if not 0 <= self._start_frame < self.motion_.length - 1:
            raise ValueError("start_frame must precede the final reference frame")
        self._align_heading = bool(cfg["control"].get("align_reference_heading", True))
        self._frame = self._start_frame
        self._finished = False
        self._policy_ticks_remaining = 0
        self._last_joint_target = self.joint_pos_array_seq.copy()

    def _configure_policy(self, cfg: dict) -> None:
        """Select the parameter and feature contract from the ONNX interface."""
        self.action_gain = np.ones(29, dtype=np.float32)
        self.reference_mask = np.zeros(29, dtype=np.float32)
        spec = self._configure_actor_policy() if self._actor_export else self._configure_embedded_policy()
        arrays = (self.joint_pos_array_seq, self.action_scale,
                  self.stiffness_array_seq, self.damping_array_seq,
                  self.action_gain, self.reference_mask)
        if (self.action_num_ != 29 or len(self.joint_seq) != 29 or set(self.joint_seq) != set(MJ_JOINTS_29)
                or any(a.shape != (29,) or not np.isfinite(a).all() for a in arrays)
                or np.any(self.action_scale <= 0) or np.any(self.stiffness_array_seq < 0)
                or np.any(self.damping_array_seq < 0)):
            raise ValueError("SuperMimic model, gains, and EVT2 joint names do not match")
        if self.motor_num_ != len(MJ_JOINTS_31) or self.obs_size_ != 99 or self.num_hist_ < 1:
            raise ValueError("Expected 31 motors and 99 features per observation")
        if not np.isfinite(self.control_dt_) or not np.isclose(self.control_dt_, 0.02, rtol=1e-6, atol=1e-9):
            raise ValueError("SuperMimic policy period must be 0.02 s for the 50 Hz reference motion")
        if (not isinstance(self.decimation_, (int, np.integer))
                or isinstance(self.decimation_, bool) or self.decimation_ < 1
                or not np.isfinite(self.dt_) or self.dt_ <= 0
                or not np.isclose(self.dt_ * self.decimation_, self.control_dt_, rtol=1e-6, atol=1e-9)):
            raise ValueError(
                "SuperMimic requires positive dt and integer decimation >= 1 with "
                f"dt * decimation = 0.02 s (got dt={self.dt_}, decimation={self.decimation_})"
            )
        expected_encoding = "current_ref" if self._actor_export else "reference_relative"
        configured_encoding = cfg.get("goal_encoding", "auto")
        if configured_encoding not in ("auto", expected_encoding):
            raise ValueError(f"This model requires {expected_encoding} goals")
        self.goal_encoding_ = expected_encoding
        if any(joint.get("name") in self.joint_seq for joint in cfg.get("fixed_joints", [])):
            raise ValueError("fixed_joints cannot override a joint controlled by the ONNX policy")
        self._build_joint_mapping(MJ_JOINTS_31, cfg.get("fixed_joints", []))
        source = "actor JSON" if self._actor_export else "ONNX"
        print(f"[FSMStateSUPERMIMIC] Parameters from {source}: iteration={spec.get('iteration', 'unknown')}, "
              f"action_scale={np.unique(self.action_scale).tolist()}, "
              f"policy_period={self.control_dt_:g}s, history={self.num_hist_}")

    def _configure_embedded_policy(self) -> dict:
        """Keep the original embedded SuperMimic interface authoritative."""
        raw = self.ort_session_.get_modelmeta().custom_metadata_map.get("supermimic_interface")
        try:
            spec = json.loads(raw) if raw else None
            if not isinstance(spec, dict) or spec.get("robot") != "tiangong":
                raise ValueError("missing Tiangong supermimic_interface")
            self.joint_seq = list(spec["joint_names"])
            self.joint_pos_array_seq = np.asarray(spec["default_joint_positions"], dtype=np.float32)
            self.action_scale = np.asarray(spec["action_scales"], dtype=np.float32)
            rows = spec["robot_config"]["joints"]
            if len(rows) != 29 or len({joint["name"] for joint in rows}) != 29:
                raise ValueError("robot_config must contain 29 distinct joints")
            joints = {joint["name"]: joint for joint in rows}
            if set(joints) != set(self.joint_seq):
                raise ValueError("robot_config joints differ from policy joint_names")
            self.stiffness_array_seq = np.asarray([joints[n]["stiffness"] for n in self.joint_seq], dtype=np.float32)
            self.damping_array_seq = np.asarray([joints[n]["damping"] for n in self.joint_seq], dtype=np.float32)
            self.control_dt_ = float(spec["control_dt_s"])
            snapshot_q = np.asarray([joints[n]["default_pos"] for n in self.joint_seq], dtype=np.float32)
            snapshot_scale = np.asarray([joints[n]["action_scale"] for n in self.joint_seq], dtype=np.float32)
            if (self.joint_pos_array_seq.shape != (29,) or self.action_scale.shape != (29,)
                    or not np.allclose(self.joint_pos_array_seq, snapshot_q, rtol=1e-6, atol=1e-7)
                    or not np.allclose(self.action_scale, snapshot_scale, rtol=1e-6, atol=1e-7)):
                raise ValueError("ONNX action/default-position fields disagree with robot_config")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"Invalid or incomplete embedded SuperMimic parameters in {self.model_path}: {error}. "
                "Re-export with tools/export_onnx.py to embed robot_config; YAML and JSON sidecars are not used."
            ) from error
        return spec

    def _configure_actor_policy(self) -> dict:
        """Load the format-2 EVT actor sidecar, including residual action semantics."""
        metadata_path = Path(self.model_path).with_suffix(".json")
        try:
            with metadata_path.open() as stream:
                spec = json.load(stream)
            task, control = spec["task_config"], spec["control"]
            obs, goal = spec["inputs"]["obs_history"], spec["inputs"]["goal"]
            obs_features = ["joint_pos_minus_default:29", "joint_vel:29", "root_gyro_local:3",
                            "projected_gravity_local:3", "previous_policy_action:29",
                            "reference_root_orientation_relative_to_robot_6d:6"]
            goal_features = ["root_position:3", "root_rotation_6d:6", "joint_pos:29", "joint_vel:29",
                             "root_linear_velocity_local:3", "root_angular_velocity_local:3"]
            if (spec["format_version"] != 2 or spec["model_file"] != Path(self.model_path).name
                    or spec["robot"] != "evt" or task["actor_ref_orientation"] is not True
                    or task["goal_pos"] != "current_ref" or task["goal_rot"] != "current_ref"
                    or task["heading"] != "xaxis" or task["rot_repr"] != "6d"
                    or task["action"] != "res" or task["res_legs"] != "abs"
                    or spec["history_frames"] != self.num_hist_
                    or spec["future_frame_offsets"] != list(range(1, self.goal_frames_ + 1))
                    or obs["shape"][1:] != [self.num_hist_ * 99]
                    or goal["shape"][1:] != [self.goal_frames_ * 73]
                    or obs["frame_order"] != "oldest to newest" or obs["per_frame"] != obs_features
                    or goal["per_frame"] != goal_features
                    or obs["reference_orientation"]["rotation_6d_order"]
                    != ["R00", "R10", "R20", "R01", "R11", "R21"]
                    or not np.isclose(float(spec["control_frequency_hz"]), 50.)):
                raise ValueError("unsupported or mismatched actor interface")
            self.control_dt_ = float(task["dt"])
            self.joint_seq = list(control["joint_names"])
            rows = control["actuators"]
            joints = {joint["joint"]: joint for joint in rows}
            if len(rows) != 29 or len(joints) != 29 or set(joints) != set(self.joint_seq):
                raise ValueError("actor actuators must match all 29 policy joints")
            self.joint_pos_array_seq = np.asarray(control["default_joint_positions_rad"], dtype=np.float32)
            self.action_scale = np.asarray(control["action_scale_rad"], dtype=np.float32)
            self.stiffness_array_seq = np.asarray([joints[n]["stiffness"] for n in self.joint_seq], dtype=np.float32)
            self.damping_array_seq = np.asarray([joints[n]["damping"] for n in self.joint_seq], dtype=np.float32)
            self.action_gain = np.asarray(control["policy_action_gain"], dtype=np.float32)
            self.reference_mask = np.asarray(control["reference_mask"], dtype=np.float32)
            legs = np.array([any(group in n for group in ("hip", "knee", "ankle")) for n in self.joint_seq])
            expected_gain = np.where(legs, 1., float(task["res_scale"]))
            if (not np.array_equal(control["leg_mask"], legs)
                    or not np.array_equal(self.reference_mask, (~legs).astype(np.float32))
                    or self.action_gain.shape != (29,)
                    or not np.allclose(self.action_gain, expected_gain)):
                raise ValueError("actor residual masks/gains disagree with task_config")
        except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid SuperMimic actor metadata in {metadata_path}: {error}") from error
        return spec

    def _init_supermimic_session(self, model_path: str) -> None:
        """Validate and warm up both ONNX inputs; the shared base warms up one."""
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        options.enable_mem_reuse = True
        self.ort_session_ = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        inputs = {value.name: value for value in self.ort_session_.get_inputs()}
        self._obs_input_name = "observation_history" if "observation_history" in inputs else "obs_history"
        self._actor_export = set(inputs) == {"obs_history", "goal"}
        self._goal_input_name = "goal" if self._actor_export else "future_goal"
        if set(inputs) != {self._obs_input_name, self._goal_input_name}:
            raise ValueError(f"Unexpected SuperMimic ONNX inputs: {list(inputs)}")
        obs_shape, goal_shape = inputs[self._obs_input_name].shape, inputs[self._goal_input_name].shape
        if self._actor_export:
            valid_obs = (len(obs_shape) == 2 and isinstance(obs_shape[1], int)
                         and obs_shape[1] >= 99 and obs_shape[1] % 99 == 0)
        else:
            valid_obs = (len(obs_shape) == 3 and isinstance(obs_shape[1], int)
                         and obs_shape[1] >= 1 and obs_shape[2] == 99)
        if (not valid_obs or len(goal_shape) != 2 or not isinstance(goal_shape[1], int)
                or goal_shape[1] < 73 or goal_shape[1] % 73
                or any(value.type != "tensor(float)" for value in inputs.values())
                or any(isinstance(shape[0], int) and shape[0] != 1 for shape in (obs_shape, goal_shape))):
            raise ValueError(f"Unexpected SuperMimic input shapes/dtypes: {obs_shape}, {goal_shape}")
        self.goal_frames_ = goal_shape[1] // 73
        outputs = self.ort_session_.get_outputs()
        if (len(outputs) != 1 or outputs[0].name != "action" or len(outputs[0].shape) != 2
                or outputs[0].shape[1] != 29 or outputs[0].type != "tensor(float)"):
            raise ValueError("Unexpected SuperMimic ONNX action output")
        self.num_hist_ = obs_shape[1] // 99 if self._actor_export else obs_shape[1]
        self.obs_size_, self.action_num_ = 99, outputs[0].shape[1]
        self._obs_input_shape = (1, self.num_hist_ * 99) if self._actor_export else (1, self.num_hist_, 99)
        self.ort_session_.run(["action"], {
            self._obs_input_name: np.zeros(self._obs_input_shape, dtype=np.float32),
            self._goal_input_name: np.zeros((1, self.goal_frames_ * 73), dtype=np.float32),
        })
        print(f"[FSMStateSUPERMIMIC] ONNX model loaded: {model_path}")

    @staticmethod
    def _load_sensor_rotation(frame: str) -> np.ndarray:
        if frame == "pelvis":
            return np.eye(3, dtype=np.float32)
        if frame != "imu_waist_link":
            raise ValueError(f"Unsupported IMU frame: {frame}")
        urdf_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "evt2_29.urdf")
        joint = next(j for j in ET.parse(urdf_path).getroot().findall("joint")
                     if j.get("name") == "imu_waist_joint")
        rpy = [float(value) for value in joint.find("origin").get("rpy").split()]
        return Rotation.from_euler("xyz", rpy).as_matrix().astype(np.float32)

    def _pelvis_rotation_and_gyro(self) -> tuple[np.ndarray, np.ndarray]:
        sensor_rotation = matrix_from_quat(self.robot_data_.get_robot_quat()).astype(np.float32)
        pelvis_rotation = sensor_rotation @ self._sensor_in_pelvis.T
        pelvis_gyro = self._sensor_in_pelvis @ self.robot_data_.get_angular_velocity()
        return pelvis_rotation, pelvis_gyro.astype(np.float32)

    def _align_reference_heading(self) -> None:
        pelvis_rotation, _ = self._pelvis_rotation_and_gyro()
        self._reference_alignment = (_yaw_matrix(pelvis_rotation)
                                     @ _yaw_matrix(self.motion_.root_R[self._start_frame]).T
                                     if self._align_heading else np.eye(3, dtype=np.float32))

    def on_enter(self) -> None:
        super().on_enter()
        self._frame = self._start_frame
        self._finished = False
        self._policy_ticks_remaining = 0
        self._reference_alignment = np.eye(3, dtype=np.float32)
        self._warm_start_count = 0
        self._warm_start_pose = self.robot_data_.get_joint_pos()[self.lab2mj].copy()
        if not np.isfinite(self._warm_start_pose).all():
            raise ValueError("Nonfinite joint state on SuperMimic entry")
        self._last_joint_target = self._warm_start_pose.copy()
        if self._warm_start_steps == 0:
            self._align_reference_heading()

    def run(self, flag: ControlFlag) -> None:
        if self._finished:
            self._write_joint_targets(self._last_joint_target)
            flag.fsm_state_command = "gotoSTOP"
        elif self._warm_start_count < self._warm_start_steps:
            self._warm_start_count += 1
            blend = self._warm_start_count / self._warm_start_steps
            target = (1.0 - blend) * self._warm_start_pose + blend * self.motion_.q[self._start_frame]
            self._write_joint_targets(target)
            if self._warm_start_count == self._warm_start_steps:
                self._align_reference_heading()
        elif self._policy_ticks_remaining:
            self._policy_ticks_remaining -= 1
            self._write_joint_targets(self._last_joint_target)
        else:
            self.compute_observation(flag)
            self.compute_actions()
            self._apply_action_to_robot()
            self._policy_ticks_remaining = self.decimation_ - 1
            self._frame += 1
            if self._frame >= self.motion_.length - 1:
                self._finished = True
                flag.fsm_state_command = "gotoSTOP"
        self._apply_pd_gains()
        self._apply_fixed_joints()

    def compute_observation(self, flag: ControlFlag) -> None:
        pelvis_rotation, gyro = self._pelvis_rotation_and_gyro()
        q = self.robot_data_.get_joint_pos()[self.lab2mj]
        qd = self.robot_data_.get_joint_vel()[self.lab2mj]
        gravity = -pelvis_rotation[2, :]
        reference_rotation = self._reference_alignment @ self.motion_.root_R[self._frame]
        relative_rotation = pelvis_rotation.T @ reference_rotation
        proprio = np.concatenate((q - self.default_angles_lab, qd, gyro, gravity,
                                  self.last_actions_, _rotation_6d(relative_rotation,
                                                                column_major=self._actor_export))).astype(np.float32)
        if not np.isfinite(proprio).all():
            raise ValueError("Nonfinite SuperMimic observation")
        self._update_proprio_history(proprio)
        self.future_goal_ = self.motion_.goal(self._frame)

    def compute_actions(self) -> None:
        action, = self.ort_session_.run(["action"], {
            self._obs_input_name: self.observations_.reshape(self._obs_input_shape),
            self._goal_input_name: self.future_goal_.reshape(1, -1),
        })
        action = action[0]
        if action.shape != (self.action_num_,) or not np.isfinite(action).all():
            raise ValueError("Invalid SuperMimic action")
        self.actions_[:] = action

    def _write_joint_targets(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError("Invalid SuperMimic joint target")
        self._last_joint_target = target.copy()
        base = self.robot_data_.q_d_.shape[0] - self.motor_num_
        indices = base + self.lab2mj
        self.robot_data_.q_d_[indices] = target
        self.robot_data_.q_dot_d_[indices] = 0.0
        self.robot_data_.tau_d_[indices] = 0.0

    def _apply_action_to_robot(self) -> None:
        target = self.default_angles_lab + self.action_scale_lab * self.action_gain * self.actions_
        if self._actor_export:
            next_q = self.motion_.q[min(self._frame + 1, self.motion_.length - 1)]
            target += self.reference_mask * (next_q - self.default_angles_lab)
        self._write_joint_targets(target)
        self.last_actions_[:] = self.actions_

    def _apply_pd_gains(self) -> None:
        self.robot_data_.joint_kp_p_[:len(self.joint_xml)] = self.stiffness_array
        self.robot_data_.joint_kd_p_[:len(self.joint_xml)] = self.damping_array
