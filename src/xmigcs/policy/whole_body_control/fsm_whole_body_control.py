"""U state: SuperMimic inference with NPZ or delayed, progressively filled GMR goals."""
import json
import os
import time
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import onnxruntime
import yaml
from scipy.spatial.transform import Rotation

from xmigcs.FSM.fsm_base import FSMState
from xmigcs.common.joint_mapping import MJ_JOINTS_29
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData
from xmigcs.utils.logging_utils import get_logger
from .reference_window import (NPZReference, DelayedReferenceWindow, encode_goal,
                               heading, joint_permutation, reference_frame, rotation)

_logger = get_logger(__name__)


def _build_onnx_session(onnx_path: str):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return onnxruntime.InferenceSession(
        onnx_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


DEFAULT_BODY_NAMES_24 = [
    "pelvis",
    "hip_pitch_l_link",
    "hip_roll_l_link",
    "hip_yaw_l_link",
    "knee_pitch_l_link",
    "ankle_pitch_l_link",
    "ankle_roll_l_link",
    "hip_pitch_r_link",
    "hip_roll_r_link",
    "hip_yaw_r_link",
    "knee_pitch_r_link",
    "ankle_pitch_r_link",
    "ankle_roll_r_link",
    "waist_yaw_link",
    "waist_roll_link",
    "waist_pitch_link",
    "shoulder_pitch_l_link",
    "shoulder_roll_l_link",
    "shoulder_yaw_l_link",
    "elbow_pitch_l_link",
    "shoulder_pitch_r_link",
    "shoulder_roll_r_link",
    "shoulder_yaw_r_link",
    "elbow_pitch_r_link",
]


class FSMStateWHOLE_BODY_CONTROL(FSMState):
    def __init__(self, robot_data: RobotData, config_path: Optional[str] = None):
        super().__init__(robot_data)
        current_dir = getattr(self, "_policy_dir", os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.abspath(config_path or os.path.join(current_dir, "config", "whole_body_control.yaml"))
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
        self.physical_dt = float(config["physical_dt"])
        self.decimation_ = config["decimation"]
        self.inference_period_s = self.physical_dt * self.decimation_
        if (not isinstance(self.decimation_, int) or isinstance(self.decimation_, bool) or self.decimation_ < 1
                or self.physical_dt <= 0 or not np.isclose(self.inference_period_s, .02, rtol=1e-6, atol=1e-9)):
            raise ValueError("WHOLE_BODY_CONTROL requires physical_dt * decimation = 0.02 s")
        self.inference_hz_nominal = 50.
        self.num_actions = self.motor_nums = 29
        if config.get("num_actions", 29) != 29 or config.get("motor_nums", 29) != 29:
            raise ValueError("WHOLE_BODY_CONTROL controls exactly 29 body joints")
        self.locked_joint_map = np.asarray(config.get("locked_joint_map", list(range(29))))
        if (self.locked_joint_map.shape != (29,) or not np.issubdtype(self.locked_joint_map.dtype, np.integer)
                or set(self.locked_joint_map) != set(range(29))):
            raise ValueError("locked_joint_map must be a permutation of 0..28")
        self.body_names = config.get("body_names", DEFAULT_BODY_NAMES_24)
        if config.get("anchor_body", "pelvis") != "pelvis":
            raise ValueError("SuperMimic requires the pelvis reference anchor")
        self.anchor_body_index = self.body_names.index("pelvis")
        self.ort_session = _build_onnx_session(self.onnx_path)
        self._configure_policy()
        self.num_obs = 99
        self.obs = np.zeros(99, dtype=np.float32)
        self.history = np.zeros((self.history_length, 99), dtype=np.float32)
        self.future_goal = np.zeros(self.goal_frames * 73, dtype=np.float32)
        self.action = np.zeros(29, dtype=np.float32)

        self._sensor_in_pelvis = np.eye(3, dtype=np.float32)
        imu_frame = config.get("imu_frame", "pelvis")
        if imu_frame == "imu_waist_link":
            urdf = os.path.join(current_dir, "..", "..", "config", "evt2_29.urdf")
            joint = next(j for j in ET.parse(urdf).getroot().findall("joint") if j.get("name") == "imu_waist_joint")
            self._sensor_in_pelvis = Rotation.from_euler("xyz", [float(v) for v in joint.find("origin").get("rpy").split()]).as_matrix().astype(np.float32)
        elif imu_frame != "pelvis":
            raise ValueError("imu_frame must be pelvis or imu_waist_link")

        self.tau_limit_enabled = bool(config.get("tau_limit_enabled", False))
        if self.tau_limit_enabled:
            # Existing YAML limits are in motor order; absent limits use the model.
            self.tau_limit = np.asarray(config.get("tau_limit", self.effort_limits_motor), dtype=np.float32)
            if self.tau_limit.shape != (29,) or not np.isfinite(self.tau_limit).all() or np.any(self.tau_limit <= 0):
                raise ValueError("tau_limit must contain 29 positive finite values in motor order")
            self._stiffness_lab = self.stiffness_array_seq
            self._damping_lab = self.damping_array_seq
            self._tau_limit_lab = self.tau_limit[self.locked_joint_map][self.mj2lab]
            self._qdot_d_lab_zeros = np.zeros(29, dtype=np.float32)

        self.warn_on_missing_ref = bool(config.get("warn_on_missing_ref", True))
        self.warn_throttle_sec = float(config.get("warn_throttle_sec", 2.))
        self.ref_source = str(config.get("ref_source", "gmr")).lower()
        if self.ref_source not in ("npz", "gmr"):
            raise ValueError("ref_source must be npz or gmr")
        self.motion_length = config.get("motion_length")
        if self.motion_length is not None and (not isinstance(self.motion_length, int) or self.motion_length < 1):
            raise ValueError("motion_length must be a positive integer or null")
        self.hold_final_reference = bool(config.get("hold_final_reference", False))
        self.motion = None
        if self.ref_source == "npz":
            motion_path = os.path.join(current_dir, "model", config["motion_file"])
            self.motion = NPZReference(motion_path, self.joint_seq, config.get("npz_joint_names"),
                                       int(config.get("npz_anchor_body_index", 0)), config.get("motion_clip"))
        self._gmr_perm = (joint_permutation(config.get("gmr_joint_names"), self.joint_seq)
                          if self.ref_source == "gmr" else None)
        self.gmr_window = DelayedReferenceWindow(self.goal_frames, self.inference_period_s,
                                                 float(config.get("gmr_gap_reset_s", .1)))
        self._reset_tracking()

    def _configure_policy(self):
        inputs = {i.name: i for i in self.ort_session.get_inputs()}
        self._obs_input_name = "observation_history" if "observation_history" in inputs else "obs_history"
        if set(inputs) != {self._obs_input_name, "future_goal"}:
            raise ValueError("Expected SuperMimic observation_history/obs_history + future_goal; old 770-d models are incompatible")
        hs, gs = inputs[self._obs_input_name].shape, inputs["future_goal"].shape
        if (len(hs) != 3 or hs[-1] != 99 or not isinstance(hs[1], int) or hs[1] < 1
                or len(gs) != 2 or not isinstance(gs[1], int) or gs[1] < 73 or gs[1] % 73
                or any(i.type != "tensor(float)" for i in inputs.values())
                or any(isinstance(s[0], int) and s[0] != 1 for s in (hs, gs))):
            raise ValueError("Expected float32 [1,H,99] and [1,73*K] inputs")
        outputs = self.ort_session.get_outputs()
        if (len(outputs) != 1 or outputs[0].name != "action" or len(outputs[0].shape) != 2
                or outputs[0].shape[1] != 29 or outputs[0].type != "tensor(float)"):
            raise ValueError("Expected float32 action [1,29] output")
        self.history_length, self.goal_frames = hs[1], gs[1] // 73
        raw = self.ort_session.get_modelmeta().custom_metadata_map.get("supermimic_interface")
        if not raw:
            raise ValueError("Re-export ONNX with supermimic_interface and robot_config metadata")
        meta = json.loads(raw)
        if meta.get("robot") != "tiangong" or meta.get("self_collisions", False):
            raise ValueError("Expected Tiangong SuperMimic metadata with self collisions disabled")
        if not np.isclose(float(meta["control_dt_s"]), .02, rtol=1e-6, atol=1e-9):
            raise ValueError("Model control_dt_s must be 0.02 s")
        self.joint_seq = list(meta["joint_names"])
        self.joint_xml = list(MJ_JOINTS_29)
        self.mj2lab = joint_permutation(self.joint_xml, self.joint_seq)
        rows = meta["robot_config"]["joints"]
        joints = {j["name"]: j for j in rows}
        if len(rows) != 29 or set(joints) != set(self.joint_seq):
            raise ValueError("ONNX robot_config must contain exactly the 29 policy joints")
        self.default_angles_lab = np.asarray(meta["default_joint_positions"], dtype=np.float32)
        self.action_scale_lab = np.asarray(meta["action_scales"], dtype=np.float32)
        self.stiffness_array_seq = np.asarray([joints[n]["stiffness"] for n in self.joint_seq], dtype=np.float32)
        self.damping_array_seq = np.asarray([joints[n]["damping"] for n in self.joint_seq], dtype=np.float32)
        effort = np.asarray([joints[n]["effort_limit"] for n in self.joint_seq], dtype=np.float32)
        arrays = (self.default_angles_lab, self.action_scale_lab, self.stiffness_array_seq, self.damping_array_seq, effort)
        if (any(a.shape != (29,) or not np.isfinite(a).all() for a in arrays)
                or np.any(self.action_scale_lab <= 0) or np.any(effort <= 0)
                or np.any(self.stiffness_array_seq < 0) or np.any(self.damping_array_seq < 0)):
            raise ValueError("Invalid ONNX joint parameters")
        for actual, key in ((self.default_angles_lab, "default_pos"), (self.action_scale_lab, "action_scale")):
            if not np.allclose(actual, [joints[n][key] for n in self.joint_seq], rtol=1e-6, atol=1e-7):
                raise ValueError(f"ONNX {key} disagrees with robot_config")
        self._motor_indices = self.locked_joint_map[self.mj2lab]
        def motor_order(values):
            result = np.empty(29, dtype=np.float32)
            result[self._motor_indices] = values
            return result
        self.joint_pos_array = motor_order(self.default_angles_lab)
        self.stiffness_array = motor_order(self.stiffness_array_seq)
        self.damping_array = motor_order(self.damping_array_seq)
        self.effort_limits_motor = motor_order(effort)

    def _reset_tracking(self):
        self.counter_step = 0
        self._npz_frame = 0
        self._alignment_ticks = 0
        self._enter_robot_time = float(self.robot_data_.time_now_)
        self._last_inference_tick = -1
        self._history_initialized = False
        self._last_missing_warn_t = 0.
        self._ref_ever_received = False
        self.init_to_world = np.eye(3, dtype=np.float32)
        self.history.fill(0)
        self.obs.fill(0)
        self.future_goal.fill(0)
        self.action.fill(0)
        self.gmr_window.clear()
        self.current_reference = None

    def on_enter(self):
        self._reset_tracking()
        if self.ref_source == "gmr":
            self._collect_gmr_reference()
        _logger.info("[WHOLE_BODY_CONTROL] %s, ref=%s, history=%d, future=%d, policy=50 Hz, GMR delay up to %.3f s",
                     os.path.basename(self.onnx_path), self.ref_source, self.history_length,
                     self.goal_frames, self.goal_frames * self.inference_period_s)

    def on_exit(self):
        self._reset_tracking()

    def check_transition(self, *args, **kwargs):
        return True

    def run(self, flag: ControlFlag):
        # Observe interpolated references on every outer tick, infer only at 50 Hz.
        if self.ref_source == "gmr":
            self._collect_gmr_reference()
        elapsed = max(0., float(self.robot_data_.time_now_) - self._enter_robot_time)
        tick = int(np.floor((elapsed + 1e-9) / self.inference_period_s))
        if tick > self._last_inference_tick:
            self._last_inference_tick = tick
            self.inner_run()
        self.set_kp_kd()

    def set_kp_kd(self):
        self.robot_data_.joint_kp_p_[:29] = self.stiffness_array
        self.robot_data_.joint_kd_p_[:29] = self.damping_array

    def _collect_gmr_reference(self):
        rd = self.robot_data_
        stamp = getattr(rd, "ref_sample_time", None)
        if stamp is None or (self.gmr_window.last_time is not None and stamp <= self.gmr_window.last_time + 1e-9):
            return False
        if any(getattr(rd, k, None) is None for k in ("ref_joint_pos", "ref_joint_vel", "ref_body_pos_w",
                "ref_body_quat_w", "ref_body_lin_vel_w", "ref_body_ang_vel_w")):
            return False
        def anchor(value, width):
            return np.asarray(value).reshape(-1, width)[self.anchor_body_index]
        frame = reference_frame(anchor(rd.ref_body_pos_w, 3), anchor(rd.ref_body_quat_w, 4),
                                anchor(rd.ref_body_lin_vel_w, 3), anchor(rd.ref_body_ang_vel_w, 3),
                                np.asarray(rd.ref_joint_pos).reshape(29)[self._gmr_perm],
                                np.asarray(rd.ref_joint_vel).reshape(29)[self._gmr_perm])
        fresh = self.gmr_window.push(stamp, frame)
        self._ref_ever_received |= fresh
        return fresh

    def _reference_window(self):
        if self.ref_source == "npz":
            end = self.motion_length if self.hold_final_reference else None
            return self.motion.window(self._npz_frame, self.goal_frames, end)
        self._collect_gmr_reference()
        return self.gmr_window.window()

    def _maybe_warn_missing(self):
        now = time.monotonic()
        received = getattr(self.robot_data_, "ref_received_time", None)
        stalled = received is None or now - received > self.gmr_window.gap_reset_s
        if self.warn_on_missing_ref and stalled and now - self._last_missing_warn_t >= self.warn_throttle_sec:
            self._last_missing_warn_t = now
            _logger.warning("[WHOLE_BODY_CONTROL] %s", "GMR stalled; tracking frozen reference window"
                            if self._ref_ever_received else "Waiting for first timestamped GMR reference; target unchanged")

    def inner_run(self):
        window = self._reference_window()
        if self.ref_source == "gmr":
            self._maybe_warn_missing()
        if window is None:
            return
        current, future, recorded = window
        self.current_reference = current
        sensor_rot = rotation(self.robot_data_.get_robot_quat())
        robot_rot = sensor_rot @ self._sensor_in_pelvis.T
        if self._alignment_ticks < 2:
            self.init_to_world = heading(robot_rot) @ heading(current['rot']).T
            self._alignment_ticks += 1
        else:
            q = self.robot_data_.get_joint_pos()[self._motor_indices]
            qd = self.robot_data_.get_joint_vel()[self._motor_indices]
            gyro = self._sensor_in_pelvis @ self.robot_data_.get_angular_velocity()
            relative = robot_rot.T @ self.init_to_world @ current['rot']
            self.obs = np.concatenate((q-self.default_angles_lab, qd, gyro, robot_rot.T @ [0, 0, -1],
                                       self.action, relative[:, :2].reshape(6))).astype(np.float32)
            if not np.isfinite(self.obs).all():
                raise ValueError("Non-finite WHOLE_BODY_CONTROL observation")
            if not self._history_initialized:
                self.history[:] = self.obs
                self._history_initialized = True
            else:
                self.history[:-1] = self.history[1:]
                self.history[-1] = self.obs
            # Goals are invariant to the common yaw/translation alignment; use raw reference coordinates.
            self.future_goal = encode_goal(current, future, recorded)
            action = np.asarray(self.ort_session.run(["action"], {
                self._obs_input_name: self.history[None], "future_goal": self.future_goal[None],
            })[0], dtype=np.float32)
            if action.shape != (1, 29) or not np.isfinite(action).all():
                raise ValueError("Policy must produce finite action [1,29]")
            self.action = action[0].copy()
            target = self.default_angles_lab + self.action_scale_lab * self.action
            if self.tau_limit_enabled:
                target = self._apply_tau_limit_lab(target)
            self.robot_data_.q_d_[6 + self._motor_indices] = target
            self.robot_data_.q_dot_d_[6 + self._motor_indices] = 0.
            self.robot_data_.tau_d_[6 + self._motor_indices] = 0.
        self.counter_step += 1
        if self.ref_source == "npz":
            # Preserve the original NPZ startup: alignment consumes its first two frames.
            self._npz_frame += 1

    # ---------- helpers borrowed from lingdao ----------
    def _get_current_joint_pos_lab(self) -> np.ndarray:
        try:
            current_q = self.robot_data_.get_joint_pos()
            current_q = current_q[self.locked_joint_map]
            return current_q[self.mj2lab].astype(np.float32)
        except Exception as exc:
            _logger.warning("[FSMStateWHOLE_BODY_CONTROL] read current joint pose failed: %s", exc)
            return np.array(self.default_angles_lab, copy=True)

    def _get_current_joint_vel_lab(self) -> np.ndarray:
        try:
            current_qdot = self.robot_data_.get_joint_vel()
            current_qdot = current_qdot[self.locked_joint_map]
            return current_qdot[self.mj2lab].astype(np.float32)
        except Exception as exc:
            _logger.warning("[FSMStateWHOLE_BODY_CONTROL] read current joint vel failed: %s", exc)
            return np.zeros(self.num_actions, dtype=np.float32)

    def _apply_tau_limit_lab(self, q_d_lab: np.ndarray) -> np.ndarray:
        """Clip implied PD torque to tau_limit by back-solving q_d (lab order)."""
        q_d_lab = np.asarray(q_d_lab, dtype=np.float32).reshape(-1)
        q_lab = self._get_current_joint_pos_lab()
        qdot_lab = self._get_current_joint_vel_lab()
        tau = (
            self._stiffness_lab * (q_d_lab - q_lab)
            + self._damping_lab * (self._qdot_d_lab_zeros - qdot_lab)
        )
        over = np.abs(tau) > self._tau_limit_lab
        if not np.any(over):
            return q_d_lab
        tau_c = np.clip(tau, -self._tau_limit_lab, self._tau_limit_lab)
        safe_kp = np.where(self._stiffness_lab == 0.0, 1.0, self._stiffness_lab)
        q_d_clipped = q_d_lab.copy()
        q_d_clipped[over] = (
            q_lab[over]
            + (
                tau_c[over]
                - self._damping_lab[over]
                * (self._qdot_d_lab_zeros[over] - qdot_lab[over])
            )
            / safe_kp[over]
        )
        return q_d_clipped
