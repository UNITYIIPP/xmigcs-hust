"""WHOLE_BODY_CONTROL FSM state.

Same ONNX / observation / output layout as ``policy/lingdao``; the only
behavioural difference is that the reference trajectory comes from the
real-time GMR teleoperation stream (``robot_data_.ref_*``, populated by
``RobotInterfaceImpl.get_realtime_gmr_data``) instead of a static NPZ file.

The 770-dim observation layout is preserved 1:1:
``[cmd58*5 | ori6*5 | ang_vel3*5 | jp29*5 | jv29*5 | act29*5]``.
"""
import os
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import onnxruntime
import yaml

from xmigcs.FSM.fsm_base import FSMState
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData
from xmigcs.utils.logging_utils import get_logger


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
        self.counter_step = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        if config_path is None:
            config_path = os.path.join(current_dir, "config", "whole_body_control.yaml")
        self.config_path = os.path.abspath(config_path)
        with open(self.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.onnx_path = config["onnx_path"]
        self.onnx_path = os.path.join(current_dir, "model", self.onnx_path)
        self.physical_dt = config["physical_dt"]
        self.decimation_ = config["decimation"]
        self.inference_period_s = float(self.decimation_) * float(self.physical_dt)
        self.inference_hz_nominal = (
            1.0 / self.inference_period_s if self.inference_period_s > 0 else 0.0
        )
        self._enter_robot_time = 0.0
        self._last_inference_tick = -1
        self.num_actions = config["num_actions"]
        self.motor_nums = config["motor_nums"]
        self.body_names = config.get("body_names", DEFAULT_BODY_NAMES_24)
        self.locked_joint_map = config["locked_joint_map"]
        self.anchor_body_name = config.get("anchor_body", "pelvis")
        if self.anchor_body_name not in self.body_names:
            raise ValueError(f"Anchor body {self.anchor_body_name} missing from body list.")
        self.anchor_body_index = self.body_names.index(self.anchor_body_name)
        self.num_bodies = len(self.body_names)

        # Per-tick observation buffers; sizes locked to the lingdao layout.
        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)

        # ONNX session + metadata (sourced from the ONNX file; matches lingdao protocol).
        self.ort_session = _build_onnx_session(self.onnx_path)
        model_meta = self.ort_session.get_modelmeta()
        custom_map = getattr(model_meta, "custom_metadata_map", {})
        metadata_props = [SimpleNamespace(key=k, value=v) for k, v in custom_map.items()]
        self.input_name = [inpt.name for inpt in self.ort_session.get_inputs()]
        self.num_obs = config.get("num_obs")
        self.obs = np.zeros(self.num_obs, dtype=np.float32)

        # 770 = 5 (history) × per-term widths.
        _H = 5
        self._buf_cmd = np.zeros((_H, 58), dtype=np.float32)
        self._buf_ori = np.zeros((_H, 6), dtype=np.float32)
        self._buf_ang_vel = np.zeros((_H, 3), dtype=np.float32)
        self._buf_joint_pos = np.zeros((_H, self.num_actions), dtype=np.float32)
        self._buf_joint_vel = np.zeros((_H, self.num_actions), dtype=np.float32)
        self._buf_actions = np.zeros((_H, self.num_actions), dtype=np.float32)
        self._history_initialized = False

        # Reference frame placeholders, filled from robot_data_.ref_* each tick.
        self.ref_joint_pos = np.zeros((1, self.num_actions), dtype=np.float32)
        self.ref_joint_vel = np.zeros((1, self.num_actions), dtype=np.float32)
        self.ref_anchor_pos_w = np.zeros(3, dtype=np.float32)
        self.ref_anchor_quat_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.joint_seq = None
        self.joint_pos_array_seq = None
        self.action_scale = None
        self.stiffness_array_seq = None
        self.damping_array_seq = None
        for prop in metadata_props:
            if prop.key == "joint_names":
                self.joint_seq = prop.value.split(",")
            elif prop.key == "default_joint_pos":
                self.joint_pos_array_seq = np.array([float(x) for x in prop.value.split(",")])
            elif prop.key == "joint_stiffness":
                self.stiffness_array_seq = np.array([float(x) for x in prop.value.split(",")])
            elif prop.key == "joint_damping":
                self.damping_array_seq = np.array([float(x) for x in prop.value.split(",")])
            elif prop.key == "action_scale":
                self.action_scale = np.array([float(x) for x in prop.value.split(",")])

        if self.joint_seq is None:
            raise RuntimeError(
                "ONNX metadata missing 'joint_names'; cannot build mj2lab map."
            )

        self.joint_xml = [
            "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
            "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
            "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
            "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
            "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
            "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
        ]
        self.mj2lab = np.array([self.joint_xml.index(j) for j in self.joint_seq])
        # mj-order copies of stiffness/damping/default that the runtime needs.
        self.joint_pos_array = np.array(
            [self.joint_pos_array_seq[self.joint_seq.index(j)] for j in self.joint_xml]
        )
        self.stiffness_array = np.array(
            [self.stiffness_array_seq[self.joint_seq.index(j)] for j in self.joint_xml]
        )
        self.damping_array = np.array(
            [self.damping_array_seq[self.joint_seq.index(j)] for j in self.joint_xml]
        )
        self.default_angles_lab = self.joint_pos_array_seq
        self.action_scale_lab = self.action_scale

        # Optional PD torque cap; identical semantics to lingdao.
        self.tau_limit_enabled = bool(config.get("tau_limit_enabled", False))
        if self.tau_limit_enabled:
            tau_limit_cfg = config.get("tau_limit")
            if tau_limit_cfg is None:
                raise ValueError("tau_limit_enabled is true but tau_limit is missing in config")
            self.tau_limit = np.array(tau_limit_cfg, dtype=np.float32)
            if self.tau_limit.shape[0] != self.num_actions:
                raise ValueError(
                    f"tau_limit length {self.tau_limit.shape[0]} != num_actions {self.num_actions}"
                )
            self._stiffness_lab = self.stiffness_array[self.mj2lab].astype(np.float32)
            self._damping_lab = self.damping_array[self.mj2lab].astype(np.float32)
            self._tau_limit_lab = self.tau_limit[self.mj2lab].astype(np.float32)
            self._qdot_d_lab_zeros = np.zeros(self.num_actions, dtype=np.float32)

        # Warning throttling for missing GMR frames.
        self.warn_on_missing_ref = bool(config.get("warn_on_missing_ref", True))
        self.warn_throttle_sec = float(config.get("warn_throttle_sec", 2.0))
        self._last_missing_warn_t = 0.0
        self._ref_ever_received = False
        self.init_to_world = np.eye(3, dtype=np.float32)

        # ---- Reference-source dispatch ("gmr" stream vs. "npz" replay) ----
        self.ref_source = str(config.get("ref_source", "gmr")).lower()
        if self.ref_source not in ("gmr", "npz"):
            raise ValueError(
                f"ref_source must be 'gmr' or 'npz', got {self.ref_source!r}"
            )
        self._npz_joint_pos = None
        self._npz_joint_vel = None
        self._npz_anchor_pos_w = None
        self._npz_anchor_quat_w = None
        self._npz_num_frames = 0
        self.motion_length = config.get("motion_length")
        self.hold_final_reference = bool(config.get("hold_final_reference", False))
        if self.ref_source == "npz":
            motion_file = config.get("motion_file")
            if not motion_file:
                raise ValueError(
                    "ref_source='npz' requires 'motion_file' in whole_body_control.yaml"
                )
            motion_path = (
                motion_file
                if os.path.isabs(motion_file)
                else os.path.join(current_dir, "model", motion_file)
            )
            if not os.path.exists(motion_path):
                raise FileNotFoundError(f"motion_file not found: {motion_path}")
            npz = np.load(motion_path)
            self._npz_joint_pos = npz["joint_pos"].astype(np.float32)
            self._npz_joint_vel = npz["joint_vel"].astype(np.float32)
            anchor_idx = int(config.get("npz_anchor_body_index", 0))
            self._npz_anchor_pos_w = npz["body_pos_w"][:, anchor_idx, :].astype(np.float32)
            self._npz_anchor_quat_w = npz["body_quat_w"][:, anchor_idx, :].astype(np.float32)
            self._npz_num_frames = int(self._npz_joint_pos.shape[0])
            if self.motion_length is None:
                self.motion_length = self._npz_num_frames
            else:
                self.motion_length = int(self.motion_length)
            # NPZ is "always available"; suppress the GMR missing-frame warning path.
            self._ref_ever_received = True

    # ---------- FSM lifecycle ----------
    def on_enter(self):
        self.counter_step = 0
        self._enter_robot_time = float(self.robot_data_.time_now_)
        self._last_inference_tick = -1
        self.qj_obs[:] = 0.0
        self.dqj_obs[:] = 0.0
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros((1, self.num_actions), dtype=np.float32)
        for _buf in (
            self._buf_cmd, self._buf_ori, self._buf_ang_vel,
            self._buf_joint_pos, self._buf_joint_vel, self._buf_actions,
        ):
            _buf[:] = 0.0
        self._history_initialized = False
        self._last_missing_warn_t = 0.0
        if self.ref_source == "npz":
            # Seed refs from frame 0; same as lingdao's on_enter.
            self.ref_joint_pos = self._npz_joint_pos[0:1]
            self.ref_joint_vel = self._npz_joint_vel[0:1]
            self.ref_anchor_pos_w = self._npz_anchor_pos_w[0]
            self.ref_anchor_quat_w = self._npz_anchor_quat_w[0]
            self._ref_ever_received = True
        else:
            self._ref_ever_received = False
            # Seed refs from whatever the interface has already buffered (may be None).
            self._sync_ref_from_robot_data()
        _logger.info(
            "[FSMStateWHOLE_BODY_CONTROL] entered: onnx=%s, ref_source=%s, "
            "nominal inference %.3f Hz (period %.3f ms = decimation=%s * physical_dt=%s)%s",
            os.path.basename(self.onnx_path),
            self.ref_source,
            self.inference_hz_nominal,
            self.inference_period_s * 1000.0,
            self.decimation_,
            self.physical_dt,
            f", npz_frames={self._npz_num_frames}, motion_length={self.motion_length}, "
            f"hold_final={self.hold_final_reference}" if self.ref_source == "npz" else "",
        )

    def on_exit(self):
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.counter_step = 0
        self._history_initialized = False

    def check_transition(self, *args, **kwargs):
        # The source project treats None as "no restriction".  This framework
        # expects a boolean, so True preserves the source-state semantics.
        return True

    def run(self, flag: ControlFlag):
        # The deployment loop may run faster than the policy.  Schedule from
        # the source policy period (physical_dt * decimation) without integer
        # division on accumulated floating-point time.
        elapsed = max(0.0, float(self.robot_data_.time_now_) - self._enter_robot_time)
        inference_tick = int(np.floor((elapsed + 1.0e-9) / self.inference_period_s))
        if inference_tick > self._last_inference_tick:
            self._last_inference_tick = inference_tick
            self.inner_run()
        self.set_kp_kd()

    def set_kp_kd(self):
        self.robot_data_.joint_kp_p_[: self.motor_nums] = self.stiffness_array
        self.robot_data_.joint_kd_p_[: self.motor_nums] = self.damping_array

    # ---------- Reference plumbing ----------
    def _sync_ref_from_robot_data(self) -> bool:
        """Copy the latest GMR frame from robot_data_ into local ref_* buffers.

        Returns True iff a fresh frame is available. When False, the previously
        cached references stay in place so the FSM still has *something* sane to
        feed into the policy / obs (and the loop keeps emitting the last q_d_).
        """
        rd = self.robot_data_
        if (
            getattr(rd, "ref_joint_pos", None) is None
            or getattr(rd, "ref_joint_vel", None) is None
            or getattr(rd, "ref_body_pos_w", None) is None
            or getattr(rd, "ref_body_quat_w", None) is None
        ):
            return False

        jp = np.asarray(rd.ref_joint_pos, dtype=np.float32)
        jv = np.asarray(rd.ref_joint_vel, dtype=np.float32)
        if jp.ndim == 1:
            jp = jp.reshape(1, -1)
        if jv.ndim == 1:
            jv = jv.reshape(1, -1)
        if jp.shape[-1] != self.num_actions or jv.shape[-1] != self.num_actions:
            return False

        body_pos = np.asarray(rd.ref_body_pos_w, dtype=np.float32)
        body_quat = np.asarray(rd.ref_body_quat_w, dtype=np.float32)
        # Expected shape (1, num_bodies, 3) / (1, num_bodies, 4); be lenient.
        if body_pos.ndim == 2:
            body_pos = body_pos[np.newaxis, ...]
        if body_quat.ndim == 2:
            body_quat = body_quat[np.newaxis, ...]
        if body_pos.shape[1] <= self.anchor_body_index:
            return False

        self.ref_joint_pos = jp
        self.ref_joint_vel = jv
        self.ref_anchor_pos_w = body_pos[0, self.anchor_body_index, :].astype(np.float32)
        self.ref_anchor_quat_w = body_quat[0, self.anchor_body_index, :].astype(np.float32)
        self._ref_ever_received = True
        return True

    def _advance_npz_reference(self) -> bool:
        """NPZ-replay mode: pick the frame for the current control tick.

        Frame index follows the inference counter (NPZ fps == policy hz == 50).
        ``hold_final_reference`` clamps to ``motion_length-1``; otherwise we
        still clamp to the NPZ length as a safety net so we never index OOB.
        """
        if self._npz_num_frames == 0 or self._npz_joint_pos is None:
            return False
        time_index = max(self.counter_step, 0)
        if (
            self.hold_final_reference
            and self.motion_length is not None
            and self.motion_length > 0
        ):
            time_index = min(time_index, self.motion_length - 1)
        motion_frame = min(time_index, self._npz_num_frames - 1)
        self.ref_joint_pos = self._npz_joint_pos[motion_frame:motion_frame + 1]
        self.ref_joint_vel = self._npz_joint_vel[motion_frame:motion_frame + 1]
        self.ref_anchor_pos_w = self._npz_anchor_pos_w[motion_frame]
        self.ref_anchor_quat_w = self._npz_anchor_quat_w[motion_frame]
        return True

    def _refresh_reference(self) -> bool:
        """Pull the next reference frame into self.ref_* according to ref_source."""
        if self.ref_source == "npz":
            return self._advance_npz_reference()
        return self._sync_ref_from_robot_data()

    def _maybe_warn_missing(self):
        if not self.warn_on_missing_ref:
            return
        now = time.monotonic()
        if now - self._last_missing_warn_t < self.warn_throttle_sec:
            return
        self._last_missing_warn_t = now
        if not self._ref_ever_received:
            _logger.warning(
                "[FSMStateWHOLE_BODY_CONTROL] no GMR reference received yet; holding zero target"
            )
        else:
            _logger.warning(
                "[FSMStateWHOLE_BODY_CONTROL] GMR reference stalled; holding last q_d_"
            )

    # ---------- Geometry helpers (kept identical to lingdao) ----------
    @staticmethod
    def matrix_from_quat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ])

    @staticmethod
    def yaw_quat(q):
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

    # ---------- Main inference step ----------
    def inner_run(self):
        # 1) Pull the next reference frame:
        #    - GMR mode: may stall; on stall keep previous refs (option A).
        #    - NPZ mode: always succeeds (frame indexed by counter_step).
        fresh = self._refresh_reference()
        if not fresh and not self._ref_ever_received:
            # No frame ever seen (GMR upstream silent at boot): keep q_d_ untouched.
            self._maybe_warn_missing()
            return
        if not fresh:
            # GMR stalled: hold last commanded posture, log throttled warning.
            self._maybe_warn_missing()

        robot_quat = self.robot_data_.get_robot_quat()
        qj = self.robot_data_.get_joint_pos()
        qj = qj[self.locked_joint_map]
        qj = qj[self.mj2lab]
        qj = qj - self.default_angles_lab

        ref_anchor_pos_w = self.ref_anchor_pos_w.copy()
        ref_anchor_ori_w = self.ref_anchor_quat_w.copy()

        # 2) First two ticks: lock-in the yaw alignment between the robot and the
        #    reference anchor (same trick as lingdao).
        if self.counter_step < 2:
            init_to_anchor = self.matrix_from_quat(self.yaw_quat(ref_anchor_ori_w))
            world_to_anchor = self.matrix_from_quat(self.yaw_quat(robot_quat))
            self.init_to_world = world_to_anchor @ init_to_anchor.T
            self.counter_step += 1
            return

        robot_rot_mat = self.matrix_from_quat(robot_quat)
        motion_anchor_ori_b = (
            robot_rot_mat.T @ self.init_to_world @ self.matrix_from_quat(ref_anchor_ori_w)
        )

        ang_vel = self.robot_data_.get_angular_velocity()
        dqj = self.robot_data_.get_joint_vel()
        dqj = dqj[self.locked_joint_map]

        command_joint_pos = self.ref_joint_pos
        command_joint_vel = self.ref_joint_vel
        action_for_history = self.action

        command_vec = np.concatenate(
            (command_joint_pos.squeeze(0), command_joint_vel.squeeze(0)),
            dtype=np.float32,
        )

        term_cmd = command_vec.astype(np.float32)
        term_ori = motion_anchor_ori_b[:, :2].reshape(-1).astype(np.float32)
        term_ang_vel = ang_vel.astype(np.float32)
        term_jp = qj.astype(np.float32)
        term_jv = dqj[self.mj2lab].astype(np.float32)
        term_act = np.asarray(action_for_history, dtype=np.float32).reshape(-1)

        if not self._history_initialized:
            self._buf_cmd[:] = term_cmd
            self._buf_ori[:] = term_ori
            self._buf_ang_vel[:] = term_ang_vel
            self._buf_joint_pos[:] = term_jp
            self._buf_joint_vel[:] = term_jv
            self._buf_actions[:] = term_act
            self._history_initialized = True
        else:
            self._buf_cmd = np.roll(self._buf_cmd, -1, axis=0); self._buf_cmd[-1] = term_cmd
            self._buf_ori = np.roll(self._buf_ori, -1, axis=0); self._buf_ori[-1] = term_ori
            self._buf_ang_vel = np.roll(self._buf_ang_vel, -1, axis=0); self._buf_ang_vel[-1] = term_ang_vel
            self._buf_joint_pos = np.roll(self._buf_joint_pos, -1, axis=0); self._buf_joint_pos[-1] = term_jp
            self._buf_joint_vel = np.roll(self._buf_joint_vel, -1, axis=0); self._buf_joint_vel[-1] = term_jv
            self._buf_actions = np.roll(self._buf_actions, -1, axis=0); self._buf_actions[-1] = term_act

        mimic_obs_tensor = np.expand_dims(
            np.concatenate([
                self._buf_cmd.reshape(-1),
                self._buf_ori.reshape(-1),
                self._buf_ang_vel.reshape(-1),
                self._buf_joint_pos.reshape(-1),
                self._buf_joint_vel.reshape(-1),
                self._buf_actions.reshape(-1),
            ]),
            axis=0,
        )

        observation = {self.input_name[0]: mimic_obs_tensor}
        outputs_result = self.ort_session.run(None, observation)
        self.action = outputs_result[0]

        target_dof_pos_mj = np.zeros(self.num_actions)
        target_dof_pos_mj_inner = np.zeros(self.num_actions)
        target_dof_pos_lab = self.action * self.action_scale_lab + self.default_angles_lab
        if target_dof_pos_lab.ndim > 1:
            target_dof_pos_lab = np.squeeze(target_dof_pos_lab, axis=0)

        if self.tau_limit_enabled:
            target_dof_pos_lab = self._apply_tau_limit_lab(target_dof_pos_lab)

        target_dof_pos_mj_inner[self.mj2lab] = target_dof_pos_lab
        target_dof_pos_mj[self.locked_joint_map] = target_dof_pos_mj_inner

        for i in range(self.motor_nums):
            joint_idx = 35 - self.motor_nums + i
            self.robot_data_.q_d_[joint_idx] = target_dof_pos_mj[i]
            self.robot_data_.q_dot_d_[joint_idx] = 0.0
            self.robot_data_.tau_d_[joint_idx] = 0.0

        self.counter_step += 1

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
