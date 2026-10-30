"""FSM state for old and upright-onehot AMP MLP policies."""
from __future__ import annotations

import json
import os

import numpy as np

from xmigcs.policy.onnx_policy_base import ONNXPolicyBase
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData
from xmigcs.common.rotation import (
    euler_single_axis_to_quat,
    euler_to_quaternion_scipy,
    matrix_from_quat,
    quat_mul,
    quat_rotate_inverse_numpy,
)
from xmigcs.common.joint_mapping import MJ_JOINTS_31


class FSMStateAMPMLP(ONNXPolicyBase):
    """AMP MLP policy with ONNX-defined observation layout.

    The root-frame angular velocity and gravity match the MuJoCo web runner.
    Both the legacy 78-dim layout and the 81-dim upright-onehot layout are
    supported without changing this file when the ONNX model is replaced.
    """

    _BASE_OBS_LAYOUT = (
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("velocity_commands", 3),
        ("joint_pos", 23),
        ("joint_vel", 23),
        ("actions", 23),
    )
    _OBS_NAME_ALIASES = {
        "last_action": "actions",
        "generated_commands": "velocity_commands",
    }
    _DEFAULT_WALK_RUN_SPEED_THRESHOLD = 1.5
    _DEFAULT_WALKING_UPRIGHTNESS = 0.7

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "amp_mlp.yaml")
        cfg = self._load_config(config_path)
        self._init_onnx_and_joints(cfg)
        self._init_buffers()
        self._init_term_history_buffers()

    def _init_term_history_buffers(self) -> None:
        """Initialize per-term history buffers matching IsaacLab ObsGroup."""
        self._term_hist_bufs = [
            np.zeros((self.num_hist_, term["size"]), dtype=np.float32)
            for term in self._obs_layout
        ]

    def _init_onnx_and_joints(self, cfg: dict) -> None:
        """Load ONNX model and build joint mapping from ONNX metadata."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "model")
        self._init_onnx_session(model_dir, cfg["model_path"])
        self._parse_onnx_metadata()
        anchor_cfg = cfg.get("anchor", {})
        self._anchor_waist_indices = (
            int(anchor_cfg.get("waist_yaw_idx", 12)),
            int(anchor_cfg.get("waist_roll_idx", 13)),
            int(anchor_cfg.get("waist_pitch_idx", 14)),
        )
        self._configure_observation_layout()
        self._build_joint_mapping(MJ_JOINTS_31, cfg.get('fixed_joints', []))

    @staticmethod
    def _parse_metadata_list(value: str) -> list[str]:
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (TypeError, ValueError):
            pass
        return [
            item.strip().strip("'\"")
            for item in value.strip().strip("[]").split(",")
            if item.strip().strip("'\"")
        ]

    @classmethod
    def _normalize_obs_name(cls, name: str) -> str:
        return cls._OBS_NAME_ALIASES.get(name, name)

    def _infer_obs_size(self, name: str) -> int | None:
        name = self._normalize_obs_name(name)
        if name in ("base_ang_vel", "projected_gravity", "velocity_commands", "amp_condition"):
            return 3
        if name in ("joint_pos", "joint_vel", "actions"):
            return len(self.joint_seq)
        return None

    def _fallback_observation_layout(self, input_dim: int) -> list[dict]:
        base_layout = [
            {"name": name, "size": len(self.joint_seq) if size == 23 else size}
            for name, size in self._BASE_OBS_LAYOUT
        ]
        upright_layout = base_layout + [{"name": "amp_condition", "size": 3}]
        if input_dim % sum(term["size"] for term in upright_layout) == 0:
            return upright_layout
        if input_dim % sum(term["size"] for term in base_layout) == 0:
            return base_layout
        raise ValueError(
            f"ONNX input dimension {input_dim} is incompatible with AMP observation layouts "
            f"({sum(term['size'] for term in base_layout)}/"
            f"{sum(term['size'] for term in upright_layout)})."
        )

    def _configure_observation_layout(self) -> None:
        """Resolve observation terms and history length from ONNX metadata."""
        meta = self.ort_session_.get_modelmeta().custom_metadata_map
        input_dim_value = self.ort_session_.get_inputs()[0].shape[-1]
        if not isinstance(input_dim_value, (int, np.integer)) or input_dim_value <= 0:
            input_dim_value = self.obs_size_ * self.num_hist_
        input_dim = int(input_dim_value)

        layout = None
        raw_layout = meta.get("observation_layout")
        if raw_layout:
            try:
                parsed_layout = json.loads(raw_layout)
                if isinstance(parsed_layout, list) and parsed_layout:
                    layout = [
                        {
                            "name": self._normalize_obs_name(str(term["name"])),
                            "size": int(term["size"]),
                        }
                        for term in parsed_layout
                    ]
            except (KeyError, TypeError, ValueError):
                layout = None

        if layout is None:
            names = self._parse_metadata_list(
                meta.get("observation_names") or meta.get("observation_terms") or ""
            )
            sizes = self._parse_metadata_list(meta.get("observation_term_sizes", ""))
            if names:
                layout = []
                for index, raw_name in enumerate(names):
                    name = self._normalize_obs_name(raw_name)
                    size = int(float(sizes[index])) if index < len(sizes) else self._infer_obs_size(name)
                    if size is None:
                        raise ValueError(f"Unsupported ONNX observation term: {raw_name}")
                    layout.append({"name": name, "size": size})

        if layout is None or input_dim % sum(term["size"] for term in layout) != 0:
            layout = self._fallback_observation_layout(input_dim)

        supported_names = {
            "base_ang_vel", "projected_gravity", "velocity_commands",
            "joint_pos", "joint_vel", "actions", "amp_condition",
        }
        unsupported = [term["name"] for term in layout if term["name"] not in supported_names]
        if unsupported:
            raise ValueError(f"Unsupported ONNX observation terms: {unsupported}")

        single_obs_dim = sum(term["size"] for term in layout)
        self._obs_layout = layout
        self.obs_size_ = single_obs_dim
        self.num_hist_ = input_dim // single_obs_dim
        self._amp_walk_run_speed_threshold = float(
            meta.get("amp_walk_run_speed_threshold", self._DEFAULT_WALK_RUN_SPEED_THRESHOLD)
        )
        self._amp_walking_uprightness = float(
            meta.get("amp_walking_uprightness", self._DEFAULT_WALKING_UPRIGHTNESS)
        )

    # ── State lifecycle ──

    def _reset_internal_state(self) -> None:
        super()._reset_internal_state()
        for buf in self._term_hist_bufs:
            buf.fill(0.0)

    # ── Observation ──

    def compute_observation(self, flag: ControlFlag) -> None:
        gravity = self._get_projected_gravity()
        ang_vel = self._get_ang_vel()
        command = self._get_command(flag)
        qj, dqj = self._get_joint_obs()

        available_terms = {
            "base_ang_vel": ang_vel,
            "projected_gravity": gravity,
            "velocity_commands": command,
            "joint_pos": qj,
            "joint_vel": dqj,
            "actions": self.last_actions_,
        }
        if any(term["name"] == "amp_condition" for term in self._obs_layout):
            available_terms["amp_condition"] = self._compute_amp_condition(command)
        terms = [available_terms[term["name"]] for term in self._obs_layout]

        # Update per-term history buffers and concatenate
        # Each term buffer is (num_hist, term_size) with oldest→newest ordering
        if self.is_first_obs_:
            for buf, val in zip(self._term_hist_bufs, terms):
                buf[:] = val[np.newaxis, :]  # fill all history slots with same value
            self.is_first_obs_ = False
        else:
            for buf, val in zip(self._term_hist_bufs, terms):
                buf[:-1] = buf[1:]  # shift: drop oldest, move newer forward
                buf[-1] = val      # append newest at the end

        self.observations_ = np.clip(
            np.concatenate([buf.flatten() for buf in self._term_hist_bufs]),
            -self.clip_obs_, self.clip_obs_
        )

    # ── Observation helpers (root frame) ──

    def _get_projected_gravity(self) -> np.ndarray:
        """Projected gravity in root (pelvis/IMU) frame — same as SWR."""
        roll, pitch, yaw = (
            float(self.robot_data_.imu_data_[2]),
            float(self.robot_data_.imu_data_[1]),
            float(self.robot_data_.imu_data_[0]),
        )
        q_xyzw = euler_to_quaternion_scipy(roll, pitch, yaw, out_xyzw=True)
        return quat_rotate_inverse_numpy(q_xyzw, np.array([0., 0., -1.], dtype=np.float32))

    def _get_ang_vel(self) -> np.ndarray:
        """Angular velocity in root (pelvis/IMU) frame — direct from IMU gyro."""
        ang_vel = self.robot_data_.get_angular_velocity()
        return (ang_vel * self.ang_vel_scale_).astype(np.float32)

    def _get_anchor_uprightness(self) -> float:
        """Return waist_pitch_link local Z projected onto world Z."""
        root_quat = self.robot_data_.get_robot_quat()
        joint_pos = self.robot_data_.get_joint_pos()
        yaw_idx, roll_idx, pitch_idx = self._anchor_waist_indices
        anchor_quat = quat_mul(
            quat_mul(
                quat_mul(root_quat, euler_single_axis_to_quat(joint_pos[yaw_idx], "z")),
                euler_single_axis_to_quat(joint_pos[roll_idx], "x"),
            ),
            euler_single_axis_to_quat(joint_pos[pitch_idx], "y"),
        )
        anchor_quat = anchor_quat / np.linalg.norm(anchor_quat)
        return float(matrix_from_quat(anchor_quat)[2, 2])

    def _compute_amp_condition(self, command: np.ndarray) -> np.ndarray:
        """Match the web runner's [walk, run, fall] condition."""
        if self._get_anchor_uprightness() < self._amp_walking_uprightness:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if np.linalg.norm(command[:2]) >= self._amp_walk_run_speed_threshold:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # ── Action application ──

    def _apply_action_to_robot(self) -> None:
        target_dof_pos_lab = self.actions_ * self.action_scale_lab + self.default_angles_lab
        if target_dof_pos_lab.ndim > 1:
            target_dof_pos_lab = np.squeeze(target_dof_pos_lab, axis=0)

        target_dof_pos_mj = np.zeros(len(self.joint_xml), dtype=np.float32)
        target_dof_pos_mj[self.lab2mj] = target_dof_pos_lab

        base = self.robot_data_.q_d_.shape[0] - self.motor_num_
        for lab_idx, mj_idx in enumerate(self.lab2mj):
            joint_idx = base + mj_idx
            self.robot_data_.q_d_[joint_idx] = target_dof_pos_mj[mj_idx]
            self.robot_data_.q_dot_d_[joint_idx] = 0.0
            self.robot_data_.tau_d_[joint_idx] = 0.0

        self.last_actions_[:] = self.actions_

    def _apply_pd_gains(self) -> None:
        """Set PD gains for all joints (fixed joints overridden by _apply_fixed_joints)."""
        self.robot_data_.joint_kp_p_[:len(self.joint_xml)] = self.stiffness_array
        self.robot_data_.joint_kd_p_[:len(self.joint_xml)] = self.damping_array
