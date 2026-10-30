"""
ONNX Policy Base Class

Common skeleton shared by all ONNX-based RL policy FSM states.
Subclasses must implement: compute_observation, _apply_action_to_robot, _apply_pd_gains.

Fixed joints (not controlled by the ONNX policy) are loaded from dex_config.yaml
and automatically asserted every control step via _apply_fixed_joints().
"""
from abc import abstractmethod

import numpy as np
import onnxruntime as ort
import os
import yaml
from typing import List, Optional, Sequence, Tuple

from xmigcs.FSM.fsm_base import FSMState
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData


class ONNXPolicyBase(FSMState):
    """Base class for ONNX-based RL policy FSM states.

    Provides:
      - YAML config loading with common field extraction
      - ONNX session initialization with warm-up
      - ONNX metadata parsing (joint names, defaults, gains, action_scale)
      - Joint mapping (lab2mj) with scatter into mj-order arrays
      - Fixed joints config loading and assertion (from dex_config.yaml)
      - Observation / action buffer management
      - Default run loop with decimation gating
      - Common on_enter / on_exit / check_transition
    """

    # ── Initialization helpers (called from subclass __init__) ──

    def _load_config(self, config_path: str) -> dict:
        """Load YAML config and extract common fields into self.

        Extracts: actions_size, motor_num, dt, size (observations_size, num_hist),
                  control (decimation), normalization (clip_scales, obs_scales).

        Returns the raw config dict so subclasses can read extra fields.
        """
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.action_num_: int = cfg['actions_size']
        self.motor_num_: int = cfg['motor_num']
        self.dt_: float = cfg['dt']

        size_cfg = cfg['size']
        self.obs_size_: int = size_cfg['observations_size']
        self.num_hist_: int = size_cfg['num_hist']

        ctrl_cfg = cfg['control']
        self.decimation_: int = ctrl_cfg['decimation']

        norm_cfg = cfg['normalization']
        clip_cfg = norm_cfg['clip_scales']
        obs_cfg = norm_cfg['obs_scales']
        self.clip_obs_: float = clip_cfg['clip_observations']
        self.clip_act_: float = clip_cfg['clip_actions']
        self.lin_vel_scale_: float = obs_cfg['lin_vel']
        self.ang_vel_scale_: float = obs_cfg['ang_vel']
        self.dof_pos_scale_: float = obs_cfg['dof_pos']
        self.dof_vel_scale_: float = obs_cfg['dof_vel']

        return cfg

    def _init_onnx_session(self, model_dir: str, model_path: str) -> None:
        """Load ONNX model with standard session options and warm-up inference.

        Args:
            model_dir: Directory containing the model file.
            model_path: Relative path to the .onnx file (joined with model_dir),
                        or an absolute path used as-is.
        """
        if os.path.isabs(model_path):
            self.model_path = model_path
        else:
            self.model_path = os.path.join(model_dir, model_path)

        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.enable_mem_pattern = False
            options.enable_mem_reuse = True

            self.ort_session_ = ort.InferenceSession(
                self.model_path, options, providers=['CPUExecutionProvider']
            )

            # Warm-up: one dummy inference to trigger graph compilation / memory allocation
            input_info = self.ort_session_.get_inputs()[0]
            input_name = input_info.name
            configured_dim = (self.obs_size_ * self.num_hist_
                              if hasattr(self, 'obs_size_') and hasattr(self, 'num_hist_')
                              else 1)
            input_shape = list(input_info.shape)
            dummy_shape = tuple(
                int(dim) if isinstance(dim, (int, np.integer)) and dim > 0
                else configured_dim if index == len(input_shape) - 1
                else 1
                for index, dim in enumerate(input_shape)
            )
            self.ort_session_.run(
                None, {input_name: np.zeros(dummy_shape, dtype=np.float32)}
            )
            print(f"[{self.__class__.__name__}] ONNX model loaded (with warm-up): {self.model_path}")
        except Exception as e:
            print(f"[{self.__class__.__name__}] Failed to load ONNX model: {e}")
            self.ort_session_ = None

    def _parse_onnx_metadata(self) -> None:
        """Parse joint_names, default_joint_pos, joint_stiffness, joint_damping,
        action_scale from ONNX metadata.

        Uses robust CSV parsing that handles brackets and empty entries.
        Raises ValueError if required fields are missing.
        """
        meta = self.ort_session_.get_modelmeta().custom_metadata_map if self.ort_session_ else {}

        # Parse joint_names
        self.joint_seq: Optional[List[str]] = None
        if meta.get("joint_names"):
            raw = meta["joint_names"].strip().strip("[]")
            self.joint_seq = [x.strip() for x in raw.split(",") if x.strip()]
        if not self.joint_seq:
            raise ValueError(
                "ONNX metadata missing required field 'joint_names'. "
                "Re-export the policy with attach_onnx_metadata()."
            )

        # Parse numeric arrays from ONNX metadata
        def _parse_csv(key: str) -> Optional[np.ndarray]:
            if key not in meta or not meta[key]:
                return None
            raw = meta[key].strip().strip("[]")
            return np.array(
                [float(x) for x in raw.split(",") if x.strip()],
                dtype=np.float32,
            )

        self.joint_pos_array_seq = _parse_csv("default_joint_pos")
        self.stiffness_array_seq = _parse_csv("joint_stiffness")
        self.damping_array_seq = _parse_csv("joint_damping")
        self.action_scale = _parse_csv("action_scale")

        missing = [k for k, v in [
            ("default_joint_pos", self.joint_pos_array_seq),
            ("joint_stiffness", self.stiffness_array_seq),
            ("joint_damping", self.damping_array_seq),
            ("action_scale", self.action_scale),
        ] if v is None]
        if missing:
            raise ValueError(
                f"ONNX metadata missing required fields: {missing}. "
                f"Re-export the policy with attach_onnx_metadata()."
            )

        # print(f"[{self.__class__.__name__}] Loaded params from ONNX metadata: {len(self.joint_seq)} joints")
        # print(f"  joint_names: {self.joint_seq}")
        # print(f"  default_joint_pos: {self.joint_pos_array_seq.tolist()}")
        # print(f"  joint_stiffness: {self.stiffness_array_seq.tolist()}")
        # print(f"  joint_damping: {self.damping_array_seq.tolist()}")
        # print(f"  action_scale: {self.action_scale.tolist()}")

    def _build_joint_mapping(self, joint_xml: Sequence[str], fixed_joints_cfg: list = None) -> None:
        """Build lab2mj mapping using joint names from ONNX and the provided joint_xml list.

        Scatters lab arrays into mj-order arrays:
          joint_pos_array, stiffness_array, damping_array.

        Then applies fixed_joints_cfg overrides for joints not controlled by the policy.

        Args:
            joint_xml: MJ-order joint name list (e.g. MJ_JOINTS_31).
            fixed_joints_cfg: List of dicts from the policy's YAML config,
                each with keys: name, target_pos, kp, kd.
                Joints not present in joint_xml are silently skipped.

        Stores: lab2mj, joint_pos_array, stiffness_array, damping_array,
                default_angles_lab, action_scale_lab, joint_seq, _fixed_joints.
        """
        self.joint_xml = list(joint_xml)

        # Build lab2mj: for each lab-order joint, find its index in the mj XML order
        self.lab2mj = np.array(
            [self.joint_xml.index(name) for name in self.joint_seq],
            dtype=int,
        )

        # Scatter lab-dim arrays into mj-order arrays
        n_mj = len(self.joint_xml)
        self.joint_pos_array = np.zeros(n_mj, dtype=np.float32)
        self.stiffness_array = np.zeros(n_mj, dtype=np.float32)
        self.damping_array = np.zeros(n_mj, dtype=np.float32)

        self.joint_pos_array[self.lab2mj] = self.joint_pos_array_seq
        self.stiffness_array[self.lab2mj] = self.stiffness_array_seq
        self.damping_array[self.lab2mj] = self.damping_array_seq

        self.default_angles_lab = self.joint_pos_array_seq
        self.action_scale_lab = self.action_scale

        # Resolve and apply fixed joints config
        self._fixed_joints: list = self._resolve_fixed_joints(
            self.joint_xml, fixed_joints_cfg or [])
        for fj in self._fixed_joints:
            mj_idx = fj['mj_idx']
            self.joint_pos_array[mj_idx] = fj['target_pos']
            self.stiffness_array[mj_idx] = fj['kp']
            self.damping_array[mj_idx] = fj['kd']

    @staticmethod
    def _resolve_fixed_joints(joint_xml: Sequence[str], fixed_joints_cfg: list) -> list:
        """Resolve fixed joints from a policy's YAML config.

        Args:
            joint_xml: MJ-order joint name list.
            fixed_joints_cfg: List of dicts with keys: name, target_pos, kp, kd.

        Returns:
            List of dicts: {'name', 'mj_idx', 'target_pos', 'kp', 'kd'}
            for each fixed joint that exists in joint_xml.
        """
        fixed = []
        for fj_cfg in fixed_joints_cfg:
            name = fj_cfg['name']
            if name in joint_xml:
                fixed.append({
                    'name': name,
                    'mj_idx': joint_xml.index(name),
                    'target_pos': float(fj_cfg.get('target_pos', 0.0)),
                    'kp': float(fj_cfg.get('kp', 0.0)),
                    'kd': float(fj_cfg.get('kd', 0.0)),
                })
        return fixed

    # ── Buffer management ──

    def _init_buffers(self) -> None:
        """Initialize observations, proprio_hist_buf, last_actions, actions buffers."""
        self.observations_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.proprio_hist_buf_ = np.zeros(self.obs_size_ * self.num_hist_, dtype=np.float32)
        self.last_actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.actions_ = np.zeros(self.action_num_, dtype=np.float32)
        self.is_first_obs_: bool = True
        self.is_first_action_: bool = True

    def _reset_internal_state(self) -> None:
        """Reset all buffers to zero and set is_first_obs / is_first_action flags."""
        self.observations_.fill(0.0)
        self.proprio_hist_buf_.fill(0.0)
        self.last_actions_.fill(0.0)
        self.actions_.fill(0.0)
        self.is_first_obs_ = True
        self.is_first_action_ = True

        # Reset robot desired state to default joint positions
        if hasattr(self, 'lab2mj') and hasattr(self, 'joint_pos_array_seq'):
            base = self.robot_data_.q_d_.shape[0] - self.motor_num_
            for lab_idx, mj_idx in enumerate(self.lab2mj):
                self.robot_data_.q_d_[base + mj_idx] = self.joint_pos_array_seq[lab_idx]
                self.robot_data_.q_dot_d_[base + mj_idx] = 0.0
                self.robot_data_.tau_d_[base + mj_idx] = 0.0

        # Assert fixed joints to their target positions
        self._apply_fixed_joints()

    # ── Main run loop ──

    def run(self, flag: ControlFlag) -> None:
        """Default run loop: decimation gate → observe → infer → act → PD gains → fixed joints."""
        if int(self.robot_data_.time_now_ / self.dt_) % self.decimation_ == 0:
            self.compute_observation(flag)
            self.compute_actions()
            self._apply_action_to_robot()
        self._apply_pd_gains()
        self._apply_fixed_joints()

    # ── Action inference ──

    def compute_actions(self) -> None:
        """ONNX inference + clip actions."""
        if self.ort_session_ is None:
            return
        try:
            input_name = self.ort_session_.get_inputs()[0].name
            output_data = self.ort_session_.run(
                None, {input_name: self.observations_.reshape(1, -1).astype(np.float32)}
            )[0][0]
            np.clip(output_data, -self.clip_act_, self.clip_act_, out=self.actions_[:self.action_num_])
            if self.is_first_action_:
                # print(f"[{self.__class__.__name__}] First obs:",
                #       " ".join(f"{v:.6f}" for v in self.observations_))
                self.is_first_action_ = False
        except Exception as e:
            print(f"[{self.__class__.__name__}] ONNX inference error: {e}")

    # ── Fixed joints handling ──

    def _apply_fixed_joints(self) -> None:
        """Assert target position, velocity, torque and PD gains for fixed joints.

        Called every control step after _apply_pd_gains() so that joints
        not controlled by the ONNX policy are always locked at their
        configured target with the correct gains.
        """
        if not hasattr(self, '_fixed_joints') or not self._fixed_joints:
            return
        base = self.robot_data_.q_d_.shape[0] - self.motor_num_
        for fj in self._fixed_joints:
            mj_idx = fj['mj_idx']
            self.robot_data_.q_d_[base + mj_idx] = fj['target_pos']
            self.robot_data_.q_dot_d_[base + mj_idx] = 0.0
            self.robot_data_.tau_d_[base + mj_idx] = 0.0
            self.robot_data_.joint_kp_p_[mj_idx] = fj['kp']
            self.robot_data_.joint_kd_p_[mj_idx] = fj['kd']

    # ── Abstract methods (subclasses must implement) ──

    @abstractmethod
    def compute_observation(self, flag: ControlFlag) -> None:
        """Build observation vector from robot state and control flag.

        Must set self.observations_ and update self.proprio_hist_buf_.
        """
        ...

    @abstractmethod
    def _apply_action_to_robot(self) -> None:
        """Write action outputs to robot_data (different indexing per robot)."""
        ...

    @abstractmethod
    def _apply_pd_gains(self) -> None:
        """Write kp/kd gains to robot_data (different indexing per robot)."""
        ...

    # ── State lifecycle ──

    def on_enter(self) -> None:
        """Reset internal state and print enter message."""
        self._reset_internal_state()
        self.actions_.fill(0.0)
        self.last_actions_.fill(0.0)
        print(f"[{self.__class__.__name__}] enter")

    def on_exit(self) -> None:
        """Print exit message."""
        print(f"[{self.__class__.__name__}] exit")

    def check_transition(self, action: str = "", target_state=None, **kwargs) -> bool:
        """Return True by default (consistent interface, allows all transitions)."""
        return True

    # ── Observation helpers ──

    def _update_proprio_history(self, proprio: np.ndarray) -> None:
        """Shift proprio_hist_buf and append new proprio observation.

        On first observation, fills the entire history buffer with the same value.
        On subsequent observations, shifts left and appends.
        """
        if self.is_first_obs_:
            for i in range(self.num_hist_):
                start_idx = i * self.obs_size_
                end_idx = start_idx + self.obs_size_
                self.proprio_hist_buf_[start_idx:end_idx] = proprio
            self.is_first_obs_ = False
        else:
            shift_size = (self.num_hist_ - 1) * self.obs_size_
            self.proprio_hist_buf_[:shift_size] = self.proprio_hist_buf_[self.obs_size_:]
            self.proprio_hist_buf_[shift_size:] = proprio

        self.observations_ = np.clip(self.proprio_hist_buf_, -self.clip_obs_, self.clip_obs_)

    def _get_joint_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (qj_offset_scaled, dqj_scaled) in lab order.

        qj = (q_mj[lab2mj] - default_angles_lab) * dof_pos_scale_
        dqj = dq_mj[lab2mj] * dof_vel_scale_
        """
        q_mj = self.robot_data_.get_joint_pos()
        dq_mj = self.robot_data_.get_joint_vel()
        qj = (q_mj[self.lab2mj] - self.default_angles_lab) * self.dof_pos_scale_
        dqj = dq_mj[self.lab2mj] * self.dof_vel_scale_
        return qj, dqj

    def _get_command(self, flag: ControlFlag) -> np.ndarray:
        """Extract 3-dim command vector from ControlFlag."""
        return np.array([
            getattr(flag, 'x_speed_command', 0.0),
            getattr(flag, 'y_speed_command', 0.0),
            getattr(flag, 'yaw_speed_command', 0.0),
        ], dtype=np.float32)
