"""
FSM State Implementations
Concrete implementations of different FSM states
"""
import numpy as np
from xmigcs.FSM.fsm_base import FSMState, FSMStateName
from xmigcs.common.joystick import ControlFlag
from xmigcs.utils.logging_utils import get_logger
from xmigcs.common.robot_data import RobotData
import os
import yaml

logger = get_logger(__name__)


class FSMStateSTANDZERO(FSMState):
    """站立零位状态实现"""

    def __init__(
        self,
        robot_data: RobotData,
        config_path: str = None,
        state_name: str = "STANDZERO",
    ):
        super().__init__(robot_data)
        self.state_name_ = state_name
        self.q_factor_ = 0.0
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "config", "standzero.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)
        self.motor_num_ = policy_config["motor_num"]
        self.zero_positions_ = np.array(policy_config["zero_positions"],
                                        dtype=float)
        self.kp_pos_ = np.array(policy_config["kp_pos"], dtype=float)
        self.kd_pos_ = np.array(policy_config["kd_pos"], dtype=float)
        self.use_split_interp_ = (
            "interp_step_hand" in policy_config
            or "interp_step_other" in policy_config
        )
        if self.use_split_interp_:
            self.interp_step_hand_ = float(
                policy_config["interp_step_hand"]
            )
            self.interp_step_other_ = float(
                policy_config["interp_step_other"]
            )
            self.q_factor_hand_ = 0.0
            self.q_factor_other_ = 0.0
        else:
            self.interp_step_ = float(policy_config["interp_step"])
        self.interp_max_ = float(policy_config["interp_max"])
        for name, values in (
            ("zero_positions", self.zero_positions_),
            ("kp_pos", self.kp_pos_),
            ("kd_pos", self.kd_pos_),
        ):
            if values.shape != (self.motor_num_,):
                raise ValueError(
                    f"{self.state_name_} {name} length {values.size} "
                    f"must match motor_num {self.motor_num_}"
                )

    def on_enter(self):
        self.q_factor_ = 0.0
        if self.use_split_interp_:
            self.q_factor_hand_ = 0.0
            self.q_factor_other_ = 0.0

    def run(self, flag: ControlFlag):
        q_est = self.robot_data_.q_a_[-self.motor_num_:].copy()  # numpy数组切片
        if self.use_split_interp_:
            pos_cmd = q_est.copy()
            hand_slice = slice(15, 29)
            other_indices = np.r_[0:15, 29:self.motor_num_]

            if self.q_factor_hand_ < self.interp_max_:
                pos_cmd[hand_slice] = (
                    (1.0 - self.q_factor_hand_) * q_est[hand_slice]
                    + self.q_factor_hand_ * self.zero_positions_[hand_slice]
                )
                self.q_factor_hand_ = min(
                    self.q_factor_hand_ + self.interp_step_hand_,
                    self.interp_max_,
                )
            else:
                pos_cmd[hand_slice] = self.zero_positions_[hand_slice]

            if self.q_factor_other_ < self.interp_max_:
                pos_cmd[other_indices] = (
                    (1.0 - self.q_factor_other_) * q_est[other_indices]
                    + self.q_factor_other_
                    * self.zero_positions_[other_indices]
                )
                self.q_factor_other_ = min(
                    self.q_factor_other_ + self.interp_step_other_,
                    self.interp_max_,
                )
            else:
                pos_cmd[other_indices] = self.zero_positions_[other_indices]
        else:
            if self.q_factor_ < self.interp_max_:
                pos_cmd = (
                    (1.0 - self.q_factor_) * q_est
                    + self.q_factor_ * self.zero_positions_
                )
                self.q_factor_ = min(
                    self.q_factor_ + self.interp_step_,
                    self.interp_max_,
                )
            else:
                pos_cmd = self.zero_positions_
        self.robot_data_.q_d_[-self.motor_num_:] = pos_cmd
        self.robot_data_.q_dot_d_[-self.motor_num_:] = 0
        self.robot_data_.tau_d_[-self.motor_num_:] = 0
        self.robot_data_.joint_kp_p_[:self.motor_num_] = self.kp_pos_
        self.robot_data_.joint_kd_p_[:self.motor_num_] = self.kd_pos_

    def on_exit(self):
        pass

    def check_transition(self, action: str = "", target_state=None, **kwargs) -> bool:
        """零位状态不附加姿态条件，是否可切换完全由状态机转换表决定。"""
        return True
