"""
FSM State Implementations
Concrete implementations of different FSM states
"""

import numpy as np

from xmigcs.FSM.fsm_base import FSMState, FSMStateName
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData
import yaml
import os

class FSMStateSTOP(FSMState):
    """停止状态实现 - 与C++版本完全一致"""

    def __init__(self, robot_data: RobotData):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.STOP
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "stop.yaml")
        with open(config_path, 'r') as f:
            policy_config = yaml.safe_load(f)

        self.motor_num_ = policy_config["motor_num"]

        # Initialize vectors
        self.hold_position_ = np.zeros(self.motor_num_)
        self.kp_pos_ = np.zeros(self.motor_num_)
        self.kd_pos_ = np.zeros(self.motor_num_)

        # Load kp and kd gains from config
        for i in range(self.motor_num_):
            self.kp_pos_[i] = policy_config["kp_pos"][i]
            self.kd_pos_[i] = policy_config["kd_pos"][i]



    def on_enter(self):
        """进入停止状态 - 与C++版本完全一致"""
        # Store the last motor positions as hold positions (equivalent to tail(motor_num_))
        self.hold_position_ = self.robot_data_.q_a_[-self.motor_num_:].copy()

    def run(self, flag: ControlFlag):
        """运行停止状态 - 与C++版本完全一致"""
        # Enforce the hold position for every frame (equivalent to tail(motor_num_))
        self.robot_data_.q_d_[-self.motor_num_:] = self.hold_position_.copy()
        # Set desired joint velocities to zero
        self.robot_data_.q_dot_d_[-self.motor_num_:] = 0.0
        # Set desired torques to zero
        self.robot_data_.tau_d_[-self.motor_num_:] = 0.0

        # Set proportional and derivative gains
        self.robot_data_.joint_kp_p_[:self.motor_num_] = self.kp_pos_.copy()
        self.robot_data_.joint_kd_p_[:self.motor_num_] = self.kd_pos_.copy()

    def on_exit(self):      
        """退出停止状态 - 与C++版本完全一致"""
        self.hold_position_.fill(0.0)

    def check_transition(self, action: str = "", target_state=None, **kwargs) -> bool:
        """检查状态转换"""
        return True