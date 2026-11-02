"""
Robot Data Structure
Python equivalent of the C++ RobotData class
"""
import numpy as np
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.rotation import euler_to_quaternion_scipy, quat_rotate_inverse_numpy
import copy


class RobotData:
    """机器人状态数据结构"""

    def __init__(self, motor_num: int = 31, whole_joint_num: int = 37):
        self.motor_num = motor_num
        self.whole_joint_num = whole_joint_num

        # Joint states (actual)
        self.q_a_ = np.zeros(whole_joint_num)  # Joint positions
        self.q_dot_a_ = np.zeros(whole_joint_num)  # Joint velocities
        self.tau_a_ = np.zeros(whole_joint_num)  # Joint torques
        self.temperature_a = np.zeros(motor_num)  # Joint temperatures
        self.q_a_last = np.zeros(whole_joint_num)  # 上一时刻关节位置
        self.qdot_a_last = np.zeros(whole_joint_num)  # 上一时刻关节速度
        self.tor_a_last = np.zeros(whole_joint_num)  # 上一时刻关节力矩

        # Joint commands (desired)
        self.q_d_ = np.zeros(whole_joint_num)  # Desired joint positions
        self.q_d_s_ = np.zeros(
            whole_joint_num)  # Desired serial joint positions
        self.q_dot_d_ = np.zeros(whole_joint_num)  # Desired joint velocities
        self.tau_d_ = np.zeros(whole_joint_num)  # Desired joint torques

        # Control gains
        self.joint_kp_p_ = np.zeros(motor_num)  # Proportional gains
        self.joint_kd_p_ = np.zeros(motor_num)  # Derivative gains

        self.joint_kp_s_ = np.zeros(motor_num)  # Serial proportional gains
        self.joint_kd_s_ = np.zeros(motor_num)  # Serial derivative gains

        # IMU data: [yaw, pitch, roll, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
        self.imu_data_ = np.zeros(13)

        # Timing
        self.time_now_ = 0.0

        # Configuration
        self.config_file_ = ""

        # control cmd
        # walk command
        self.walk_cmd_ = np.zeros(3)  # x_speed, y_speed, yaw_speed
        # walk height command
        self.walk_height_cmd_ = 0.0
        # floating base command
        self.floating_base_cmd_ = np.zeros(
            4)  # waist_roll, waist_pitch, waist_yaw, height_cmd
        # footpoint command
        self.footpoint_cmd_ = np.array([False, 0, 0, 0],
                                       dtype=object)  # footflag,x,y,yaw
        # 状态切换时间
        self.trans_start_time = 0.0
        # 上一次状态的最后命令
        self.q_d_s_last_state = np.zeros(whole_joint_num)

        # 诊断信息: 电量，关节温度等
        self.diagnostic_info = {'battery_level': 100.0, 'joint_temperatures': self.temperature_a, 'motor_status': {}}

        # Source deployment GMR reference; populated by RobotInterface.
        self.ref_body_pos_w = None
        self.ref_body_quat_w = None
        self.ref_body_lin_vel_w = None
        self.ref_body_ang_vel_w = None
        self.ref_joint_pos = None
        self.ref_joint_vel = None
        # Monotonic times: represented interpolated sample and latest receipt.
        # Existing I/V consumers retain the same reference arrays.
        self.ref_sample_time = None
        self.ref_received_time = None

    def copy_from(self, other: 'RobotData'):
        """从另一个RobotData对象复制数据"""
        self.q_a_[:] = other.q_a_[:]
        self.q_dot_a_[:] = other.q_dot_a_[:]
        self.tau_a_[:] = other.tau_a_[:]
        self.q_d_[:] = other.q_d_[:]
        self.q_dot_d_[:] = other.q_dot_d_[:]
        self.tau_d_[:] = other.tau_d_[:]
        self.joint_kp_p_[:] = other.joint_kp_p_[:]
        self.joint_kd_p_[:] = other.joint_kd_p_[:]
        self.imu_data_[:] = other.imu_data_[:]
        self.time_now_ = other.time_now_
        self.config_file_ = other.config_file_
        self.walk_cmd_[:] = other.walk_cmd_[:]
        self.floating_base_cmd_[:] = other.floating_base_cmd_[:]

    def get_joint_pos(self) -> np.ndarray:
        joint_start_idx = self.whole_joint_num - self.motor_num
        joint_pos = self.q_a_[joint_start_idx:].astype(np.float32)
        return joint_pos

    def get_desired_joint_pos(self) -> np.ndarray:
        joint_start_idx = self.whole_joint_num - self.motor_num
        desired_joint_pos = self.q_d_[joint_start_idx:].astype(np.float32)
        return desired_joint_pos

    def get_serial_joint_pos_desired(self) -> np.ndarray:
        joint_start_idx = self.whole_joint_num - self.motor_num
        joint_pos_desired = self.q_d_s_[joint_start_idx:].astype(np.float32)
        return joint_pos_desired

    def get_joint_vel(self) -> np.ndarray:
        joint_start_idx = self.whole_joint_num - self.motor_num
        joint_vel = self.q_dot_a_[joint_start_idx:].astype(np.float32)
        return joint_vel

    def get_angular_velocity(self) -> np.ndarray:
        omega_xyz = np.array(
            [self.imu_data_[3], self.imu_data_[4], self.imu_data_[5]],
            dtype=np.float32)
        return omega_xyz

    def get_robot_quat(self):
        rpy = np.array(
            [
                self.imu_data_[2],  # roll
                self.imu_data_[1],  # pitch
                self.imu_data_[0]  # yaw
            ],
            dtype=np.float32) * 1.0
        robot_quat_wxyz = euler_to_quaternion_scipy(
            rpy[0], rpy[1], rpy[2])
        return robot_quat_wxyz

    def get_waist_yrp(self) -> np.ndarray:
        joint_pos = self.get_joint_pos()
        waist_yaw, waist_roll, waist_pitch = joint_pos[12], joint_pos[
            13], joint_pos[14]
        return np.array([waist_yaw, waist_roll, waist_pitch], dtype=np.float32)

    def get_base_linear_acceleration(self) -> np.ndarray:
        lin_acc = np.array(
            [self.imu_data_[6], self.imu_data_[7], self.imu_data_[8]],
            dtype=np.float32)
        return lin_acc

    def get_project_gravity(self) -> np.ndarray:
        """根据机器人姿态计算重力投影"""
        robot_quat_wxyz = self.get_robot_quat()
        robot_quat_xyzw = np.array([
            robot_quat_wxyz[1], robot_quat_wxyz[2], robot_quat_wxyz[3],
            robot_quat_wxyz[0]
        ])
        g = np.array([0., 0., -1.])
        projected_gravity = quat_rotate_inverse_numpy(robot_quat_xyzw, g)
        return projected_gravity

    def get_walk_cmd(self) -> np.ndarray:
        """获取行走命令: [x_speed, y_speed, yaw_speed]"""
        return self.walk_cmd_

    def get_floating_base_cmd(self) -> np.ndarray:
        """获取浮动基座命令: [x, y, z, roll, pitch, yaw]"""
        return self.floating_base_cmd_

    def get_footpoint_cmd(self) -> np.ndarray:
        """获取足点命令: [footflag, x, y, yaw]"""
        return self.footpoint_cmd_

    def record_transition(self):
        """设置转换开始时间
        """
        self.trans_start_time = self.time_now_
        self.q_d_s_last_state = self.q_d_s_.copy()
        self.joint_kp_s_last_state = self.joint_kp_s_.copy()
        self.joint_kd_s_last_state = self.joint_kd_s_.copy()

    def get_last_state_serial_qd(self) -> np.ndarray:
        """获取上一个状态输出的串联关节指令"""
        return self.q_d_s_last_state[self.whole_joint_num - self.motor_num:].astype(np.float32)
    
    def get_last_state_serial_kp(self) -> np.ndarray:
        """获取上一个状态输出的串联关节刚度"""
        return self.joint_kp_s_last_state
    
    def get_last_state_serial_kd(self) -> np.ndarray:
        """获取上一个状态输出的串联关节阻尼"""
        return self.joint_kd_s_last_state
    
    def get_joint_kp(self) -> np.ndarray:
        """获取关节刚度"""
        return self.joint_kp_p_
    
    def get_joint_kd(self) -> np.ndarray:
        """获取关节阻尼"""
        return self.joint_kd_p_
    
    def set_walk_cmd(self, walk_cmd: np.ndarray, xyyaw_speed_limits: dict):
        """设置行走命令: [x_speed, y_speed, yaw_speed]"""
        self.walk_cmd_[:] = walk_cmd[:]
        self.walk_cmd_[0] = np.clip(self.walk_cmd_[0] + xyyaw_speed_limits["x_command_offset"], -xyyaw_speed_limits["max_x_minus"], xyyaw_speed_limits["max_x_plus"])
        self.walk_cmd_[1] = np.clip(self.walk_cmd_[1] + xyyaw_speed_limits["y_command_offset"], -xyyaw_speed_limits["max_y"], xyyaw_speed_limits["max_y"])
        self.walk_cmd_[2] = np.clip(self.walk_cmd_[2] + xyyaw_speed_limits["yaw_command_offset"], -xyyaw_speed_limits["max_yaw"], xyyaw_speed_limits["max_yaw"])
    
    def set_walk_height_cmd(self, walk_height_cmd: float):
        """设置行走高度命令"""
        self.walk_height_cmd_ = walk_height_cmd
    
    def set_floating_base_cmd(self, floating_base_cmd: np.ndarray):
        """设置浮动基座命令: [roll, pitch, yaw, height]"""
        self.floating_base_cmd_ = floating_base_cmd

    # ── Semantic motor command writers ──
    # These methods encapsulate the index arithmetic that was previously
    # duplicated across every policy module (base = whole_joint_num - motor_num).

    @property
    def _motor_offset(self) -> int:
        """Index where motor joints start in the whole_joint_num arrays."""
        return self.whole_joint_num - self.motor_num

    def set_motor_position(self, motor_idx: int, position: float) -> None:
        """Set desired position for a single motor joint (0-based motor index)."""
        self.q_d_[self._motor_offset + motor_idx] = position

    def set_motor_velocity(self, motor_idx: int, velocity: float) -> None:
        """Set desired velocity for a single motor joint (0-based motor index)."""
        self.q_dot_d_[self._motor_offset + motor_idx] = velocity

    def set_motor_torque(self, motor_idx: int, torque: float) -> None:
        """Set desired torque for a single motor joint (0-based motor index)."""
        self.tau_d_[self._motor_offset + motor_idx] = torque

    def set_motor_pd_gains(self, motor_idx: int, kp: float, kd: float) -> None:
        """Set PD gains for a single motor joint (0-based motor index)."""
        self.joint_kp_p_[motor_idx] = kp
        self.joint_kd_p_[motor_idx] = kd

    def set_all_motor_commands(self, q_d: np.ndarray, q_dot_d: np.ndarray = None,
                                tau_d: np.ndarray = None) -> None:
        """Set position/velocity/torque commands for all motor joints at once.

        Args:
            q_d: desired positions (motor_num elements)
            q_dot_d: desired velocities (motor_num elements), defaults to zeros
            tau_d: desired torques (motor_num elements), defaults to zeros
        """
        base = self._motor_offset
        self.q_d_[base:] = q_d
        self.q_dot_d_[base:] = q_dot_d if q_dot_d is not None else 0.0
        self.tau_d_[base:] = tau_d if tau_d is not None else 0.0

    def set_all_pd_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """Set PD gains for all motor joints at once.

        Args:
            kp: proportional gains (motor_num elements)
            kd: derivative gains (motor_num elements)
        """
        self.joint_kp_p_[:] = kp
        self.joint_kd_p_[:] = kd
