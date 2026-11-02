"""
Robot Interface
Python equivalent of the C++ RobotInterface class
"""
from __future__ import annotations
import queue
from xmigcs.common.peekqueue import PeekableQueue
import yaml
import os
from types import SimpleNamespace
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np
# ROS messages
# from bodyctrl_msgs.msg import MotorStatusMsg, CmdMotorCtrl, MotorCtrl, Imu, MotorStatus
import rclpy
from std_msgs.msg import String, Float32MultiArray
from rclpy.node import Node
from rclpy.qos import QoSProfile
# TF2 imports for transformations
from tf2_ros import TransformListener, Buffer
import tf2_ros
import transforms3d as t3d
from xmigcs.common.body_id_map import BodyServoIdMap
from xmigcs.common.robot_data import RobotData
from xmigcs.common.gmr_shared_receiver import GMRSharedReceiver
import functools
import time
import math
# from common.func_sp_trans_dex_opt import OptimizedFuncSPTrans as FuncSPTrans
# from xmigcs.common.func_sp_trans_evt_opt import OptimizedFuncSPTrans as FuncSPTrans
# from sptlib_python import funcSPTrans as FuncSPTrans
from xmigcs.common.func_sp_trans_evt_analytic import AnalyticFuncSPTrans as FuncSPTrans

from xmigcs.common.joystick import ControlFlag
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64
from xmigcs.FSM.fsm_base import FSMStateName
from sensor_msgs.msg import Joy
from xmigcs.common.joystick import JoystickHumanoid
from xmigcs.common.xbox_control import XBOXController
from xmigcs.common.stdin_keyboard_control import KeyboardController
from xmigcs.common.dynamic_model import DynamicModel
from diagnostic_msgs.msg import KeyValue
from diagnostic_msgs.msg import DiagnosticStatus
from xmigcs.utils.xlog_utils import xlog
from xmigcs.utils.logging_utils import get_logger
from xmigcs.common.communication_adapter import CommunicationAdapter
from xmigcs.common.motor_command_utils import convert_to_motor_commands, interpolate_action
logger = get_logger(__name__)

# 避免在没有导航消息的环境中报错
class FootPoint: pass
try:
    from hric_msgs.msg import FootPoint
    has_navigate_msg = True
except ImportError:
    has_navigate_msg = False

import threading
# ros2_bridge_msgs 可选导入（与 bodyctrl_msgs 二选一用于 ROS2 底层）
try:
    from ros2_bridge_msgs.msg import (
        RobotState, ArmCtrl, LegCtrl, WaistCtrl, HeadCtrl,
        LegStatus, MotorStatus as BridgeMotorStatus, MotorCtrl as BridgeMotorCtrl,
        ArmStatus, WaistStatus, ImuStatus,
    )
    _ros2_bridge_msgs_available = True
except ImportError:
    _ros2_bridge_msgs_available = False

# 全局变量，用于存储动态导入的模块
_imported_modules = {
    'bodyctrl_msgs': None,
    'ros2_bridge_msgs': None,  # 用于 comm_mode == 'ros2_bridge'
    'robot_control_lpc_python': None,      # LPC 模块名（LPC - Local Procedure Call）
    'robot_control_shm_rpc_python': None,  # SHM RPC 模块名（Shared Memory RPC - ICEORYX2）
    'robot_control_tcp_rpc_python': None   # TCP RPC 模块名（TCP RPC - YLT）
}

def import_bodyctrl_msgs():
    """
    动态导入bodyctrl_msgs模块
    """
    global _imported_modules
    if _imported_modules['bodyctrl_msgs'] is not None:
        return _imported_modules['bodyctrl_msgs']

    # 如果 Python bridge 已经导入，则不能导入bodyctrl_msgs
    if (_imported_modules['robot_control_lpc_python'] is not None or
        _imported_modules['robot_control_shm_rpc_python'] is not None or
        _imported_modules['robot_control_tcp_rpc_python'] is not None):
        raise RuntimeError("Cannot import bodyctrl_msgs when Python bridge is already imported")

    try:
        from bodyctrl_msgs.msg import MotorStatusMsg, CmdMotorCtrl, MotorCtrl, Imu, MotorStatus
        _imported_modules['bodyctrl_msgs'] = {
            'MotorStatusMsg': MotorStatusMsg,
            'CmdMotorCtrl': CmdMotorCtrl,
            'MotorCtrl': MotorCtrl,
            'Imu': Imu,
            'MotorStatus': MotorStatus
        }
        return _imported_modules['bodyctrl_msgs']
    except ImportError as e:
        _imported_modules['bodyctrl_msgs'] = None
        return None

def import_control_sdk_lpc_python_bridge():
    """
    动态导入 robot_control_lpc_python 模块（LPC - Local Procedure Call）
    """
    global _imported_modules

    # 如果已经导入，直接返回
    if _imported_modules['robot_control_lpc_python'] is not None:
        return _imported_modules['robot_control_lpc_python']

    # 如果bodyctrl_msgs已经导入，则不能导入 Python bridge
    if _imported_modules['bodyctrl_msgs'] is not None:
        raise RuntimeError("Cannot import robot_control_lpc_python when bodyctrl_msgs is already imported")

    # 如果其他 bridge 已经导入，则不能导入 LPC bridge
    if (_imported_modules['robot_control_shm_rpc_python'] is not None or
        _imported_modules['robot_control_tcp_rpc_python'] is not None):
        raise RuntimeError("Cannot import robot_control_lpc_python when other RPC bridges are already imported")

    # 导入 robot_control_lpc_python 模块
    try:
        import robot_control_lpc_python
        _imported_modules['robot_control_lpc_python'] = robot_control_lpc_python
        return _imported_modules['robot_control_lpc_python']
    except ImportError as e:
        _imported_modules['robot_control_lpc_python'] = None
        return None

def import_control_sdk_shm_rpc_python_bridge():
    """
    动态导入 robot_control_shm_rpc_python 模块（SHM RPC - Shared Memory RPC，使用 ICEORYX2）
    """
    global _imported_modules

    # 如果已经导入，直接返回
    if _imported_modules['robot_control_shm_rpc_python'] is not None:
        return _imported_modules['robot_control_shm_rpc_python']

    # 如果bodyctrl_msgs已经导入，则不能导入 Python bridge
    if _imported_modules['bodyctrl_msgs'] is not None:
        raise RuntimeError("Cannot import robot_control_shm_rpc_python when bodyctrl_msgs is already imported")

    # 如果其他 bridge 已经导入，则不能导入 SHM RPC bridge
    if (_imported_modules['robot_control_lpc_python'] is not None or
        _imported_modules['robot_control_tcp_rpc_python'] is not None):
        raise RuntimeError("Cannot import robot_control_shm_rpc_python when other bridges are already imported")

    # 导入 robot_control_shm_rpc_python 模块
    try:
        import robot_control_shm_rpc_python
        _imported_modules['robot_control_shm_rpc_python'] = robot_control_shm_rpc_python
        return _imported_modules['robot_control_shm_rpc_python']
    except ImportError as e:
        _imported_modules['robot_control_shm_rpc_python'] = None
        return None

def import_control_sdk_tcp_rpc_python_bridge():
    """
    动态导入 robot_control_tcp_rpc_python 模块（TCP RPC - TCP Socket RPC，使用 YLT）
    """
    global _imported_modules

    # 如果已经导入，直接返回
    if _imported_modules['robot_control_tcp_rpc_python'] is not None:
        return _imported_modules['robot_control_tcp_rpc_python']

    # 如果bodyctrl_msgs已经导入，则不能导入 Python bridge
    if _imported_modules['bodyctrl_msgs'] is not None:
        raise RuntimeError("Cannot import robot_control_tcp_rpc_python when bodyctrl_msgs is already imported")

    # 如果其他 bridge 已经导入，则不能导入 TCP RPC bridge
    if (_imported_modules['robot_control_lpc_python'] is not None or
        _imported_modules['robot_control_shm_rpc_python'] is not None):
        raise RuntimeError("Cannot import robot_control_tcp_rpc_python when other bridges are already imported")

    # 导入 robot_control_tcp_rpc_python 模块
    try:
        import robot_control_tcp_rpc_python
        _imported_modules['robot_control_tcp_rpc_python'] = robot_control_tcp_rpc_python
        return _imported_modules['robot_control_tcp_rpc_python']
    except ImportError as e:
        _imported_modules['robot_control_tcp_rpc_python'] = None
        return None

def get_imported_modules():
    """
    获取当前导入的模块信息
    """
    if _imported_modules['robot_control_lpc_python']:
        return 'robot_control_lpc_python'
    elif _imported_modules['robot_control_shm_rpc_python']:
        return 'robot_control_shm_rpc_python'
    elif _imported_modules['robot_control_tcp_rpc_python']:
        return 'robot_control_tcp_rpc_python'
    elif _imported_modules['ros2_bridge_msgs']:
        return 'ros2_bridge_msgs'
    elif _imported_modules['bodyctrl_msgs']:
        return 'bodyctrl_msgs'
    else:
        return None


def timing_decorator(func):
    """
    装饰器：记录函数执行时间
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result
    return wrapper


class RobotInterface(ABC):
    """机器人接口抽象基类"""

    def __init__(self, robot_data: RobotData):
        self.robot_data_ = robot_data

    @abstractmethod
    def init(self, node: Node):
        """初始化接口"""
        pass

    @abstractmethod
    def update_robot_data(self, time_passed: float):
        """更新机器人状态"""
        pass


    @abstractmethod
    def send_motor_commands(self):
        """发布电机控制命令"""
        pass



class RobotInterfaceImpl(RobotInterface):
    """机器人接口具体实现"""

    def __init__(self, robot_data: RobotData, config_path: str = ''):
        super().__init__(robot_data)
        self.initialized = False
        self.node = None
        self.config_path = config_path

        # ID映射
        self.id_map = BodyServoIdMap()
        self.id_map.body_can_id_map_init()

        # 消息队列
        self.queue_leg_motor_state = PeekableQueue(maxsize=1)
        self.queue_arm_motor_state = PeekableQueue(maxsize=1)
        self.queue_waist_motor_state = PeekableQueue(maxsize=1)
        self.queue_head_motor_state = PeekableQueue(maxsize=1)

        self.queue_imu_xsens = PeekableQueue(maxsize=1)
        self.queue_walk_cmd = PeekableQueue(maxsize=1)
        self.queue_walk_height_cmd = PeekableQueue(maxsize=1)
        self.queue_stand_cmd = PeekableQueue(maxsize=1)
        self.queue_footpoint_cmd = PeekableQueue(maxsize=1)
        self.queue_fsm_state_cmd = PeekableQueue(maxsize=1)
        self.queue_fsm_resume_cmd = PeekableQueue(maxsize=1)

        # joysticks 消息队列
        self.queue_joy_cmd = queue.Queue(maxsize=1)
        self.queue_xbox_cmd = queue.Queue(maxsize=1)
        self.control_flag = ControlFlag()

        # 关节维度
        self.floating_base_dof = 6
        self.whole_joint_nums = self.id_map.whole_motor_nums + self.floating_base_dof

        # 临时变量用于优化计算
        self.temp_q = np.empty(self.id_map.whole_motor_nums)
        # 预分配另一个用于存储中间计算的临时数组
        self._temp_zero_cnt = np.empty(self.id_map.whole_motor_nums)

        # 电机控制参数
        self.motor_dir = np.ones(self.id_map.whole_motor_nums)
        self.zero_offset = np.zeros(self.id_map.whole_motor_nums)

        # 添加标志位，用于跟踪是否是首次接收数据
        self.first_leg_data_received = False
        self.first_arm_data_received = False
        self.first_waist_data_received = False

        # 关节限位
        self.joint_limits = {}
        self.joint_pos_threshold = math.pi
        self.joint_vel_threshold = 30  # rad/s
        # 串并联转换器（仅真机使用）
        self.fun_s2p = FuncSPTrans() if FuncSPTrans is not None else None

        # 串并联转换相关变量
        self.left_ankle_indices = np.array([4, 5]) + self.floating_base_dof
        self.right_ankle_indices = np.array([10, 11]) + self.floating_base_dof
        self.q_a_p = np.zeros(4)  # 并联关节位置
        self.qdot_a_p = np.zeros(4)  # 并联关节速度
        self.tor_a_p = np.zeros(4)  # 并联关节力矩
        self.ankle_kp_p = np.zeros(4)  # 并联关节刚度
        self.ankle_kd_p = np.zeros(4)  # 并联关节阻尼

        # TF相关属性
        self.tf_buffer = None
        self.tf_listener = None

        # ROS publishers and subscribers
        self.pub_leg_motor_cmd = None
        self.pub_arm_motor_cmd = None
        self.pub_waist_motor_cmd = None
        self.pub_head_motor_cmd = None   # 头部控制 publisher
        self.sub_leg_state = None
        self.sub_arm_state = None
        self.sub_waist_state = None

        # 当前机器人所处状态
        self.current_state: FSMStateName = FSMStateName.STOP
        self.last_state: FSMStateName = FSMStateName.STOP
        self.fsm_state_command = ""
        self.fsm_resume_command = ""

        # 动力学模型
        self.dynamic_model = DynamicModel(robot_data=self.robot_data_)
        self._cached_transition_start_time = None
        self._cached_last_state_serial_qd = None
        self._cached_last_state_serial_kp = None
        self._cached_last_state_serial_kd = None
        # Real-time GMR teleoperation reference stream.  Keep the source
        # project's 232-float protocol and interpolation cache layout.
        self._gmr_num_bodies = 24
        self._gmr_num_actions = 29
        self._gmr_expected_len = (
            self._gmr_num_bodies * 3
            + self._gmr_num_bodies * 4
            + self._gmr_num_actions * 2
            + 6
        )
        self.queue_realtime_gmr_state = queue.Queue(maxsize=32)
        self._gmr_interp_delay_sec = 0.012
        self._gmr_prev_frame = None
        self._gmr_curr_frame = None
        self._gmr_prev_time = None
        self._gmr_curr_time = None
        self._gmr_warned_invalid_len = False

        # The shared receiver helper from the source project carries a dormant
        # /gmr_6kp slot.  TELEOP_YIBO does not consume it, but retaining the
        # slot lets the helper stay source-compatible.
        self._gmr_6kp_num_bodies = 26
        self._gmr_6kp_expected_len = self._gmr_6kp_num_bodies * 3 + self._gmr_6kp_num_bodies * 4
        self._gmr_shared_receiver = None
        self._gmr_shared_enabled = False
        self._gmr_shared_pending_gmr = None
        self._gmr_shared_pending_gmr_6kp = None

        self.sdk_stop_event = threading.Event()
        self._command_lock = threading.Lock()
        # Serialize synchronous I/U publishing with the background sender.
        self._command_publish_lock = threading.Lock()
        self._latest_command_frame = None
        self._command_thread = None
        self._command_stop_event = threading.Event()
        self._high_level_timer = None
        self.motor_command_period_s = None
        self.high_level_publish_period_s = None
        self.command_stats = {
            'generated': 0,
            'published': 0,
            'last_publish_s': 0.0,
            'max_publish_s': 0.0,
        }
        self.sensor_stats = {
            'imu': {'updates': 0, 'stale_cycles': 0},
            'leg': {'updates': 0, 'stale_cycles': 0},
            'arm': {'updates': 0, 'stale_cycles': 0},
            'waist': {'updates': 0, 'stale_cycles': 0},
            'head': {'updates': 0, 'stale_cycles': 0},
        }

    @staticmethod
    def _put_latest(buffer, message):
        """Publish a state sample without blocking the control thread."""
        if hasattr(buffer, 'put_latest'):
            buffer.put_latest(message)
            return
        try:
            buffer.put_nowait(message)
        except queue.Full:
            try:
                buffer.get_nowait()
            except queue.Empty:
                pass
            try:
                buffer.put_nowait(message)
            except queue.Full:
                pass

    def update_param(self, current_state: FSMStateName = None):
        """更新机器人接口"""
        if current_state is not None:
            if self.current_state != current_state:
                self.last_state = self.current_state
                self.current_state = current_state

    def load_config(self):
        """从配置文件加载关键参数"""
        config_path = self.config_path
        if not os.path.exists(config_path):
            self.node.get_logger().error(
                f"Joint limits config file not found: {config_path}")
            raise FileNotFoundError(
                f"Joint limits config file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.node.get_logger().error(
                f"Failed to load joint limits config from {config_path}: {e}")
            raise RuntimeError(
                f"Failed to load joint limits config from {config_path}: {e}")
        # 读取运行模式
        self.sim = config.get('sim')
        self.debug = config.get('debug')
        # 机器人接口配置
        interface_config = config.get('robot_interface')
        self.motor_command_period_s = float(config.get('dt', 0.01))
        self.high_level_publish_period_s = 0.02
        if self.motor_command_period_s <= 0.0:
            raise ValueError('motor_command_period_s must be positive')
        if self.high_level_publish_period_s <= 0.0:
            raise ValueError('high_level_publish_period_s must be positive')

        # 优先使用节点参数中的 comm_mode 覆盖 YAML 中的配置
        param = self.node.get_parameter('comm_mode')
        comm_mode_from_param = str(param.value)
        self.comm_mode = comm_mode_from_param

        # 启动时打印当前使用的通信方式，便于确认
        if self.node is not None:
            source = "ROS 参数 (-p comm_mode)"

        # 是否限位
        self.clip_actions = interface_config.get('clip_actions')
        # 加载关节限位值
        self._load_joint_limits(interface_config)
        # 加载控制状态
        self._load_control_status(interface_config)
        # 零位
        self.zero_pos = np.array(interface_config.get('zero_pos'))
        # 电流转换比例
        self.ct_scale_sim = np.array(interface_config.get('ct_scale_sim'))
        self.ct_scale_real = np.array(interface_config.get('ct_scale_real'))
        if self.sim:
            self.ct_scale = self.ct_scale_sim
        else:
            self.ct_scale = self.ct_scale_real

        # IMU 数据偏移（度 → 弧度）
        self.imu_ypr_offset = (
            np.array(interface_config.get('imu_ypr_offset')) * np.pi / 180.0
        )
        # 禁用电机
        self.disable_joints_ = interface_config.get('disable_joints', False)
        # 脚踝Kp,Kd
        self.ankle_kp_p = np.array(interface_config.get('ankle_kp_p'))
        self.ankle_kd_p = np.array(interface_config.get('ankle_kd_p'))
        # 实时遥操作插值延迟（默认 0.012s，与源项目一致）
        try:
            self._gmr_interp_delay_sec = max(
                0.0, float(interface_config.get('gmr_interp_delay_sec', self._gmr_interp_delay_sec))
            )
        except (TypeError, ValueError):
            pass
        shared_cfg = interface_config.get('gmr_shared_receiver') or {}
        self._gmr_shared_enabled = bool(shared_cfg.get('enabled', False))
        self._gmr_shared_cfg = shared_cfg
        # 获取控制器类型
        self.control_tool = config.get('control_tool')
        # 状态速度配置
        self._load_state_speed_limits(config.get('joystick'))
        # 过渡时间配置
        self._load_trans_time(interface_config)
    
    def rewrite_config(self, config: Dict[str, Any]):
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key == 'sim':
                    if self.sim:
                        self.ct_scale = self.ct_scale_sim
                    else:
                        self.ct_scale = self.ct_scale_real
                    xlog.info(f"Rewrite ct_scale: {self.ct_scale}")
            else:
                xlog.warning(f"Key {key} not found in robot interface")


    def _load_trans_time(self, config: Dict[str, Any]):
        self.trans_time = config['trans_time']

    def _get_current_transition_time(self) -> float:
        """获取上一状态到当前状态的过渡时间"""
        current_state_name = self.current_state.name
        last_state_name = self.last_state.name
        state_trans_time = self.trans_time['state_trans_time']

        if current_state_name in state_trans_time:
            current_state_config = state_trans_time[current_state_name]
            last_state_trans_time = current_state_config.get('last_state', {})
            if last_state_name in last_state_trans_time:
                return last_state_trans_time[last_state_name]
            if 'default_trans_time' in current_state_config:
                return current_state_config['default_trans_time']

        return self.trans_time['default_time']

    def _load_state_speed_limits(self, config: Dict[str, Any]):
        self.x_command_offset = config.get("x_command_offset")
        self.y_command_offset = config.get("y_command_offset")
        self.yaw_command_offset = config.get("yaw_command_offset")
        self.max_x_plus_speed = config.get("max_x_plus_speed")
        self.max_x_minus_speed = config.get("max_x_minus_speed")
        self.max_y_speed = config.get("max_y_speed")
        self.max_yaw_speed = config.get("max_yaw_speed")
        state_limits = config.get('state_speed_limits')
        self.state_speed_limits = {}
        for state_name, limits in state_limits.items():
            self.state_speed_limits[state_name] = {
                "max_x_plus":
                limits.get("max_x_plus", self.max_x_plus_speed),
                "max_x_minus":
                limits.get("max_x_minus", self.max_x_minus_speed),
                "max_y":
                limits.get("max_y", self.max_y_speed),
                "max_yaw":
                limits.get("max_yaw", self.max_yaw_speed),
                "x_command_offset":
                limits.get("x_command_offset", self.x_command_offset),
                "y_command_offset":
                limits.get("y_command_offset", self.y_command_offset),
                "yaw_command_offset":
                limits.get("yaw_command_offset", self.yaw_command_offset)
            }

    def safe_get_states(self, state_names):
        """安全地获取状态枚举，跳过无效的状态名称"""
        valid_states = []
        for name in state_names:
            try:
                valid_states.append(FSMStateName[name])
            except (KeyError, AttributeError):
                pass
        return valid_states

    def _load_control_status(self, config: Dict[str, Any]):
        # 字符串命令到枚举值的映射
        all_states = list(FSMStateName.__members__.values())
        self.head_control_status = self.safe_get_states(
            config.get('head_control_status', []))
        self.waist_control_status = self.safe_get_states(
            config.get('waist_control_status', []))
        self.legs_control_status = self.safe_get_states(
            config.get('legs_control_status', []))
        self.head_control_status = self.safe_get_states(
            config.get('head_control_status', []))
        if self.legs_control_status == []:
            # 如果未配置腿部控制状态，则默认添加所有状态
            self.legs_control_status = all_states
        self.arms_control_status = self.safe_get_states(
            config.get('arms_control_status', []))
        self.left_arm_only_status = self.safe_get_states(
            config.get('left_arm_only_status', []))
        self.right_arm_only_status = self.safe_get_states(
            config.get('right_arm_only_status', []))
        #配置需要双臂动力学插值的状态列表
        self.dynamic_inter_status = self.safe_get_states(
            config.get('dynamic_inter_status', []))

        # 为每一个状态创建控制的电机id映射，如stop=[0,1,2...,28], MLP=[0,1,...,11]
        self.state_motor_id_map = {}
        for state_enum in all_states:
            motor_ids = []
            # 如果状态在腿部控制状态中，添加腿部电机ID
            if state_enum in self.legs_control_status:
                motor_ids.extend(list(range(self.id_map.leg_motor_nums)))

            # 如果状态在腰部控制状态中，添加腰部电机ID
            if state_enum in self.waist_control_status:
                start_idx = self.id_map.leg_motor_nums
                end_idx = start_idx + self.id_map.waist_motor_nums
                motor_ids.extend(list(range(start_idx, end_idx)))

            # 如果状态在手臂控制状态中，添加手臂电机ID
            if state_enum in self.arms_control_status:
                if state_enum in self.left_arm_only_status:
                    start_idx = self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                    end_idx = start_idx + self.id_map.arm_motor_nums // 2
                elif state_enum in self.right_arm_only_status:
                    start_idx = self.id_map.leg_motor_nums + self.id_map.waist_motor_nums + self.id_map.arm_motor_nums // 2
                    end_idx = start_idx + self.id_map.arm_motor_nums // 2
                else:
                    start_idx = self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                    end_idx = start_idx + self.id_map.arm_motor_nums
                motor_ids.extend(list(range(start_idx, end_idx)))

            # 去重并排序
            motor_ids = sorted(list(set(motor_ids)))

            self.state_motor_id_map[state_enum] = motor_ids

        # 打印映射关系用于调试
        for state_enum, motor_ids in self.state_motor_id_map.items():
            pass

    def _load_joint_limits(self, config: Dict[str, Any]):
        """从配置文件加载关节限位值"""
        # 从配置中获取关节限位信息
        joint_limits_config = config.get('joint_limits', None)

        if joint_limits_config is None:
            error_msg = "No joint_limits section in config"
            self.node.get_logger().error(error_msg)
            raise ValueError(error_msg)
        else:
            # 从配置中加载限位值
            for joint_name, limits in joint_limits_config.items():
                if 'min' in limits and 'max' in limits:
                    self.joint_limits[joint_name] = {
                        'min': float(limits['min']),
                        'max': float(limits['max'])
                    }
                else:
                    error_msg = f"Invalid limits for joint {joint_name}"
                    self.node.get_logger().error(error_msg)
                    raise ValueError(error_msg)

            xlog.info(f"Loaded joint limits from {config}")
        # 预计算ID到限位的映射
        self.id_to_limits = {}
        for joint_name, limits in self.joint_limits.items():
            index = self.id_map.get_index_by_name(joint_name)
            if index >= 0:
                motor_id = self.id_map.get_id_by_index(index)
                self.id_to_limits[motor_id] = limits

        # 记录加载的限位值
        for joint_name, limits in self.joint_limits.items():
            self.node.get_logger().debug(
                f"Joint {joint_name}: [{limits['min']}, {limits['max']}]")

    def init(self, node: Node):
        """初始化接口"""
        self.node = node
        self.initialized = True
        # 加载配置文件
        self.load_config()

        if self.comm_mode == 'ros2':
            # 初始化 ROS 底层接口（bodyctrl_msgs：/leg/cmd_ctrl, /leg/status 等）
            self._init_ros_lowlevel_interfaces()
        elif self.comm_mode == 'ros2_bridge':
            # 初始化 ROS 底层接口（ros2_bridge_msgs：/robot_state, /leg/cmd 等）
            self._init_ros_lowlevel_new_interfaces()
        elif self.comm_mode == 'lpc':
            # 初始化LPC SDK接口（本地过程调用）
            self._init_sdk_lpc_interface()
        elif self.comm_mode == 'shm_rpc':
            # 初始化SHM RPC SDK接口（共享内存 RPC，使用 ICEORYX2）
            self._init_sdk_shm_rpc_interface()
        elif self.comm_mode == 'tcp_rpc':
            # 初始化TCP RPC SDK接口（TCP Socket RPC，使用 YLT）
            self._init_sdk_tcp_rpc_interface()
        else:
            raise ValueError(f"Invalid communication mode: {self.comm_mode}")
        self._init_gmr_shared_receiver()
        self._init_ros_highlevel_interfaces()

        # 初始化控制系统
        self._init_control_system()
        # 初始化TF缓冲区和监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self._start_command_thread()
        self._high_level_timer = self.node.create_timer(
            self.high_level_publish_period_s,
            self._publish_background_robot_cmd,
        )

        xlog.info("Robot interface initialized")

    def _init_gmr_shared_receiver(self):
        if not self._gmr_shared_enabled:
            return
        cfg = getattr(self, "_gmr_shared_cfg", {}) or {}
        cpu_ids = cfg.get("cpu_ids", cfg.get("cpu_id"))
        fifo_prio = cfg.get("fifo_prio")
        try:
            self._gmr_shared_receiver = GMRSharedReceiver(
                gmr_expected_len=self._gmr_expected_len,
                gmr_6kp_expected_len=self._gmr_6kp_expected_len,
                gmr_topic=str(cfg.get("gmr_topic", "/gmr_info")),
                gmr_6kp_topic=str(cfg.get("gmr_6kp_topic", "/gmr_6kp")),
                cpu_ids=cpu_ids,
                fifo_prio=int(fifo_prio) if fifo_prio is not None else None,
            )
            ready = self._gmr_shared_receiver.start(
                timeout_s=float(cfg.get("startup_timeout_s", 5.0))
            )
            if ready:
                xlog.info("GMR shared receiver subprocess started")
            else:
                xlog.warning("GMR shared receiver subprocess did not report ready before timeout")
                self._gmr_shared_receiver.stop()
                self._gmr_shared_receiver = None
                self._gmr_shared_enabled = False
        except Exception as exc:
            self._gmr_shared_receiver = None
            self._gmr_shared_enabled = False
            xlog.warning(f"GMR shared receiver disabled after startup failure: {exc}")

    def _init_sdk_lpc_interface(self):
        """初始化 LPC SDK 接口（本地过程调用）"""
        # 尝试动态导入 robot_control_lpc_python 模块
        robot_control_lpc_python = import_control_sdk_lpc_python_bridge()
        if robot_control_lpc_python is None:
            return

        # 检查 Python Bridge 是否可用（单例已初始化且 Python Bridge 已启用）
        if not robot_control_lpc_python.is_available():
            return

        # 保存模块引用，后续通过模块函数调用 SDK
        self.sdk_lpc_interface = robot_control_lpc_python

        # 启动 SDK 状态轮询循环
        self._start_sdk_polling_loop()

    def _init_sdk_shm_rpc_interface(self):
        """初始化 SHM RPC SDK 接口（共享内存 RPC，使用 ICEORYX2）"""
        # 尝试动态导入 robot_control_shm_rpc_python 模块
        robot_control_shm_rpc_python = import_control_sdk_shm_rpc_python_bridge(
        )
        if robot_control_shm_rpc_python is None:
            return

        # 连接到 RPC 服务器（进程内通信，使用共享内存）
        try:
            # 无参 connect() 使用进程内通信（SHM/ICEORYX2）
            if not robot_control_shm_rpc_python.connect():
                return
        except Exception as e:
            return

        # 验证连接状态
        if not robot_control_shm_rpc_python.is_connected():
            return

        # 保存模块引用，后续通过模块函数调用 SDK
        self.sdk_rpc_interface = robot_control_shm_rpc_python

        # 启动 SDK 状态轮询循环
        self._start_sdk_polling_loop()

    def _init_sdk_tcp_rpc_interface(self):
        """初始化 TCP RPC SDK 接口（TCP Socket RPC，使用 YLT）"""
        # 尝试动态导入 robot_control_tcp_rpc_python 模块
        robot_control_tcp_rpc_python = import_control_sdk_tcp_rpc_python_bridge(
        )
        if robot_control_tcp_rpc_python is None:
            return

        # 连接到 RPC 服务器（跨机器通信，使用 TCP）
        try:
            # 默认连接到本地 RPC 服务器（端口 50051）
            if not robot_control_tcp_rpc_python.connect("127.0.0.1", "50051"):
                return
        except Exception as e:
            return

        # 验证连接状态
        if not robot_control_tcp_rpc_python.is_connected():
            return

        # 保存模块引用，后续通过模块函数调用 SDK
        self.sdk_rpc_interface = robot_control_tcp_rpc_python

        # 启动 SDK 状态轮询循环
        self._start_sdk_polling_loop()

    def _init_ros_lowlevel_interfaces(self):
        """初始化ROS底层接口"""
        # 动态导入bodyctrl_msgs模块
        bodyctrl_msgs = import_bodyctrl_msgs()
        if bodyctrl_msgs is None:
            return

        MotorStatusMsg = bodyctrl_msgs['MotorStatusMsg']
        CmdMotorCtrl = bodyctrl_msgs['CmdMotorCtrl']
        Imu = bodyctrl_msgs['Imu']

        qos_profile = QoSProfile(
            # reliability=ReliabilityPolicy.RELIABLE,
            # history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # 发布者
        self.pub_leg_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/leg/cmd_ctrl', qos_profile)
        self.pub_arm_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/arm/cmd_ctrl', qos_profile)
        self.pub_waist_motor_cmd = self.node.create_publisher(
            CmdMotorCtrl, '/waist/cmd_ctrl', qos_profile)

        # 订阅者
        self.sub_leg_state = self.node.create_subscription(
            MotorStatusMsg, '/leg/status', self._leg_motor_status_callback,
            qos_profile)
        self.sub_arm_state = self.node.create_subscription(
            MotorStatusMsg, '/arm/status', self._arm_motor_status_callback,
            qos_profile)
        self.sub_waist_state = self.node.create_subscription(
            MotorStatusMsg, '/waist/status', self._waist_motor_status_callback,
            qos_profile)

        #（非电机相关）
        self.sub_imu_xsens = self.node.create_subscription(
            Imu, '/imu/status', self._imu_status_callback, qos_profile)

    def _init_ros_lowlevel_new_interfaces(self):
        """初始化 ROS 底层接口（ros2_bridge_msgs：订阅 /robot_state，发布 /leg/cmd、/arm/cmd、/waist/cmd）"""
        if not _ros2_bridge_msgs_available:
            return
        _imported_modules['ros2_bridge_msgs'] = True

        qos_profile = QoSProfile(depth=10)

        # 发布者：ros2_bridge 使用 /leg/cmd, /arm/cmd, /waist/cmd
        self.pub_leg_motor_cmd = self.node.create_publisher(
            LegCtrl, '/leg/cmd', qos_profile)
        self.pub_arm_motor_cmd = self.node.create_publisher(
            ArmCtrl, '/arm/cmd', qos_profile)
        self.pub_waist_motor_cmd = self.node.create_publisher(
            WaistCtrl, '/waist/cmd', qos_profile)
        self.pub_head_motor_cmd = self.node.create_publisher(
            HeadCtrl, '/head/cmd', qos_profile)

        # 订阅者：单一话题 /robot_state 汇总腿/臂/腰/IMU
        self.sub_robot_state = self.node.create_subscription(
            RobotState, '/robot_state', self._robot_state_callback_ros2_bridge,
            qos_profile)

        # 无独立 leg/arm/waist/imu 订阅，由 _robot_state_callback_ros2_bridge 解包
        self.sub_leg_state = None
        self.sub_arm_state = None
        self.sub_waist_state = None
        self.sub_imu_xsens = None

        self.node.get_logger().info(
            "ROS2 bridge (ros2_bridge_msgs) low-level interface initialized")

    def _init_ros_highlevel_interfaces(self):
        """初始化高层抽象控制接口"""
        qos_profile = QoSProfile(
            # reliability=ReliabilityPolicy.RELIABLE,
            # history=HistoryPolicy.KEEP_LAST,
            depth=10)
        if not self._gmr_shared_enabled:
            self.sub_gmr_info = self.node.create_subscription(
                Float32MultiArray, '/gmr_info', self._gmr_callback, qos_profile)
        else:
            self.sub_gmr_info = None
        # roy 添加你的发布话题
        self.pub_walk_cmd = self.node.create_publisher(
            TwistStamped, '/hric/robot/cmd_vel_status', qos_profile)
        self.pub_walk_height_cmd = self.node.create_publisher(
            Float64, '/hric/robot/walk_height_status', qos_profile)
        self.pub_walk_speed = self.node.create_publisher(
            TwistStamped, '/hric/robot/walk_speed', qos_profile)
        self.pub_stand_cmd = self.node.create_publisher(
            TwistStamped, '/hric/robot/stand_cmd_status', qos_profile)
        self.pub_stand_status = self.node.create_publisher(
            TwistStamped, '/hric/robot/stand_status', qos_profile)
        # 只有在 ROS2 模式下才创建此发布者（bodyctrl 或 ros2_bridge 二选一）
        # if self.comm_mode == 'ros2' and _imported_modules['bodyctrl_msgs'] is not None:
        #     MotorStatusMsg = _imported_modules['bodyctrl_msgs']['MotorStatusMsg']
        #     self.pub_legs_status_serial = self.node.create_publisher(
        #         MotorStatusMsg, '/leg/status_serial', qos_profile)
        # elif self.comm_mode == 'ros2_bridge' and _imported_modules.get('ros2_bridge_msgs'):
        #     self.pub_legs_status_serial = self.node.create_publisher(
        #         LegStatus, '/leg/status_serial', qos_profile)
        # else:
        #     self.pub_legs_status_serial = None
        self.pub_robot_state = self.node.create_publisher(
            DiagnosticStatus, '/hric/robot/rl_state', qos_profile)
        # roy 添加你的订阅话题
        self.sub_walk_cmd = self.node.create_subscription(
            TwistStamped, '/hric/robot/cmd_vel', self._walk_cmd_callback,
            qos_profile)
        self.sub_walk_height_cmd = self.node.create_subscription(
            Float64, '/hric/robot/walk_height_cmd',
            self._walk_height_cmd_callback, qos_profile)
        self.sub_stand_cmd = self.node.create_subscription(
            TwistStamped, '/hric/robot/stand_cmd', self._stand_cmd_callback,
            qos_profile)
        if has_navigate_msg:
            self.sub_footpoint_cmd = self.node.create_subscription(
                FootPoint, '/hric/robot/footpoint',
                self._footpoint_cmd_callback, qos_profile)
        self.sub_fsm_state_cmd = self.node.create_subscription(
            String, '/hric/robot/fsm_state_cmd', self._fsm_state_cmd_callback,
            qos_profile)
        
        self.sub_fsm_resume_cmd = self.node.create_subscription(
            String, '/hric/robot/fsm_resume_cmd', self._fsm_resume_cmd_callback,
            qos_profile)
        #TODO:添加机器人状态信息订阅：如电量、电压等信息

        if self.control_tool == "joystick":
            self.sub_joy_cmd = self.node.create_subscription(
                Joy, '/sbus_data', self._joy_callback, qos_profile)
        if self.control_tool == "xbox":
            self.sub_xbox_cmd = self.node.create_subscription(
                Joy, '/xbox_data', self._xbox_callback, qos_profile)

    def _init_control_system(self):
        """初始化控制系统"""
        # 手柄控制器
        if self.control_tool == "joystick":
            self.joystick_humanoid = JoystickHumanoid()
            self.joystick_humanoid.init()

        # 键盘控制器
        if self.control_tool == "keyboard":
            self.keyboard_controller = KeyboardController()
            self.keyboard_controller.init()
            # 如果使用键盘控制，启动键盘监听
            self.keyboard_controller.start()

        # Xbox控制器
        if self.control_tool == "xbox":
            self.xbox_controller = XBOXController()
            self.xbox_controller.init()

    def process_controller_data(self):
        # 处理控制器输入
        if self.control_tool == "joystick":
            # 处理手柄输入
            while not self.queue_joy_cmd.empty():
                try:
                    msg = self.queue_joy_cmd.get_nowait()
                    self.joystick_humanoid.joy_map_read(msg)
                    self.joystick_humanoid.joy_flag_update()
                    break
                except queue.Empty:
                    break
        if self.control_tool == "xbox":
            while not self.queue_xbox_cmd.empty():
                try:
                    msg = self.queue_xbox_cmd.get_nowait()
                    self.xbox_controller.xbox_map_read(msg)
                    self.xbox_controller.xbox_flag_update()
                    break
                except queue.Empty:
                    break

        if self.control_tool == "keyboard":
            self.keyboard_controller.update_flag()
            flag = self.keyboard_controller.get_keyboard_flag()
        elif self.control_tool == "joystick":
            flag = self.joystick_humanoid.get_joy_flag()
            # 打印手柄状态切换
            _fsm_cmd = getattr(flag, 'fsm_state_command', '')
            if _fsm_cmd and _fsm_cmd != getattr(self, '_last_joy_fsm_cmd', ''):
                self._last_joy_fsm_cmd = _fsm_cmd
                xlog.info(f"[joystick] -> {_fsm_cmd}")
        elif self.control_tool == "xbox":
            flag = self.xbox_controller.get_xbox_flag()
            # 打印xbox状态切换
            _fsm_cmd = getattr(flag, 'fsm_state_command', '')
            if _fsm_cmd and _fsm_cmd != getattr(self, '_last_xbox_fsm_cmd', ''):
                self._last_xbox_fsm_cmd = _fsm_cmd
                xlog.info(f"[xbox] -> {_fsm_cmd}")
        else:
            pass
        self.control_flag = flag

    # @timing_decorator
    def get_imu_data(self):
        """Update IMU once and retain the last sample when no data is ready."""
        try:
            msg = self.queue_imu_xsens.get_latest()
        except queue.Empty:
            self.sensor_stats['imu']['stale_cycles'] += 1
            return False
        if hasattr(msg, 'euler') and hasattr(msg, 'angular_velocity') and hasattr(msg, 'linear_acceleration'):
            self.robot_data_.imu_data_[0] = msg.euler.yaw
            self.robot_data_.imu_data_[1] = msg.euler.pitch
            self.robot_data_.imu_data_[2] = msg.euler.roll
            self.robot_data_.imu_data_[3] = msg.angular_velocity.x
            self.robot_data_.imu_data_[4] = msg.angular_velocity.y
            self.robot_data_.imu_data_[5] = msg.angular_velocity.z
            self.robot_data_.imu_data_[6] = msg.linear_acceleration.x
            self.robot_data_.imu_data_[7] = msg.linear_acceleration.y
            self.robot_data_.imu_data_[8] = msg.linear_acceleration.z
        else:
            self.robot_data_.imu_data_[0] = getattr(msg, 'yaw', 0.0)
            self.robot_data_.imu_data_[1] = getattr(msg, 'pitch', 0.0)
            self.robot_data_.imu_data_[2] = getattr(msg, 'roll', 0.0)
            av = getattr(msg, 'angular_velocity', None)
            self.robot_data_.imu_data_[3] = av.x if av is not None else getattr(msg, 'wx', 0.0)
            self.robot_data_.imu_data_[4] = av.y if av is not None else getattr(msg, 'wy', 0.0)
            self.robot_data_.imu_data_[5] = av.z if av is not None else getattr(msg, 'wz', 0.0)
            la = getattr(msg, 'linear_acceleration', None)
            self.robot_data_.imu_data_[6] = la.x if la is not None else getattr(msg, 'ax', 0.0)
            self.robot_data_.imu_data_[7] = la.y if la is not None else getattr(msg, 'ay', 0.0)
            self.robot_data_.imu_data_[8] = la.z if la is not None else getattr(msg, 'az', 0.0)
        self.sensor_stats['imu']['updates'] += 1
        return True

    def update_robot_state(self):
        """Apply each latest motor sample once without waiting for other groups."""
        motor_groups = (
            ('leg', self.queue_leg_motor_state, 'first_leg_data_received'),
            ('arm', self.queue_arm_motor_state, 'first_arm_data_received'),
            ('waist', self.queue_waist_motor_state, 'first_waist_data_received'),
        )
        for group_name, state_queue, received_attr in motor_groups:
            try:
                msg = state_queue.get_latest()
            except queue.Empty:
                self.sensor_stats[group_name]['stale_cycles'] += 1
                continue
            if self.debug and hasattr(msg, 'header'):
                current_time = self.node.get_clock().now().to_msg()
                msg_time = msg.header.stamp
                time_diff = (current_time.sec - msg_time.sec) * 1000000000 + (current_time.nanosec - msg_time.nanosec)
                _ = time_diff / 1000000.0
            for status in msg.status:
                self.motor_state_to_robot_state(status, getattr(self, received_attr))
            setattr(self, received_attr, True)
            self.sensor_stats[group_name]['updates'] += 1

        try:
            msg = self.queue_head_motor_state.get_latest()
        except queue.Empty:
            self.sensor_stats['head']['stale_cycles'] += 1
        else:
            for motor_status in msg.status:
                index = self.id_map.get_index_by_id(motor_status.name)
                robotdata_index = index + self.floating_base_dof
                self.robot_data_.q_a_[robotdata_index] = float(motor_status.pos)
            self.sensor_stats['head']['updates'] += 1

    def initial_motor_state_ready(self) -> bool:
        return bool(
            self.first_leg_data_received
            and self.first_arm_data_received
            and self.first_waist_data_received
        )

    def wait_for_initial_motor_state(self, timeout_s: float = 10.0) -> bool:
        if self.sim:
            return True

        deadline = time.perf_counter() + timeout_s
        while rclpy.ok():
            self.update_robot_data(0.0)
            if self.initial_motor_state_ready():
                return True
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.01)
        return self.initial_motor_state_ready()

    def get_walk_cmd(self):
        """处理运动控制命令"""
        try:
            #TODO:添加平滑处理
            msg: TwistStamped = self.queue_walk_cmd.get_nowait()
            xyyaw_speed_limits = self.get_state_xyyaw_speed_limits(self.current_state.name)
            self.robot_data_.set_walk_cmd(np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z]), xyyaw_speed_limits)
        except queue.Empty:
            pass  # 如果队列为空，保持上次命令不变

    def get_walk_height_cmd(self):
        """处理运动控制命令"""
        try:
            #TODO:添加平滑处理
            msg: Float64 = self.queue_walk_height_cmd.get_nowait()
            self.robot_data_.set_walk_height_cmd(msg.data)
        except queue.Empty:
            pass  # 如果队列为空，保持上次命令不变

    def get_stand_cmd(self):
        """Parse stand command from TwistStamped topic"""
        try:
            msg: TwistStamped = self.queue_stand_cmd.get_nowait()
            self.robot_data_.set_floating_base_cmd(np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z, msg.twist.linear.z]))
        except queue.Empty:
            pass  # no new command, keep last value

    def get_footpoint_cmd(self):
        """获取足点命令"""
        if not has_navigate_msg:
            return
        try:
            msg: FootPoint = self.queue_footpoint_cmd.get_nowait()
            self.robot_data_.footpoint_cmd_[0] = msg.footflag
            self.robot_data_.footpoint_cmd_[1] = msg.relative_x
            self.robot_data_.footpoint_cmd_[2] = msg.relative_y
            self.robot_data_.footpoint_cmd_[3] = msg.relative_yaw

            current_time = self.node.get_clock().now().to_msg()
            msg_time = msg.stamp
            time_diff = (current_time.sec - msg_time.sec) * 1000000000 + (
                current_time.nanosec - msg_time.nanosec)
            time_diff_ms = time_diff / 1000000.0
            # print(f"Time difference: {time_diff_ms} ms")
        except queue.Empty:
            pass  # 如果队列为空，保持上次命令不变

    def get_fsm_state_cmd(self):
        """获取FSM状态命令"""
        try:
            msg: String = self.queue_fsm_state_cmd.get_nowait()
            self.fsm_state_command = msg.data
        except queue.Empty:
            pass  # 如果队列为空，保持上次命令不变

    def get_fsm_resume_cmd(self):
       """获取FSM RESUME状态命令"""
       try:
           msg: String = self.queue_fsm_resume_cmd.get_nowait()
           self.fsm_resume_command = msg.data
       except queue.Empty:
           pass  # 如果队列为空，保持上次命令不变

    def motor_state_to_robot_state(self, status, received_flag: bool):
        index = self.id_map.get_index_by_id(status.name)
        if index >= 0:
            self.robot_data_.diagnostic_info['motor_status'][status.name] = status.error
            robotdata_index = index + self.floating_base_dof  # 偏移到完整关节数组中
            # 直接赋值
            self.robot_data_.q_a_[robotdata_index] = status.pos
            last_qdot_a = self.robot_data_.q_dot_a_[robotdata_index]
            # 计算速度差值
            speed_gap = status.speed - last_qdot_a
            if abs(speed_gap) < self.joint_vel_threshold:
                self.robot_data_.q_dot_a_[robotdata_index] = status.speed
            self.robot_data_.tau_a_[
                robotdata_index] = status.current * self.ct_scale[min(
                    index,
                    len(self.ct_scale) - 1)]
            self.robot_data_.temperature_a[
                robotdata_index - self.floating_base_dof] = status.temperature

            self.robot_data_.q_a_[robotdata_index] = (
                status.pos - self.zero_pos[index]
            ) * self.motor_dir[index] + self.zero_offset[index]
            self.robot_data_.q_dot_a_[robotdata_index] *= self.motor_dir[index]
            self.robot_data_.tau_a_[robotdata_index] *= self.motor_dir[index]

            if not received_flag or abs(
                    self.robot_data_.q_a_[robotdata_index] -
                    self.robot_data_.q_a_last[robotdata_index]
            ) > self.joint_pos_threshold:
                if received_flag:
                    self.node.get_logger().warn(
                        f"Joint {index} error detected")
                    self.robot_data_.q_a_[
                        robotdata_index] = self.robot_data_.q_a_last[
                            robotdata_index]
                    self.robot_data_.q_dot_a_[
                        robotdata_index] = self.robot_data_.qdot_a_last[
                            robotdata_index]
                    self.robot_data_.tau_a_[
                        robotdata_index] = self.robot_data_.tor_a_last[
                            robotdata_index]
                else:
                    # 首次接收数据，更新标志位
                    received_flag = True
            self.robot_data_.q_a_last[robotdata_index] = self.robot_data_.q_a_[
                robotdata_index]
            self.robot_data_.qdot_a_last[
                robotdata_index] = self.robot_data_.q_dot_a_[robotdata_index]
            self.robot_data_.tor_a_last[
                robotdata_index] = self.robot_data_.tau_a_[robotdata_index]

    def update_sensor_states(self):
        # 获取Imu数据
        self.get_imu_data()
        # 添加IMU偏移
        self.robot_data_.imu_data_[0:3] += self.imu_ypr_offset

    def get_pelvis_ankle_transform(self):
        """
        获取pelvis_link相对于ankle_roll_l_link的坐标变换
        将四元数转换为xyz顺序的欧拉角，并记录z轴高度
        """
        try:
            # 获取pelvis_link相对于ankle_roll_l_link的变换
            trans = self.tf_buffer.lookup_transform(
                'ankle_roll_l_link',  # 目标坐标系
                'pelvis',  # 源坐标系
                rclpy.time.Time(),  # 最新可用的变换
                timeout=rclpy.duration.Duration(seconds=0.0)  # 超时时间
            )

            # 提取平移部分（xyz位置）
            z_offset = 0.056
            self.robot_data_.q_a_[0] = trans.transform.translation.x
            self.robot_data_.q_a_[1] = trans.transform.translation.y
            self.robot_data_.q_a_[2] = trans.transform.translation.z + z_offset

            # 提取旋转部分（四元数）
            quat = [
                trans.transform.rotation.w,
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
            ]

            # 将四元数转换为xyz顺序的欧拉角
            euler = t3d.euler.quat2euler(quat, axes="rxyz")
            # 存储欧拉角
            self.robot_data_.q_a_[3] = euler[0]  # x轴旋转
            self.robot_data_.q_a_[4] = euler[1]  # y轴旋转
            self.robot_data_.q_a_[5] = euler[2]  # z轴旋转

            return True

        except tf2_ros.TransformException as ex:
            # print(f'Could not get transform: {ex}')
            return False

    def update_robot_data(self, time_passed: float):
        # 更新传感器数据
        self.update_sensor_states()
        # 更新电机状态数据
        self.update_robot_state()
        # 更新机器人控制命令
        self.update_robot_cmd()
        # 脚踝并联转串联
        if not self.sim:
            #真机
            self.ankle_parallel_to_serial()
        # 获取stand的状态反馈
        self.dynamic_model.compute_stand_status()
        # 更新robot_data 时间戳
        self.robot_data_.time_now_ = time_passed

    def backup_serial_cmd(self):
        self.robot_data_.q_d_s_ = self.robot_data_.q_d_.copy()
        self.robot_data_.joint_kp_s_ = self.robot_data_.joint_kp_p_.copy()
        self.robot_data_.joint_kd_s_ = self.robot_data_.joint_kd_p_.copy()

    def _check_and_clip_joint_limits_fast(
            self, cmd_name: int, position: float) -> tuple[bool, float]:
        """
        快速检查并修正关节位置限位（避免重复查询）
        """
        if not self.sim and self.clip_actions:
            limit = self.id_to_limits[cmd_name]
            clipped_pos = np.clip(position, limit["min"], limit["max"])
            return clipped_pos == position, clipped_pos
        else:
            return True, position

    # @timing_decorator
    def convert_to_motor_commands(self):
        """将机器人控制命令转换为电机控制命令"""
        convert_to_motor_commands(
            robot_data=self.robot_data_,
            floating_base_dof=self.floating_base_dof,
            zero_offset=self.zero_offset,
            motor_dir=self.motor_dir,
            zero_pos=self.zero_pos,
            disable_joints=self.disable_joints_,
        )

    def _get_latest_command_frame(self):
        with self._command_lock:
            return self._latest_command_frame

    def _start_command_thread(self):
        self._command_stop_event.clear()
        self._command_thread = threading.Thread(
            target=self._command_publish_loop,
            name='xmigcs_motor_command',
            daemon=True,
        )
        self._command_thread.start()

    def _command_publish_loop(self):
        next_deadline = time.perf_counter()
        while not self._command_stop_event.is_set() and rclpy.ok():
            self._publish_latest_command()
            next_deadline += self.motor_command_period_s
            time_until_deadline = next_deadline - time.perf_counter()
            if time_until_deadline > 0.0:
                self._command_stop_event.wait(time_until_deadline)
            else:
                next_deadline = time.perf_counter()

    def _publish_latest_command(self):
        with self._command_publish_lock:
            command_frame = self._get_latest_command_frame()
            if command_frame is None or command_frame['state'].name in ('TELEOP_YIBO', 'WHOLE_BODY_CONTROL', 'ALLMIMIC'):
                return
            self._publish_command_frame(command_frame)

    def _publish_background_robot_cmd(self):
        with self._command_publish_lock:
            if self.current_state.name not in ('TELEOP_YIBO', 'WHOLE_BODY_CONTROL', 'ALLMIMIC'):
                self.publish_robot_cmd()

    def _publish_command_frame(self, command_frame):
        if not self.initialized:
            return

        publish_start = time.perf_counter()
        if self.comm_mode in ('ros2', 'ros2_bridge'):
            self.publish_motor_commands(command_frame)
        elif self.comm_mode in ['lpc', 'shm_rpc', 'tcp_rpc']:
            self._send_motor_commands(command_frame)
        publish_duration = time.perf_counter() - publish_start
        self.command_stats['published'] += 1
        self.command_stats['last_publish_s'] = publish_duration
        self.command_stats['max_publish_s'] = max(
            self.command_stats['max_publish_s'], publish_duration)

    # @timing_decorator
    def _command_view(self, command_frame):
        if command_frame is None:
            return self.robot_data_, self.current_state
        return SimpleNamespace(
            q_d_=command_frame['q_d'],
            q_dot_d_=command_frame['q_dot_d'],
            tau_d_=command_frame['tau_d'],
            joint_kp_p_=command_frame['kp'],
            joint_kd_p_=command_frame['kd'],
        ), command_frame['state']

    def publish_motor_commands(self, command_frame=None):
        """发布电机控制命令，支持 bodyctrl_msgs 与 ros2_bridge_msgs 两种接口"""
        command_data, current_state = self._command_view(command_frame)

        if self.comm_mode == 'ros2_bridge' and _imported_modules.get(
                'ros2_bridge_msgs'):
            self._publish_motor_commands_ros2_bridge(command_frame)
            return

        if _imported_modules['bodyctrl_msgs'] is None:
            return
        CmdMotorCtrl = _imported_modules['bodyctrl_msgs']['CmdMotorCtrl']
        MotorCtrl = _imported_modules['bodyctrl_msgs']['MotorCtrl']
        # 发布腿部控制命令
        if current_state in self.legs_control_status:
            leg_msg = CmdMotorCtrl()
            leg_msg.header.stamp = self.node.get_clock().now().to_msg()
            for i in range(self.id_map.leg_motor_nums):
                cmd = MotorCtrl()
                cmd.name = self.id_map.get_id_by_index(i)
                cmd.kp = float(command_data.joint_kp_p_[i])
                cmd.kd = float(command_data.joint_kd_p_[i])
                cmd.pos = float(command_data.q_d_[i +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    command_data.q_dot_d_[i + self.floating_base_dof])
                cmd.tor = float(
                    command_data.tau_d_[i + self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                leg_msg.cmds.append(cmd)
            self.pub_leg_motor_cmd.publish(leg_msg)

        # 只在特定模式下控制腰部
        if current_state in self.waist_control_status:
            # 腰部控制命令
            waist_msg = CmdMotorCtrl()
            waist_msg.header.stamp = self.node.get_clock().now().to_msg()
            for i in range(self.id_map.waist_motor_nums):
                cmd = MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums
                cmd.name = self.id_map.get_id_by_index(
                    motor_idx)  # 12 -> 33(yaw)
                cmd.kp = float(command_data.joint_kp_p_[motor_idx])
                cmd.kd = float(command_data.joint_kd_p_[motor_idx])
                cmd.pos = float(command_data.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    command_data.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    command_data.tau_d_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                waist_msg.cmds.append(cmd)
            # print(f'waist_msg {waist_msg}')
            self.pub_waist_motor_cmd.publish(waist_msg)

        # 只在特定模式下控制手臂
        if current_state in self.arms_control_status:
            # 手臂控制命令
            arm_msg = CmdMotorCtrl()
            arm_msg.header.stamp = self.node.get_clock().now().to_msg()
            if current_state in self.left_arm_only_status:
                control_index = np.arange(0, 7)
            elif current_state in self.right_arm_only_status:
                control_index = np.arange(self.id_map.arm_motor_nums - 7,
                                          self.id_map.arm_motor_nums)
            else:
                control_index = np.arange(0, self.id_map.arm_motor_nums)
            for i in control_index:
                cmd = MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(command_data.joint_kp_p_[motor_idx])
                cmd.kd = float(command_data.joint_kd_p_[motor_idx])
                cmd.pos = float(command_data.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    command_data.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    command_data.tau_d_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                arm_msg.cmds.append(cmd)
            # print(f'arm_msg {arm_msg}')
            self.pub_arm_motor_cmd.publish(arm_msg)

    def _publish_motor_commands_ros2_bridge(self, command_frame=None):
        """ros2_bridge_msgs：发布 LegCtrl/ArmCtrl/WaistCtrl 到 /leg/cmd、/arm/cmd、/waist/cmd"""
        command_data, current_state = self._command_view(command_frame)
        if self.pub_leg_motor_cmd is None:
            return
        # 腿部
        if current_state in self.legs_control_status:
            leg_msg = LegCtrl()
            leg_msg.header.stamp = self.node.get_clock().now().to_msg()
            leg_msg.header.frame_id = "leg_ctrl"
            leg_msg.mode = 1
            leg_msg.label = 0
            leg_msg.ctrl = []
            for i in range(self.id_map.leg_motor_nums):
                cmd = BridgeMotorCtrl()
                cmd.name = self.id_map.get_id_by_index(i)
                cmd.kp = float(command_data.joint_kp_p_[i])
                cmd.kd = float(command_data.joint_kd_p_[i])
                cmd.pos = float(command_data.q_d_[i +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    command_data.q_dot_d_[i + self.floating_base_dof])
                cmd.tor = float(
                    command_data.tau_d_[i + self.floating_base_dof])
                cmd.cur = 0.0
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                leg_msg.ctrl.append(cmd)
            self.pub_leg_motor_cmd.publish(leg_msg)
        # 腰部
        if current_state in self.waist_control_status:
            waist_msg = WaistCtrl()
            waist_msg.header.stamp = self.node.get_clock().now().to_msg()
            waist_msg.header.frame_id = "waist_ctrl"
            waist_msg.mode = 1
            waist_msg.label = 0
            waist_msg.ctrl = []
            for i in range(self.id_map.waist_motor_nums):
                motor_idx = i + self.id_map.leg_motor_nums
                cmd = BridgeMotorCtrl()
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(command_data.joint_kp_p_[motor_idx])
                cmd.kd = float(command_data.joint_kd_p_[motor_idx])
                cmd.pos = float(command_data.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    command_data.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    command_data.tau_d_[motor_idx +
                                            self.floating_base_dof])
                cmd.cur = 0.0
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                waist_msg.ctrl.append(cmd)
            self.pub_waist_motor_cmd.publish(waist_msg)
        # 手臂
        if current_state in self.arms_control_status:
            arm_msg = ArmCtrl()
            arm_msg.header.stamp = self.node.get_clock().now().to_msg()
            arm_msg.header.frame_id = "arm_ctrl"
            arm_msg.mode = 1
            arm_msg.label = 0
            arm_msg.ctrl = []
            if current_state in self.left_arm_only_status:
                control_index = np.arange(0, 7)
            elif current_state in self.right_arm_only_status:
                control_index = np.arange(self.id_map.arm_motor_nums - 7,
                                          self.id_map.arm_motor_nums)
            else:
                control_index = np.arange(0, self.id_map.arm_motor_nums)
            for i in control_index:
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                cmd = BridgeMotorCtrl()
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(command_data.joint_kp_p_[motor_idx])
                cmd.kd = float(command_data.joint_kd_p_[motor_idx])
                cmd.pos = float(command_data.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    command_data.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    command_data.tau_d_[motor_idx +
                                            self.floating_base_dof])
                cmd.cur = 0.0
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                arm_msg.ctrl.append(cmd)
            self.pub_arm_motor_cmd.publish(arm_msg)

        # 头部
        if current_state in self.head_control_status:
            if self.pub_head_motor_cmd is None:
                return
            head_msg = HeadCtrl()
            head_msg.header.stamp = self.node.get_clock().now().to_msg()
            head_msg.header.frame_id = "head_ctrl"
            head_msg.mode = 1   # ControlMode::CTRL（力位混合）
            head_msg.label = 0
            head_msg.ctrl = []
            for i in range(self.id_map.head_motor_nums):
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums + self.id_map.arm_motor_nums
                cmd = BridgeMotorCtrl()
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(command_data.joint_kp_p_[motor_idx])
                cmd.kd = float(command_data.joint_kd_p_[motor_idx])
                cmd.pos = float(command_data.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    command_data.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    command_data.tau_d_[motor_idx +
                                            self.floating_base_dof])
                cmd.cur = 0.0
                head_msg.ctrl.append(cmd)
            self.pub_head_motor_cmd.publish(head_msg)

    # @timing_decorator
    def publish_robot_cmd(self):
        """发布机器人控制命令"""
        flag = self.control_flag
        # roy 在这里实现你的话题发布代码
        if flag.enable:
            timestamp = self.node.get_clock().now().to_msg()
            # 发布行走控制命令
            walk_msg = TwistStamped()
            walk_msg.header.stamp = timestamp
            walk_msg.twist.linear.x = float(self.robot_data_.walk_cmd_[0])
            walk_msg.twist.linear.y = float(self.robot_data_.walk_cmd_[1])
            walk_msg.twist.angular.z = float(self.robot_data_.walk_cmd_[2])
            self.pub_walk_cmd.publish(walk_msg)
            # 发布行走高度控制命令
            walk_height_msg = Float64(data=self.robot_data_.walk_height_cmd_)
            self.pub_walk_height_cmd.publish(walk_height_msg)
            # 发布腰部控制命令
            stand_msg = TwistStamped()
            stand_msg.header.stamp = timestamp
            stand_msg.twist.angular.x = float(
                self.robot_data_.floating_base_cmd_[0])
            stand_msg.twist.angular.y = float(
                self.robot_data_.floating_base_cmd_[1])
            stand_msg.twist.angular.z = float(
                self.robot_data_.floating_base_cmd_[2])
            stand_msg.twist.linear.z = float(
                self.robot_data_.floating_base_cmd_[3])
            self.pub_stand_cmd.publish(stand_msg)

        # 发布基座线速度
        timestamp = self.node.get_clock().now().to_msg()
        walk_speed_msg = TwistStamped()
        walk_speed_msg.header.stamp = timestamp
        walk_speed_msg.twist.linear.x = self.robot_data_.q_dot_a_[0]
        walk_speed_msg.twist.linear.y = self.robot_data_.q_dot_a_[1]
        walk_speed_msg.twist.linear.z = self.robot_data_.q_dot_a_[2]
        self.pub_walk_speed.publish(walk_speed_msg)

        # 发布站立状态
        stand_status_msg = TwistStamped()
        stand_status_msg.header.stamp = timestamp
        stand_status_msg.twist.angular.x = float(self.robot_data_.q_a_[3])
        stand_status_msg.twist.angular.y = float(self.robot_data_.q_a_[4])
        stand_status_msg.twist.angular.z = float(self.robot_data_.q_a_[5])
        stand_status_msg.twist.linear.x = float(self.robot_data_.q_a_[0])
        stand_status_msg.twist.linear.y = float(self.robot_data_.q_a_[1])
        stand_status_msg.twist.linear.z = float(self.robot_data_.q_a_[2])
        self.pub_stand_status.publish(stand_status_msg)

        # 发布机器人的其余状态
        rl_state_msg = DiagnosticStatus()
        rl_state_msg.name = "rl_state"
        # 添加多个键值对
        pairs = [("current_state", f"{self.current_state.name}"),
                 ("control_tool", "joystick" if flag.enable else "topic")]
        for key, value in pairs:
            kv = KeyValue()
            kv.key = key
            kv.value = value
            rl_state_msg.values.append(kv)
        self.pub_robot_state.publish(rl_state_msg)

    @staticmethod
    def _make_sdk_ctrl_cmd_data(sdk_module, frame_id: str, motor_ctrls):
        """将 MotorCtrl 列表封装为 CmdData（与 bridge 中 send_*_cmd(ControlMode.CTRL, CmdData) 一致）。"""
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1e9)
        ctrl_msg = sdk_module.CtrlMsg()
        ctrl_msg.header.stamp.sec = sec
        ctrl_msg.header.stamp.nanosec = nanosec
        ctrl_msg.header.frame_id = frame_id
        ctrl_msg.ctrl = motor_ctrls
        cmd_data = sdk_module.CmdData()
        cmd_data.ctrl_msg = ctrl_msg
        return cmd_data

    def _send_motor_commands(self, command_frame=None):
        """发送电机控制命令，根据 comm_mode 使用对应的 SDK 接口"""
        command_data, current_state = self._command_view(command_frame)

        # 根据 comm_mode 选择对应的 SDK 接口和模块
        if self.comm_mode == 'lpc' and hasattr(self, 'sdk_lpc_interface'):
            sdk_interface = self.sdk_lpc_interface
            sdk_module = _imported_modules['robot_control_lpc_python']
        elif self.comm_mode == 'shm_rpc' and hasattr(self,
                                                     'sdk_rpc_interface'):
            sdk_interface = self.sdk_rpc_interface
            sdk_module = _imported_modules['robot_control_shm_rpc_python']
        elif self.comm_mode == 'tcp_rpc' and hasattr(self,
                                                     'sdk_rpc_interface'):
            sdk_interface = self.sdk_rpc_interface
            sdk_module = _imported_modules['robot_control_tcp_rpc_python']
        else:
            return

        if sdk_module is None:
            return

        # 发布腿部控制命令（统一为 send_leg_cmd(ControlMode.CTRL, CmdData)）
        if self.legs_control_status == [] or current_state in self.legs_control_status:
            leg_cmds = []
            for i in range(self.id_map.leg_motor_nums):
                cmd = sdk_module.MotorCtrl()
                cmd.name = self.id_map.get_id_by_index(i)
                cmd.kp = float(command_data.joint_kp_p_[i])
                cmd.kd = float(command_data.joint_kd_p_[i])
                cmd.pos = float(command_data.q_d_[i +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    command_data.q_dot_d_[i + self.floating_base_dof])
                cmd.tor = float(
                    command_data.tau_d_[i + self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                leg_cmds.append(cmd)
            leg_cmd_data = self._make_sdk_ctrl_cmd_data(
                sdk_module, "leg_ctrl", leg_cmds)
            try:
                ok = sdk_interface.send_leg_cmd(
                    sdk_module.ControlMode.CTRL, leg_cmd_data)
                if not ok:
                    pass
            except Exception as e:
                pass

        # 只在特定模式下控制腰部
        if current_state in self.waist_control_status:
            # 腰部控制命令
            waist_cmds = []
            for i in range(self.id_map.waist_motor_nums):
                cmd = sdk_module.MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums
                cmd.name = self.id_map.get_id_by_index(
                    motor_idx)  # 12 -> 33(yaw)
                cmd.kp = float(command_data.joint_kp_p_[motor_idx])
                cmd.kd = float(command_data.joint_kd_p_[motor_idx])
                cmd.pos = float(command_data.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    command_data.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    command_data.tau_d_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                waist_cmds.append(cmd)
            waist_cmd_data = self._make_sdk_ctrl_cmd_data(
                sdk_module, "waist_ctrl", waist_cmds)
            try:
                ok = sdk_interface.send_waist_cmd(
                    sdk_module.ControlMode.CTRL, waist_cmd_data)
                if not ok:
                    pass
            except Exception as e:
                pass

        # 只在特定模式下控制手臂
        if current_state in self.arms_control_status:
            # 手臂控制命令
            arm_cmds = []
            if current_state in self.left_arm_only_status:
                control_index = np.arange(0, 7)
            elif current_state in self.right_arm_only_status:
                control_index = np.arange(self.id_map.arm_motor_nums - 7,
                                          self.id_map.arm_motor_nums)
            else:
                control_index = np.arange(0, self.id_map.arm_motor_nums)
            for i in control_index:
                cmd = sdk_module.MotorCtrl()
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(command_data.joint_kp_p_[motor_idx])
                cmd.kd = float(command_data.joint_kd_p_[motor_idx])
                cmd.pos = float(command_data.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    command_data.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    command_data.tau_d_[motor_idx +
                                            self.floating_base_dof])

                # 检查关节位置限位
                within_limit, cmd.pos = self._check_and_clip_joint_limits_fast(
                    cmd.name, cmd.pos)
                if not within_limit:
                    pass
                arm_cmds.append(cmd)
            arm_cmd_data = self._make_sdk_ctrl_cmd_data(
                sdk_module, "arm_ctrl", arm_cmds)
            try:
                ok = sdk_interface.send_arm_cmd(
                    sdk_module.ControlMode.CTRL, arm_cmd_data)
                if not ok:
                    pass
            except Exception as e:
                pass
        
        # 头部
        if current_state in self.head_control_status:
            head_cmds = []
            for i in range(self.id_map.head_motor_nums):
                motor_idx = i + self.id_map.leg_motor_nums + self.id_map.waist_motor_nums + self.id_map.arm_motor_nums
                cmd = sdk_module.MotorCtrl()
                cmd.name = self.id_map.get_id_by_index(motor_idx)
                cmd.kp = float(command_data.joint_kp_p_[motor_idx])
                cmd.kd = float(command_data.joint_kd_p_[motor_idx])
                cmd.pos = float(command_data.q_d_[motor_idx +
                                                      self.floating_base_dof])
                cmd.spd = float(
                    command_data.q_dot_d_[motor_idx +
                                              self.floating_base_dof])
                cmd.tor = float(
                    command_data.tau_d_[motor_idx +
                                            self.floating_base_dof])
                head_cmds.append(cmd)
            head_cmd_data = self._make_sdk_ctrl_cmd_data(
                sdk_module, "head_ctrl", head_cmds)
            try:
                ok = sdk_interface.send_head_cmd(
                    sdk_module.ControlMode.CTRL, head_cmd_data)
                if not ok:
                    pass
            except Exception as e:
                pass

    # @timing_decorator
    def _capture_command_frame(self):
        """Capture one consistent command frame for diagnostics and publishing."""
        frame = {
            'state': self.current_state,
            'q_d': self.robot_data_.q_d_.copy(),
            'q_dot_d': self.robot_data_.q_dot_d_.copy(),
            'tau_d': self.robot_data_.tau_d_.copy(),
            'kp': self.robot_data_.joint_kp_p_.copy(),
            'kd': self.robot_data_.joint_kd_p_.copy(),
            'created_at': time.perf_counter(),
        }
        with self._command_lock:
            self._latest_command_frame = frame
            self.command_stats['generated'] += 1
        return frame

    def send_motor_commands(self):
        """Build and publish the latest command frame."""
        if not self.initialized:
            return
        self.interpolate_action()
        self.backup_serial_cmd()
        if not self.sim:
            #真机
            self.ankle_serial_to_parallel()
        self.convert_to_motor_commands()
        # The frame's state determines who owns publishing. Swapping the frame
        # and sending I/U synchronously under the same lock prevents duplicates
        # and stale I/U frames from being sent when another state takes over.
        with self._command_publish_lock:
            frame = self._capture_command_frame()
            if self.current_state.name in ('TELEOP_YIBO', 'WHOLE_BODY_CONTROL', 'ALLMIMIC'):
                self._publish_command_frame(frame)
                self.publish_robot_cmd()

    # ROS回调函数
    def _leg_motor_status_callback(self, msg):
        """腿部电机状态回调"""
        try:
            self._put_latest(self.queue_leg_motor_state, msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_leg_motor_state.get_nowait()  # 移除旧数据
                self._put_latest(self.queue_leg_motor_state, msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _arm_motor_status_callback(self, msg):
        """手臂电机状态回调"""
        try:
            self._put_latest(self.queue_arm_motor_state, msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_arm_motor_state.get_nowait()  # 移除旧数据
                self._put_latest(self.queue_arm_motor_state, msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _waist_motor_status_callback(self, msg):
        """腰部电机状态回调"""
        try:
            self._put_latest(self.queue_waist_motor_state, msg)
        except queue.Full:
            # 队列满时移除旧数据，加入新数据
            try:
                self.queue_waist_motor_state.get_nowait()  # 移除旧数据
                self._put_latest(self.queue_waist_motor_state, msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _imu_status_callback(self, msg):
        """IMU状态回调"""
        try:
            self._put_latest(self.queue_imu_xsens, msg)
        except queue.Full:
            try:
                self.queue_imu_xsens.get_nowait()  # 移除旧数据
                self._put_latest(self.queue_imu_xsens, msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _robot_state_callback_ros2_bridge(self, msg):
        """ros2_bridge_msgs：将 /robot_state 解包为 leg/arm/waist/imu 队列（与 bodyctrl 下游一致）"""
        try:
            if hasattr(msg, 'leg') and msg.leg.status:
                leg_msg = type('LegStatusMsg', (), {
                    'header': msg.leg.header,
                    'status': msg.leg.status
                })()
                try:
                    self._put_latest(self.queue_leg_motor_state, leg_msg)
                except queue.Full:
                    try:
                        self.queue_leg_motor_state.get_nowait()
                        self._put_latest(self.queue_leg_motor_state, leg_msg)
                    except Exception:
                        pass
            if hasattr(msg, 'arm') and msg.arm.status:
                arm_msg = type('ArmStatusMsg', (), {
                    'header': msg.arm.header,
                    'status': msg.arm.status
                })()
                try:
                    self._put_latest(self.queue_arm_motor_state, arm_msg)
                except queue.Full:
                    try:
                        self.queue_arm_motor_state.get_nowait()
                        self._put_latest(self.queue_arm_motor_state, arm_msg)
                    except Exception:
                        pass
            if hasattr(msg, 'waist') and msg.waist.status:
                waist_msg = type('WaistStatusMsg', (), {
                    'header': msg.waist.header,
                    'status': msg.waist.status
                })()
                try:
                    self._put_latest(self.queue_waist_motor_state, waist_msg)
                except queue.Full:
                    try:
                        self.queue_waist_motor_state.get_nowait()
                        self._put_latest(self.queue_waist_motor_state, waist_msg)
                    except Exception:
                        pass
            if hasattr(msg, 'imu') and msg.imu is not None:
                try:
                    self._put_latest(self.queue_imu_xsens, msg.imu)
                except queue.Full:
                    try:
                        self.queue_imu_xsens.get_nowait()
                        self._put_latest(self.queue_imu_xsens, msg.imu)
                    except Exception:
                        pass
            if hasattr(msg, 'head') and msg.head.status:
                head_msg = type('HeadStatusMsg', (), {
                    'header': msg.head.header,
                    'status': msg.head.status
                })()
                try:
                    self._put_latest(self.queue_head_motor_state, head_msg)
                except queue.Full:
                    try:
                        self.queue_head_motor_state.get_nowait()
                        self._put_latest(self.queue_head_motor_state, head_msg)
                    except Exception:
                        pass
        except Exception as e:
            self.node.get_logger().warn(
                f"Error in _robot_state_callback_ros2_bridge: {e}")

    def _walk_cmd_callback(self, msg: TwistStamped):
        """步态控制回调"""
        try:
            self.queue_walk_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_walk_cmd.get_nowait()  # 移除旧数据
                self.queue_walk_cmd.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _stand_cmd_callback(self, msg: TwistStamped):
        """Stand posture command callback (TwistStamped)"""
        try:
            self.queue_stand_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_stand_cmd.get_nowait()
                self.queue_stand_cmd.put_nowait(msg)
            except:
                pass

    def _walk_height_cmd_callback(self, msg: Float64):
        """步态高度控制回调"""
        try:
            self.queue_walk_height_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_walk_height_cmd.get_nowait()  # 移除旧数据
                self.queue_walk_height_cmd.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _footpoint_cmd_callback(self, msg: FootPoint):
        """Footpoint command callback"""
        try:
            self.queue_footpoint_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_footpoint_cmd.get_nowait()
                self.queue_footpoint_cmd.put_nowait(msg)
            except:
                pass

    def _fsm_state_cmd_callback(self, msg: String):
        """FSM状态命令回调"""
        try:
            self.queue_fsm_state_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_fsm_state_cmd.get_nowait()  # 移除旧数据
                self.queue_fsm_state_cmd.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略
    
    def _fsm_resume_cmd_callback(self, msg: String):
        """FSM RESUME状态命令回调"""
        try:
            self.queue_fsm_resume_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_fsm_resume_cmd.get_nowait()  # 移除旧数据
                self.queue_fsm_resume_cmd.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _joy_callback(self, msg):
        """云卓手柄输入回调"""
        try:
            self.queue_joy_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_joy_cmd.get_nowait()  # 移除旧数据
                self.queue_joy_cmd.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

    def _xbox_callback(self, msg):
        """xbox手柄输入回调"""
        try:
            self.queue_xbox_cmd.put_nowait(msg)
        except queue.Full:
            try:
                self.queue_xbox_cmd.get_nowait()  # 移除旧数据
                self.queue_xbox_cmd.put_nowait(msg)  # 加入新数据
            except:
                pass  # 如果仍然无法加入，忽略

# sdk接口获取数据

    def _start_sdk_polling_loop(self):
        self.sdk_stop_event.clear()
        self.sdk_polling_thread = threading.Thread(
            target=self._sdk_robot_status_polling_loop, daemon=True)
        self.sdk_polling_thread.start()

    def _sdk_robot_status_polling_loop(self):
        """SDK 状态轮询循环，根据 comm_mode 使用对应的接口"""
        while not self.sdk_stop_event.is_set():
            time_start = time.perf_counter()
            ideal_time_interval = 1.0 / 400
            # 根据 comm_mode 选择对应的 SDK 接口
            if self.comm_mode == 'lpc' and hasattr(self, 'sdk_lpc_interface'):
                sdk_interface = self.sdk_lpc_interface
            elif self.comm_mode == 'shm_rpc' and hasattr(
                    self, 'sdk_rpc_interface'):
                sdk_interface = self.sdk_rpc_interface
            elif self.comm_mode == 'tcp_rpc' and hasattr(
                    self, 'sdk_rpc_interface'):
                sdk_interface = self.sdk_rpc_interface
            else:
                self.sdk_stop_event.wait(0.1)
                continue
            # 手臂状态轮询
            try:
                success, arm_status = sdk_interface.get_arm_status()
            except Exception as e:
                success = False
            if success:
                try:
                    self._put_latest(self.queue_arm_motor_state, arm_status)
                except queue.Full:
                    # 队列满时移除旧数据，加入新数据
                    try:
                        self.queue_arm_motor_state.get_nowait()  # 移除旧数据
                        self._put_latest(self.queue_arm_motor_state, arm_status)  # 加入新数据
                    except:
                        pass  # 如果仍然无法加入，忽略

            else:
                pass
            # 腿部状态轮询
            try:
                success, leg_status = sdk_interface.get_leg_status()
            except Exception as e:
                success = False
            if success:
                try:
                    self._put_latest(self.queue_leg_motor_state, leg_status)
                except queue.Full:
                    # 队列满时移除旧数据，加入新数据
                    try:
                        self.queue_leg_motor_state.get_nowait()  # 移除旧数据
                        self._put_latest(self.queue_leg_motor_state, leg_status)  # 加入新数据
                    except:
                        pass
            else:
                pass
            # 腰部状态轮询
            try:
                success, waist_status = sdk_interface.get_waist_status()
            except Exception as e:
                success = False
            if success:
                try:
                    self._put_latest(self.queue_waist_motor_state, waist_status)
                except queue.Full:
                    # 队列满时移除旧数据，加入新数据
                    try:
                        self.queue_waist_motor_state.get_nowait()  # 移除旧数据
                        self._put_latest(self.queue_waist_motor_state, waist_status)  # 加入新数据
                    except:
                        pass
            else:
                pass
            # 头部状态轮询
            try:
                success, head_status = sdk_interface.get_head_status()
            except Exception as e:
                success = False
            if success:
                try:
                    self._put_latest(self.queue_head_motor_state, head_status)
                except queue.Full:
                    try:
                        self.queue_head_motor_state.get_nowait()
                        self._put_latest(self.queue_head_motor_state, head_status)
                    except:
                        pass
            else:
                pass
            # Imu状态轮询
            try:
                success, imu_status = sdk_interface.get_imu_status()
            except Exception as e:
                success = False
            if success:
                try:
                    self._put_latest(self.queue_imu_xsens, imu_status)
                except queue.Full:
                    # 队列满时移除旧数据，加入新数据
                    try:
                        self.queue_imu_xsens.get_nowait()  # 移除旧数据
                        self._put_latest(self.queue_imu_xsens, imu_status)  # 加入新数据
                    except:
                        pass
            else:
                pass
            time_left = ideal_time_interval - (time.perf_counter() -
                                               time_start)
            if time_left < 0:
                pass
            elif time_left > 0.001:
                self.sdk_stop_event.wait(time_left)

    # @timing_decorator
    def ankle_parallel_to_serial(self):
        # 串并联转换：并转串 (类似C++版本中的处理)
        # 提取左右脚两个踝关节（并联关节）
        q_a_p = np.zeros(4)  # 并联关节角度（实际）
        qdot_a_p = np.zeros(4)  # 并联关节速度（实际）
        tor_a_p = np.zeros(4)  # 并联关节力矩（实际）
        q_a_s = np.zeros(4)  # 串联关节角度（实际）
        qdot_a_s = np.zeros(4)  # 串联关节速度（实际）
        tor_a_s = np.zeros(4)  # 串联关节力矩（实际）

        q_a_p[:2] = self.robot_data_.q_a_[
            self.left_ankle_indices]  # 左脚踝关节 (pitch, roll)
        q_a_p[2:] = self.robot_data_.q_a_[
            self.right_ankle_indices]  # 右脚踝关节 (pitch, roll)

        qdot_a_p[:2] = self.robot_data_.q_dot_a_[self.left_ankle_indices]
        qdot_a_p[2:] = self.robot_data_.q_dot_a_[self.right_ankle_indices]

        tor_a_p[:2] = self.robot_data_.tau_a_[self.left_ankle_indices]
        tor_a_p[2:] = self.robot_data_.tau_a_[self.right_ankle_indices]

        self.q_a_p = q_a_p.copy()
        self.qdot_a_p = qdot_a_p.copy()
        self.tor_a_p = tor_a_p.copy()
        if self.debug:
            pass

        # 计算并转串（正运动学）
        self.fun_s2p.set_p_est(q_a_p, qdot_a_p, tor_a_p)
        self.fun_s2p.calcFK()
        self.fun_s2p.calcIK()

        self.fun_s2p.get_s_state(q_a_s, qdot_a_s, tor_a_s)

        if self.debug:
            pass

        # 用串联关节值替换原来的并联关节值
        self.robot_data_.q_a_[self.left_ankle_indices] = q_a_s[:2]  # 左脚踝关节串联值
        self.robot_data_.q_a_[self.right_ankle_indices] = q_a_s[2:]  # 右脚踝关节串联值

        self.robot_data_.q_dot_a_[self.left_ankle_indices] = qdot_a_s[:2]
        self.robot_data_.q_dot_a_[self.right_ankle_indices] = qdot_a_s[2:]

        self.robot_data_.tau_a_[self.left_ankle_indices] = tor_a_s[:2]
        self.robot_data_.tau_a_[self.right_ankle_indices] = tor_a_s[2:]

    # @timing_decorator
    def ankle_serial_to_parallel(self):
        # 串转并：将串联关节命令转换为并联关节命令（类似C++版本）
        # 提取踝关节两关节的串联命令
        q_d_p = np.zeros(4)  # 并联关节角度（期望）
        qdot_d_p = np.zeros(4)  # 并联关节速度（期望）
        tor_d_p = np.zeros(4)  # 并联关节力矩（期望）
        q_d_s = np.zeros(4)  # 串联关节角度（期望）
        qdot_d_s = np.zeros(4)  # 串联关节速度（期望）
        tor_d_s = np.zeros(4)  # 串联关节力矩（期望）

        q_d_s[:2] = self.robot_data_.q_d_[self.left_ankle_indices]  # 左脚踝关节串联命令
        q_d_s[2:] = self.robot_data_.q_d_[
            self.right_ankle_indices]  # 右脚踝关节串联命令

        qdot_d_s[:2] = self.robot_data_.q_dot_d_[self.left_ankle_indices]
        qdot_d_s[2:] = self.robot_data_.q_dot_d_[self.right_ankle_indices]

        tor_d_s[:2] = self.robot_data_.tau_d_[self.left_ankle_indices]
        tor_d_s[2:] = self.robot_data_.tau_d_[self.right_ankle_indices]

        q_a_s = np.zeros(4)  # 串联关节角度（实际）
        qdot_a_s = np.zeros(4)  # 串联关节速度（实际）
        q_a_s[:2] = self.robot_data_.q_a_[self.left_ankle_indices]  # 左脚踝关节串联值
        q_a_s[2:] = self.robot_data_.q_a_[self.right_ankle_indices]  # 右脚踝关节串联值
        qdot_a_s[:2] = self.robot_data_.q_dot_a_[
            self.left_ankle_indices]  # 左脚踝关节串联速度
        qdot_a_s[2:] = self.robot_data_.q_dot_a_[
            self.right_ankle_indices]  # 右脚踝关节串联速度

        kp = np.zeros(4)  # 串联关节刚度
        kd = np.zeros(4)  # 串联关节阻尼
        kp[:2] = self.robot_data_.joint_kp_p_[self.left_ankle_indices -
                                              self.floating_base_dof]
        kp[2:] = self.robot_data_.joint_kp_p_[self.right_ankle_indices -
                                              self.floating_base_dof]
        kd[:2] = self.robot_data_.joint_kd_p_[self.left_ankle_indices -
                                              self.floating_base_dof]
        kd[2:] = self.robot_data_.joint_kd_p_[self.right_ankle_indices -
                                              self.floating_base_dof]

        tor_d_s = kp * (q_d_s - q_a_s) + kd * (qdot_d_s - qdot_a_s)

        if self.debug:
            pass

        # 串转并计算
        self.fun_s2p.set_s_des(q_d_s, qdot_d_s, tor_d_s)
        self.fun_s2p.calc_joint_pos_ref()
        self.fun_s2p.calc_joint_tor_des()
        self.fun_s2p.get_p_des(q_d_p, qdot_d_p, tor_d_p)

        q_d_p = (tor_d_p - self.ankle_kd_p *
                 (qdot_d_p - self.qdot_a_p)) / self.ankle_kp_p + self.q_a_p

        if self.debug:
            pass

        # 用并联关节命令覆盖原来的串联命令
        self.robot_data_.q_d_[self.left_ankle_indices] = q_d_p[:2]  # 左脚踝关节并联命令
        self.robot_data_.q_d_[self.right_ankle_indices] = q_d_p[
            2:]  # 右脚踝关节并联命令

        self.robot_data_.q_dot_d_[self.left_ankle_indices] = qdot_d_p[:2]
        self.robot_data_.q_dot_d_[self.right_ankle_indices] = qdot_d_p[2:]

        self.robot_data_.tau_d_[self.left_ankle_indices] = 0.0
        self.robot_data_.tau_d_[self.right_ankle_indices] = 0.0
        # 更新脚踝Kp,Kd
        self.robot_data_.joint_kp_p_[
            self.left_ankle_indices -
            self.floating_base_dof] = self.ankle_kp_p[:2]
        self.robot_data_.joint_kp_p_[
            self.right_ankle_indices -
            self.floating_base_dof] = self.ankle_kp_p[2:]
        self.robot_data_.joint_kd_p_[
            self.left_ankle_indices -
            self.floating_base_dof] = self.ankle_kd_p[:2]
        self.robot_data_.joint_kd_p_[
            self.right_ankle_indices -
            self.floating_base_dof] = self.ankle_kd_p[2:]

    def update_robot_cmd(self):
        """更新机器人控制命令"""
        flag = self.control_flag
        if flag.enable:
            # 使用getattr设置默认值，避免AttributeError
            x_command = getattr(flag, 'x_speed_command')
            y_command = getattr(flag, 'y_speed_command')
            yaw_command = getattr(flag, 'yaw_speed_command')
            xyyaw_speed_limits = self.get_state_xyyaw_speed_limits(self.current_state.name)
            self.robot_data_.set_walk_cmd(np.array([x_command, y_command, yaw_command]), xyyaw_speed_limits)

            # 获取walk height command
            if hasattr(flag, 'walk_height_command'):
                walk_height_command = getattr(flag, 'walk_height_command')
                self.robot_data_.set_walk_height_cmd(walk_height_command)

            # stand模式下的手柄姿态指令（roll/pitch/yaw/height）
            if all(
                    hasattr(flag, attr) for attr in [
                        'waist_roll_command', 'waist_pitch_command',
                        'waist_yaw_command', 'waist_height_command'
                    ]):
                waist_roll_command = getattr(flag, 'waist_roll_command')
                waist_pitch_command = getattr(flag, 'waist_pitch_command')
                waist_yaw_command = getattr(flag, 'waist_yaw_command')
                waist_height_command = getattr(flag, 'waist_height_command')
                self.robot_data_.set_floating_base_cmd(np.array([
                    waist_roll_command, waist_pitch_command, waist_yaw_command,
                    waist_height_command
                ]))
            # 导航模式下, 覆写footpoint_cmd_
            self.robot_data_.footpoint_cmd_[0] = False
        else:
            # roy 在这里把topic 收到的指令传递给robot_data中的命令变量
            self.get_walk_cmd()
            self.get_walk_height_cmd()
            self.get_stand_cmd()
            self.get_footpoint_cmd()
            #TODO: 通过话题获取状态切换指令
            self.get_fsm_state_cmd()
            #TODO: 通过话题获取resume指令
            self.get_fsm_resume_cmd()

            # 更新手柄值
            self.sync_flag_cmd_by_topic()

    def sync_flag_cmd_by_topic(self):
        flag = self.control_flag
        # 同步walk高度值,防止高度跳变
        if hasattr(flag, 'walk_height_command'):
            flag.walk_height_command = self.robot_data_.walk_height_cmd_
        # 同步其他指令
        # stand posture
        if all(
                hasattr(flag, attr) for attr in [
                    'waist_roll_command', 'waist_pitch_command',
                    'waist_yaw_command', 'waist_height_command'
                ]):
            flag.waist_roll_command = self.robot_data_.floating_base_cmd_[0]
            flag.waist_pitch_command = self.robot_data_.floating_base_cmd_[1]
            flag.waist_yaw_command = self.robot_data_.floating_base_cmd_[2]
            flag.waist_height_command = self.robot_data_.floating_base_cmd_[3]
        # 同步状态切换命令
        if flag.fsm_state_command != "gotoSTOP" and self.fsm_state_command != "":
            #仅在非停止状态下同步
            flag.fsm_state_command = self.fsm_state_command
            self.fsm_state_command = ""
            
        if flag.fsm_state_command != "gotoSTOP" and self.fsm_resume_command != "":
            #仅在非停止状态下同步
            if hasattr(flag, "fsm_resume_command"):
                flag.fsm_resume_command = self.fsm_resume_command
            self.fsm_resume_command = ""

    # ---- 实时遥操作参考流 (/gmr_info) ----
    def _gmr_callback(self, msg: Float32MultiArray):
        """ROS 回调：把 /gmr_info 帧塞进队列（满了丢最旧帧）。"""
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != self._gmr_expected_len:
            if not self._gmr_warned_invalid_len:
                logger.warning(
                    f"[gmr] payload length mismatch: expected {self._gmr_expected_len}, "
                    f"got {data.size}; ignoring (will not warn again)"
                )
                self._gmr_warned_invalid_len = True
            return
        item = (time.monotonic(), data)
        try:
            self.queue_realtime_gmr_state.put_nowait(item)
        except queue.Full:
            try:
                self.queue_realtime_gmr_state.get_nowait()
                self.queue_realtime_gmr_state.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _poll_gmr_shared_receiver(self) -> None:
        receiver = self._gmr_shared_receiver
        if receiver is None:
            return
        snap = receiver.read_latest()
        if snap.gmr_data is not None:
            self._gmr_shared_pending_gmr = (snap.gmr_recv_t, snap.gmr_data)
        if snap.gmr_6kp_data is not None:
            self._gmr_shared_pending_gmr_6kp = (
                snap.gmr_6kp_recv_t,
                snap.gmr_6kp_data,
            )

    def get_realtime_gmr_data(self):
        """每个控制周期取最新两帧插值，写回 robot_data_.ref_*。"""
        self._poll_gmr_shared_receiver()
        latest = None
        if self._gmr_shared_pending_gmr is not None:
            latest = self._gmr_shared_pending_gmr
            self._gmr_shared_pending_gmr = None
        else:
            while True:
                try:
                    latest = self.queue_realtime_gmr_state.get_nowait()
                except queue.Empty:
                    break
        now = time.monotonic()
        if latest is not None:
            recv_time, data = latest
            frame = self._decode_gmr_frame(data)
            if self._gmr_curr_frame is not None and self._gmr_curr_time is not None:
                self._gmr_prev_frame = self._gmr_curr_frame
                self._gmr_prev_time = self._gmr_curr_time
            self._gmr_curr_frame = frame
            self._gmr_curr_time = recv_time

        output_frame = None
        sample_time = None
        if (
            self._gmr_prev_frame is not None
            and self._gmr_curr_frame is not None
            and self._gmr_prev_time is not None
            and self._gmr_curr_time is not None
        ):
            dt = self._gmr_curr_time - self._gmr_prev_time
            if dt > 1e-6:
                target_time = now - self._gmr_interp_delay_sec
                alpha = (target_time - self._gmr_prev_time) / dt
                alpha = float(np.clip(alpha, 0.0, 1.0))
            else:
                alpha = 1.0
            output_frame = self._interpolate_gmr_frame(
                self._gmr_prev_frame, self._gmr_curr_frame, alpha
            )
            sample_time = self._gmr_prev_time + alpha * dt
        elif self._gmr_curr_frame is not None:
            output_frame = self._gmr_curr_frame
            sample_time = self._gmr_curr_time

        if output_frame is None:
            return
        self._publish_gmr_reference(output_frame)
        # Clamp with the interpolation itself: a cached frame must not acquire
        # a new timestamp on every 100 Hz outer tick (U uses this for its window).
        self.robot_data_.ref_sample_time = sample_time
        self.robot_data_.ref_received_time = self._gmr_curr_time

    def _decode_gmr_frame(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Decode 232 float32 payload into typed arrays (lab-order joints)."""
        nb = self._gmr_num_bodies
        na = self._gmr_num_actions
        off_pos = nb * 3
        off_quat = off_pos + nb * 4
        off_jp = off_quat + na
        off_jv = off_jp + na
        off_lin = off_jv + 3
        return {
            "body_pos_w": data[:off_pos].reshape(1, nb, 3).astype(np.float32, copy=False),
            "body_quat_w": data[off_pos:off_quat].reshape(1, nb, 4).astype(np.float32, copy=False),
            "joint_pos": data[off_quat:off_jp].reshape(1, na).astype(np.float32, copy=False),
            "joint_vel": data[off_jp:off_jv].reshape(1, na).astype(np.float32, copy=False),
            "root_lin_vel_w": data[off_jv:off_lin].reshape(1, 3).astype(np.float32, copy=False),
            "root_ang_vel_w": data[off_lin:].reshape(1, 3).astype(np.float32, copy=False),
        }

    @staticmethod
    def _interpolate_gmr_frame(
        prev_frame: Dict[str, np.ndarray],
        curr_frame: Dict[str, np.ndarray],
        alpha: float,
    ) -> Dict[str, np.ndarray]:
        inv_alpha = 1.0 - alpha
        body_pos = inv_alpha * prev_frame["body_pos_w"] + alpha * curr_frame["body_pos_w"]
        body_quat = inv_alpha * prev_frame["body_quat_w"] + alpha * curr_frame["body_quat_w"]
        quat_norm = np.linalg.norm(body_quat, axis=-1, keepdims=True)
        body_quat = body_quat / np.clip(quat_norm, 1e-6, None)
        joint_pos = inv_alpha * prev_frame["joint_pos"] + alpha * curr_frame["joint_pos"]
        joint_vel = inv_alpha * prev_frame["joint_vel"] + alpha * curr_frame["joint_vel"]
        root_lin_vel_w = inv_alpha * prev_frame["root_lin_vel_w"] + alpha * curr_frame["root_lin_vel_w"]
        root_ang_vel_w = inv_alpha * prev_frame["root_ang_vel_w"] + alpha * curr_frame["root_ang_vel_w"]
        return {
            "body_pos_w": body_pos.astype(np.float32, copy=False),
            "body_quat_w": body_quat.astype(np.float32, copy=False),
            "joint_pos": joint_pos.astype(np.float32, copy=False),
            "joint_vel": joint_vel.astype(np.float32, copy=False),
            "root_lin_vel_w": root_lin_vel_w.astype(np.float32, copy=False),
            "root_ang_vel_w": root_ang_vel_w.astype(np.float32, copy=False),
        }

    def _publish_gmr_reference(self, frame: Dict[str, np.ndarray]) -> None:
        """把一帧 GMR 参考写到 robot_data_，让 FSM 在本周期消费。"""
        nb = self._gmr_num_bodies
        lin_vel_w = np.zeros((1, nb, 3), dtype=np.float32)
        ang_vel_w = np.zeros((1, nb, 3), dtype=np.float32)
        lin_vel_w[:, 0, :] = frame["root_lin_vel_w"]
        ang_vel_w[:, 0, :] = frame["root_ang_vel_w"]
        self.robot_data_.ref_body_pos_w = np.array(frame["body_pos_w"], copy=True)
        self.robot_data_.ref_body_quat_w = np.array(frame["body_quat_w"], copy=True)
        self.robot_data_.ref_body_lin_vel_w = lin_vel_w
        self.robot_data_.ref_body_ang_vel_w = ang_vel_w
        self.robot_data_.ref_joint_pos = np.array(frame["joint_pos"], copy=True)
        self.robot_data_.ref_joint_vel = np.array(frame["joint_vel"], copy=True)

    def get_control_flag(self):
        return self.control_flag

    def destroy_node(self):
        """销毁节点"""
        if self._gmr_shared_receiver is not None:
            self._gmr_shared_receiver.stop()
            self._gmr_shared_receiver = None
        self._command_stop_event.set()
        if self._command_thread is not None and self._command_thread.is_alive():
            self._command_thread.join(timeout=1.0)
        self._command_thread = None
        if self._high_level_timer is not None:
            self._high_level_timer.cancel()
            self._high_level_timer = None
        # # 先停止键盘控制器（重要！）
        self.sdk_stop_event.set()
        if hasattr(self, 'sdk_polling_thread') and self.sdk_polling_thread.is_alive():
            self.sdk_polling_thread.join(timeout=1.0)
        if hasattr(self,
                   'keyboard_controller') and self.control_tool == "keyboard":
            self.keyboard_controller.stop()

    def interpolate_action(self):
        (self._cached_transition_start_time,
         self._cached_last_state_serial_qd,
         self._cached_last_state_serial_kp,
         self._cached_last_state_serial_kd) = interpolate_action(
            robot_data=self.robot_data_,
            current_state=self.current_state,
            last_state=self.last_state,
            state_motor_id_map=self.state_motor_id_map,
            floating_base_dof=self.floating_base_dof,
            transition_time=self.robot_data_.trans_start_time,
            get_current_transition_time_fn=self._get_current_transition_time,
                    cached_transition_start_time=self._cached_transition_start_time,
            cached_last_state_serial_qd=self._cached_last_state_serial_qd,
            cached_last_state_serial_kp=self._cached_last_state_serial_kp,
            cached_last_state_serial_kd=self._cached_last_state_serial_kd,
            dynamic_model=self.dynamic_model,
            dynamic_inter_status=self.dynamic_inter_status,
        )
    def get_state_xyyaw_speed_limits(self, state_name: str):
        xyyaw_speed_limits = {
            "max_x_plus": self.max_x_plus_speed,
            "max_x_minus": self.max_x_minus_speed,
            "max_y": self.max_y_speed,
            "max_yaw": self.max_yaw_speed,
            "x_command_offset": self.x_command_offset,
            "y_command_offset": self.y_command_offset,
            "yaw_command_offset": self.yaw_command_offset
        }
        if state_name in self.state_speed_limits:
            state_limits = self.state_speed_limits[state_name]
            xyyaw_speed_limits["max_x_plus"] = state_limits.get("max_x_plus", self.max_x_plus_speed)
            xyyaw_speed_limits["max_x_minus"] = state_limits.get("max_x_minus", self.max_x_minus_speed)
            xyyaw_speed_limits["max_y"] = state_limits.get("max_y", self.max_y_speed)
            xyyaw_speed_limits["max_yaw"] = state_limits.get("max_yaw", self.max_yaw_speed)
            xyyaw_speed_limits["x_command_offset"] = state_limits.get("x_command_offset", self.x_command_offset)
            xyyaw_speed_limits["y_command_offset"] = state_limits.get("y_command_offset", self.y_command_offset)
            xyyaw_speed_limits["yaw_command_offset"] = state_limits.get("yaw_command_offset", self.yaw_command_offset)
        return xyyaw_speed_limits

def get_robot_interface(robot_data: RobotData, config_path: str) -> RobotInterface:
    """工厂函数，返回机器人接口实例"""
    return RobotInterfaceImpl(robot_data, config_path)
