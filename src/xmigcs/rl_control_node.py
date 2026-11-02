"""
RL Control Plugin (Python Version)
Main ROS2 node for humanoid robot RL control system
"""
import math
import os
import queue
import threading
import time
import sys

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# ROS messages
# from sensor_msgs.msg import Joy


from xmigcs.common.joystick import JoystickHumanoid, ControlFlag
# from xmigcs.common.xbox_control import XBOXController
# Local imports
from xmigcs.common.robot_data import RobotData
from xmigcs.FSM.robot_fsm import get_robot_fsm
from xmigcs.FSM.fsm_base import FSMStateName
from xmigcs.common.robot_interface import get_robot_interface
from xmigcs.utils.logging_utils import get_logger
from xmigcs.utils.xlog_utils import configure_xlog, xlog
# from xmigcs.common.stdin_keyboard_control  import KeyboardController
import functools


logger = get_logger(__name__)

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
        logger.info(f"[TIMING] {func.__name__} executed in {execution_time:.6f} seconds")
        return result
    return wrapper


class XMIGCSControlNode(Node):
    """xMIGCS控制节点Python版本"""

    def __init__(self, ):
        super().__init__('xmigcs_control_node')

        # 配置和参数
        self.floating_base_dof = 6

        # 加载配置
        self._load_config()

        # 初始化数据结构
        self._init_data_structures()

        # 机器人FSM
        self.robot_fsm = get_robot_fsm(
            self.robot_data,
            self.config,
        )

        # 机器人接口
        self.robot_interface = get_robot_interface(self.robot_data,
                                                   self.config_file)
        self.robot_interface.init(self)  # 传入node实例



        # 启动控制线程
        self._start_control_thread()

    def _load_config(self):
        """加载配置文件"""
        # 获取包路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(current_dir, 'config',
                                        'dex_config.yaml')

        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # 提取关键配置参数
        self.motor_num = self.config.get('motor_num')
        self.sim = self.config.get('sim')
        self.debug = self.config.get('debug')

        # 非策略状态的默认高层 tick；策略状态优先使用自身配置中的 dt。
        self._default_non_policy_tick_period_s = 0.01

        # 从配置中获取 robot_interface 配置，作为参数默认值来源
        robot_interface_cfg = self.config.get('robot_interface')
        default_comm_mode = robot_interface_cfg.get('comm_mode', 'ros2')

        # 声明 ROS2 参数，用于在运行时覆盖通讯方式
        # 例如：--ros-args -p comm_mode:=ros2_bridge / lpc / shm_rpc / tcp_rpc
        if not self.has_parameter('comm_mode'):
            self.declare_parameter('comm_mode', default_comm_mode)

        # 检查当前用户名，如果是ubuntu则抛出异常
        import getpass
        self.user_name = getpass.getuser().lower()
        if self.sim and self.user_name == 'ubuntu':
            raise RuntimeError("On ubuntu user, sim must be set to false")

    def _init_data_structures(self):
        """初始化数据结构"""
        # 机器人数据
        self.robot_data = RobotData(self.motor_num, self.motor_num + self.floating_base_dof)
        self.robot_data.config_file_ = getattr(self, 'config_file', '')


    def _start_control_thread(self):
        """启动控制线程"""
        self.control_running = True
        self.control_stop_event = threading.Event()
        self.control_stats = {
            'cycles': 0,
            'deadline_misses': 0,
            'max_cycle_s': 0.0,
            'last_cycle_s': 0.0,
            'cycle_durations_s': [],
        }
        self.control_thread = threading.Thread(target=self._rl_control_loop,
                                               daemon=True)
        self.control_thread.start()
        xlog.info("Control thread started")

    def _rl_control_loop(self):
        """主控制循环"""
        xlog.info("RL control loop starting...")
        try:
            self._rl_control_loop_impl()
        except Exception:
            logger.exception("RL control loop crashed")
            raise
        finally:
            xlog.info("RL control loop ended")

    def _rl_control_loop_impl(self):
        """主控制循环实现"""

        # 初始化时间戳
        time_passed = 0.0
        # 处理手柄数据
        self.robot_interface.process_controller_data()
        # 更新机器人数据
        self._update_robot_data(time_passed)
        if not self.robot_interface.wait_for_initial_motor_state():
            raise RuntimeError(
                "Initial motor state was not received; refusing to enter STOP"
            )
        # 切换到状态机初始状态
        self.robot_fsm.first_enter()
        last_control_freq_log_time = time.perf_counter()
        policy_tick_period_s = self._get_policy_tick_period_s()
        next_deadline = time.perf_counter() + policy_tick_period_s

        while self.control_running and rclpy.ok():
            loop_start = time.perf_counter()

            # 与 MosaicAgent.run 顺序对齐：policy.run(obs_prev) → env.step(action) → sleep。
            # env.step 内部是 "apply_action → physics_step → compute_observation(读 sensors)"，
            # 因此 obs 采样放在命令快照生成之后；FSM 本拍使用的是上一拍末尾算出的 obs。
            # 顺序：clock → joystick → FSM(policy, write q_d_) → cache command → sensors → sleep
            # I/U follow the source synchronous send path; other states use the sender thread.
            policy_tick_period_s = self._get_policy_tick_period_s()
            time_passed += policy_tick_period_s
            self.robot_interface.process_controller_data()
            # Source deployment order: sample/interpolate GMR before the FSM.
            self.robot_interface.get_realtime_gmr_data()
            # self._update_robot_data(time_passed)
            self.robot_fsm.run_fsm(self.robot_interface.get_control_flag())

            self.robot_interface.update_param(
                current_state=self.robot_fsm.get_current_state())
            self._send_control_commands()

            self._update_robot_data(time_passed)

            cycle_time = time.perf_counter() - loop_start
            self.control_stats['cycles'] += 1
            self.control_stats['last_cycle_s'] = cycle_time
            self.control_stats['max_cycle_s'] = max(
                self.control_stats['max_cycle_s'], cycle_time)
            durations = self.control_stats['cycle_durations_s']
            durations.append(cycle_time)
            if len(durations) > 1000:
                del durations[:-1000]

            next_period_s = self._get_policy_tick_period_s()
            if self.robot_fsm.get_current_state().name in ('TELEOP_YIBO', 'WHOLE_BODY_CONTROL', 'ALLMIMIC'):
                # Source uses loop_start + period, not accumulated deadlines.
                next_deadline = loop_start + next_period_s
            elif abs(next_period_s - policy_tick_period_s) > 1e-9:
                policy_tick_period_s = next_period_s
                next_deadline = time.perf_counter() + policy_tick_period_s
            else:
                next_deadline += policy_tick_period_s
            time_until_deadline = next_deadline - time.perf_counter()
            if time_until_deadline > 0:
                self.control_stop_event.wait(time_until_deadline)
            else:
                self.control_stats['deadline_misses'] += 1
                if not self.sim:
                    logger.warning(
                        "Control deadline missed by %.3f ms",
                        -time_until_deadline * 1000.0)
                next_deadline = time.perf_counter() + policy_tick_period_s

            control_freq = 1.0 / max(cycle_time, 1e-9)
            if time.perf_counter() - last_control_freq_log_time >= 5.0:
                xlog.info(f"current control freq: {control_freq:.2f} Hz")
                last_control_freq_log_time = time.perf_counter()

    def _get_policy_tick_period_s(self) -> float:
        state_name = self.robot_fsm.get_current_state()
        state_objects = getattr(self.robot_fsm, 'state_objects', {})
        state = state_objects.get(state_name)
        period = getattr(state, 'dt_', None)
        if period is None:
            return self._default_non_policy_tick_period_s
        period = float(period)
        if period <= 0.0:
            logger.warning('Invalid policy dt: %.6f', period)
            return self._default_non_policy_tick_period_s
        return period

    def _wall_clock_align_control_period(self, loop_start: float) -> None:
        """
        与 MosaicAgent.run 中的延迟处理等价：
            time_until_next_step = period - (now - loop_start)
            if > 0: time.sleep(time_until_next_step)
            else:   logger.warning("Time until next step is negative: ...")
        纯 sleep、无 busy-wait；每拍超时都打 warning（不做 rate limit），仿真模式下不打。
        """
        time_until_next_step = (
            self._get_policy_tick_period_s() - (time.perf_counter() - loop_start)
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        elif not self.sim:
            logger.warning(
                f"Time until next step is negative: {time_until_next_step}"
            )

    def _wait_for_start_signal(self):
        """等待启动信号"""
        start_file = "/tmp/rl_start_signal"
        self.get_logger().info("Waiting for start signal...")
        self.get_logger().info("Run: touch /tmp/rl_start_signal")

        # 删除可能存在的旧文件
        if os.path.exists(start_file):
            os.remove(start_file)

        # 等待启动文件出现
        while not os.path.exists(start_file) and rclpy.ok():
            time.sleep(0.5)

        self.get_logger().info("Start signal received, beginning RL control!")

    # @timing_decorator
    def _update_robot_data(self, time_passed: float):
        """更新机器人数据"""
        self.robot_interface.update_robot_data(time_passed)

    def _send_control_commands(self, ):
        """发布控制命令"""
        # I/U send synchronously; other states cache a frame for the sender thread.
        self.robot_interface.send_motor_commands()

    def destroy_node(self):
        """节点销毁"""
        self.control_running = False
        if hasattr(self, 'control_stop_event'):
            self.control_stop_event.set()
        self.robot_interface.destroy_node()
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)

        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

        super().destroy_node()


def main(args=None):
    """主函数"""
    configure_xlog(logger_id="xmigcs")
    rclpy.init(args=args)
    node = None
    try:
        node = XMIGCSControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals() and node is not None:
            node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
