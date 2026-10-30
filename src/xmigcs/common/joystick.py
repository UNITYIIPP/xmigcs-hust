"""
Joystick Control Module
Python equivalent of the C++ Joystick functionality for ROS Joy messages
"""
import os
import yaml
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from sensor_msgs.msg import Joy
import numpy as np
import time
from xmigcs.utils.logging_utils import get_logger
logger = get_logger(__name__)
from xmigcs.utils.xlog_utils import xlog


@dataclass
class ControlFlag:
    """手柄控制标志"""
    fsm_state_command: str = ""
    # 禁用、启用手柄控制标志
    enable: bool = True


@dataclass
class YUNZHUOMap:
    """云卓T12手柄按键映射 (对应ROS Joy消息)"""
    a: float = -1.0   # axes[8] #a,b,c,d手柄轴初始值为-1
    b: float = -1.0   # axes[9]
    c: float = -1.0   # axes[10]
    d: float = -1.0   # axes[11]
    e: float = 0.0   # axes[4]  e,f,g,h手柄轴初始值为0.0
    f: float = 0.0   # axes[7]
    g: float = 0.0   # axes[5]
    h: float = 0.0   # axes[6]
    x1: float = 0.0  # axes[3]
    x2: float = 0.0  # axes[0]
    y1: float = 0.0  # axes[2]
    y2: float = 0.0  # axes[1]


class YUNZHUOFlag(ControlFlag):  # 继承ControlFlag
    def __init__(self):
        super().__init__()  # 调用父类初始化
        self.fsm_state_command = ""
        # walk command
        self.x_speed_command: float = 0.0
        self.y_speed_command: float = 0.0
        self.yaw_speed_command: float = 0.0
        self.walk_height_command: float = 0.0
        # floating base command
        self.waist_roll_command: float = 0.0
        self.waist_pitch_command: float = 0.0
        self.waist_yaw_command: float = 0.0
        self.waist_height_command: float = 0.0
        # One-shot trigger used by kneeldownup to leave the hold phase.
        self.fsm_resume_command: str = ""
        # 按键有效按下次数（register_press_counter 注册的键；间隔内重复按下不计数）
        self.press_count_by_button: Dict[str, int] = {}


class JoystickHumanoid:
    """人形机器人手柄控制器 (ROS Joy版本)"""

    def __init__(self):

        # 初始化成员变量
        self.joy_map = YUNZHUOMap()
        self.joy_flag = YUNZHUOFlag()
        self.data_mutex = threading.Lock()

        # 配置参数
        self.initial_height = 0.0
        self.current_height = 0.0
        self.max_height = 0.0
        self.min_height = 0.0
        self.x_command_offset = 0.0
        self.y_command_offset = 0.0
        self.yaw_command_offset = 0.0
        self.max_x_plus_speed = 0.0
        self.max_x_minus_speed = 0.0
        self.max_y_speed = 0.0
        self.max_yaw_speed = 0.0
        # 高度平滑控制
        self.target_height = 0.0

        # 按键时间跟踪（按键级长按阈值）
        self.track_buttons = ['a', 'c']
        self.button_press_times = { button: 0.0 for button in self.track_buttons }
        self.button_last_states = { button: -1.0 for button in self.track_buttons }
        self.button_press_states = { button: None for button in self.track_buttons }
        self.long_press_thresholds = {
            button: 1.0 for button in self.track_buttons
        }  # 不同按键可设置不同长按阈值（秒）

        self._press_counter_interval: Dict[str, float] = {}
        self._press_counter_max_interval: Dict[str, float] = {}
        self._press_counter_counts: Dict[str, int] = {}
        self._press_counter_prev: Dict[str, float] = {}
        self._press_counter_last_accept: Dict[str, float] = {}

        # 加载配置文件
        self._load_config()

    def _load_config(self):
        """加载YAML配置文件"""
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        import getpass
        self.user_name = getpass.getuser().lower()
        if self.user_name == 'ubuntu':
            # config_path = "/home/ubuntu/dex_config.yaml"
            config_path = os.path.join(current_dir, "config", "dex_config.yaml")
        else:
            config_path = os.path.join(current_dir, "config", "dex_config.yaml")

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        if not config:
            return

        joystick_cfg = config.get("joystick", {})

        long_press_thresholds = joystick_cfg.get("long_press_thresholds", {})
        if isinstance(long_press_thresholds, dict):
            for button_name, threshold in long_press_thresholds.items():
                if button_name in self.track_buttons:
                    self.set_long_press_threshold(threshold, button_name=button_name)

        # 加载配置参数
        self.initial_height = joystick_cfg.get("initial_height")
        self.x_command_offset = joystick_cfg.get("x_command_offset")
        self.y_command_offset = joystick_cfg.get("y_command_offset")
        self.yaw_command_offset = joystick_cfg.get("yaw_command_offset")
        self.max_x_plus_speed = joystick_cfg.get("max_x_plus_speed")
        self.max_x_minus_speed = joystick_cfg.get("max_x_minus_speed")
        self.max_y_speed = joystick_cfg.get("max_y_speed")
        self.max_yaw_speed = joystick_cfg.get("max_yaw_speed")
        self.max_height = joystick_cfg.get("max_height")
        self.min_height = joystick_cfg.get("min_height")
        # 加载状态速度配置
        state_limits = joystick_cfg.get("state_speed_limits", {})
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


        self.current_height = self.initial_height
        self.target_height = self.initial_height
        self.joy_flag.waist_height_command = self.current_height
        self.joy_flag.walk_height_command = self.current_height

        pcfg = joystick_cfg.get("press_counters")
        if isinstance(pcfg, dict):
            for btn, spec in pcfg.items():
                if not isinstance(btn, str) or spec is None:
                    continue
                try:
                    min_iv, max_iv = self._parse_press_counter_config_entry(btn, spec)
                    self.register_press_counter(btn, min_iv, max_iv)
                except (TypeError, ValueError, KeyError) as e:
                    print(f"[Joystick_humanoid] Invalid press_counters entry {btn}={spec!r}: {e}")

    @staticmethod
    def _parse_press_counter_config_entry(btn: str, spec) -> Tuple[float, Optional[float]]:
        """标量为仅 min；dict 支持 min_interval/min 与可选 max_interval/max。"""
        if isinstance(spec, (int, float)):
            return float(spec), None
        if isinstance(spec, dict):
            min_iv = spec.get("min_interval", spec.get("min"))
            max_iv = spec.get("max_interval", spec.get("max"))
            if min_iv is None:
                raise ValueError("mapping form requires min_interval or min")
            min_iv = float(min_iv)
            max_iv = float(max_iv) if max_iv is not None else None
            if max_iv is not None and min_iv > max_iv:
                print(
                    f"[Joystick_humanoid] press_counters {btn}: min ({min_iv}) > max ({max_iv}), swapping"
                )
                min_iv, max_iv = max_iv, min_iv
            return min_iv, max_iv
        raise TypeError(f"expected number or mapping, got {type(spec).__name__}")

    def register_press_counter(
        self,
        button_name: str,
        min_interval_sec: float = 0.35,
        max_interval_sec: Optional[float] = None,
    ) -> None:
        """注册需要统计按下次数的按键（a/b/c/d）。

        在松开到按下的边沿上计数；若距上一次有效计数不足 ``min_interval_sec`` 秒则忽略本次。
        ``min_interval_sec <= 0`` 表示每次边沿都计数。

        若给定 ``max_interval_sec > 0``：在边沿上若距上次有效计数超过该值且当前计数非零，则先清零再判断是否累加。
        """
        if button_name not in YUNZHUOMap.__dataclass_fields__:
            raise ValueError(f"Unknown joystick button name: {button_name}")
        self._press_counter_interval[button_name] = max(0.0, float(min_interval_sec))
        if max_interval_sec is not None and max(0.0, float(max_interval_sec)) > 0.0:
            self._press_counter_max_interval[button_name] = max(0.0, float(max_interval_sec))
        else:
            self._press_counter_max_interval.pop(button_name, None)
        self._press_counter_counts.setdefault(button_name, 0)
        self._press_counter_prev.setdefault(button_name, -1.0)
        self._press_counter_last_accept.setdefault(button_name, -1e9)
        self.joy_flag.press_count_by_button.setdefault(button_name, 0)

    def unregister_press_counter(self, button_name: str) -> None:
        """取消注册，该键不再随 Joy 更新计数。"""
        self._press_counter_interval.pop(button_name, None)
        self._press_counter_max_interval.pop(button_name, None)

    def set_press_counter_interval(self, button_name: str, min_interval_sec: float) -> None:
        """修改已注册按键的有效计数最小时间间隔（秒）。"""
        if button_name not in self._press_counter_interval:
            raise KeyError(f"Button not registered for press counting: {button_name}")
        self._press_counter_interval[button_name] = max(0.0, float(min_interval_sec))

    def set_press_counter_max_interval(
        self, button_name: str, max_interval_sec: Optional[float]
    ) -> None:
        """修改已注册按键的计数序列最大间隔（秒）；``None`` 或 ``<=0`` 表示不限制。"""
        if button_name not in self._press_counter_interval:
            raise KeyError(f"Button not registered for press counting: {button_name}")
        if max_interval_sec is not None and max(0.0, float(max_interval_sec)) > 0.0:
            self._press_counter_max_interval[button_name] = max(0.0, float(max_interval_sec))
        else:
            self._press_counter_max_interval.pop(button_name, None)

    def get_press_count(self, button_name: str) -> int:
        """读取某键累计有效按下次数。"""
        return int(self._press_counter_counts.get(button_name, 0))

    def reset_press_count(self, button_name: Optional[str] = None) -> None:
        """计数清零；``button_name`` 为 None 时清零所有已注册键。"""
        if button_name is None:
            for k in list(self._press_counter_counts.keys()):
                self._press_counter_counts[k] = 0
                self.joy_flag.press_count_by_button[k] = 0
            return
        if button_name in self._press_counter_counts:
            self._press_counter_counts[button_name] = 0
        if button_name in self.joy_flag.press_count_by_button:
            self.joy_flag.press_count_by_button[button_name] = 0

    def _update_press_counters(self, current_time: float) -> None:
        """边沿检测 + 最大间隔超时清零 + 最小间隔过滤后累加计数。"""
        for name, min_interval in self._press_counter_interval.items():
            cur = float(getattr(self.joy_map, name, -1.0))
            prev = self._press_counter_prev.get(name, -1.0)
            if prev == -1.0 and cur == 1.0:
                last_acc = self._press_counter_last_accept.get(name, -1e9)
                dt = current_time - last_acc
                count = self._press_counter_counts.get(name, 0)
                max_interval = self._press_counter_max_interval.get(name)
                if (
                    max_interval is not None
                    and max_interval > 0.0
                    and count > 0
                    and dt > max_interval
                ):
                    self._press_counter_counts[name] = 0
                    count = 0
                if dt >= min_interval:
                    self._press_counter_counts[name] = count + 1
                    self._press_counter_last_accept[name] = current_time
            self._press_counter_prev[name] = cur
        for name in self._press_counter_counts:
            self.joy_flag.press_count_by_button[name] = self._press_counter_counts[name]

    def joy_map_read(self, msg: Joy):
        """处理ROS Joy消息，更新手柄映射"""
        with self.data_mutex:
            if len(msg.axes) >= 12:  # 确保有足够的轴数据
                yunzhuo_map = YUNZHUOMap(
                    a=msg.axes[8] if len(msg.axes) > 8 else -1.0,
                    b=msg.axes[9] if len(msg.axes) > 9 else -1.0,
                    c=msg.axes[10] if len(msg.axes) > 10 else -1.0,
                    d=msg.axes[11] if len(msg.axes) > 11 else -1.0,
                    e=msg.axes[4] if len(msg.axes) > 4 else 0.0,
                    f=msg.axes[7] if len(msg.axes) > 7 else 0.0,
                    g=msg.axes[5] if len(msg.axes) > 5 else 0.0,
                    h=msg.axes[6] if len(msg.axes) > 6 else 0.0,
                    x1=msg.axes[3] if len(msg.axes) > 3 else 0.0,
                    x2=msg.axes[0] if len(msg.axes) > 1 else 0.0,
                    y1=msg.axes[2] if len(msg.axes) > 2 else 0.0,
                    y2=msg.axes[1] if len(msg.axes) > 0 else 0.0)
                self.joy_map = yunzhuo_map

    def set_long_press_threshold(self, threshold: float, button_name: Optional[str] = None):
        """设置长按阈值（秒）"""
        threshold = max(0.025, float(threshold))  # 最小0.1秒
        if button_name is None:
            for tracked_button in self.track_buttons:
                self.long_press_thresholds[tracked_button] = threshold
            return
        self.long_press_thresholds[button_name] = threshold

    def _update_button_state(self, button_name: str, current_state: float, current_time: float):
        """更新单个按键状态，检测长按/短按
        
        Args:
            button_name: 按键名称
            current_state: 当前按键状态 (1.0按下, -1.0松开,)
            current_time: 当前时间戳（秒）
        """
        last_state = self.button_last_states.get(button_name, -1.0)
        
        # 检测按键从松开到按下的瞬间
        if last_state == -1.0 and current_state == 1.0:
            # 按键按下
            self.button_press_times[button_name] = current_time
            self.button_press_states[button_name] = None  # 重置按压状态
        
        # 按键持续按下中，检查是否达到长按阈值
        elif last_state == 1.0 and current_state == 1.0:
            # 按键松开
            press_duration = current_time - self.button_press_times.get(button_name, current_time)
            long_press_threshold = self.long_press_thresholds.get(button_name, 1.0)
            
            # 根据按压时间判断是否达到长按
            if press_duration >= long_press_threshold and self.button_press_states[button_name] == None:
                # 长按事件
                self.button_press_states[button_name] = 'long'
        elif last_state == 1.0 and current_state == -1.0:
            # 按键松开
            press_duration = current_time - self.button_press_times.get(button_name, current_time)
            if self.button_press_states[button_name] != 'long':
                # 短按事件
                self.button_press_states[button_name] = 'short'
        else:
            # 按键状态未变化
            self.button_press_states[button_name] = 'none'

        # 更新最后状态
        self.button_last_states[button_name] = current_state

    def _update_all_buttons(self):
        """更新所有按键状态（在 joy_flag_update 中调用）"""
        if not self.joy_flag.enable:
            return
        current_time = time.perf_counter()
        # 更新按键的状态
        buttons = self.track_buttons
        for button_name in buttons:
            # 获取按键当前状态
            current_state = getattr(self.joy_map, button_name, -1.0)
            # 更新按键状态
            self._update_button_state(button_name, current_state, current_time)

    def joy_flag_update(self):
        """根据手柄输入更新控制标志"""
        with self.data_mutex:
            # 更新手柄启动标志
            if self.joy_map.e == -1.0:
                # e上拨
                self.joy_flag.enable = False
            else:
                self.joy_flag.enable = True
        
            # 更新所有按键状态，检测长按/短按
            self._update_all_buttons()
            self._update_press_counters(time.perf_counter())
            # FSM状态切换命令
            button_pressed_nums = self.check_button_pressed_nums(
                self.joy_map)
            if self.joy_map.c == 1.0:
                self.joy_flag.fsm_state_command = "gotoSTOP"
            elif button_pressed_nums == 0:
                if self.joy_map.d == 1.0:
                    self.joy_flag.fsm_state_command = "gotoZERO"
                elif self.joy_map.a == 1.0 and self.joy_map.b == 1.0:
                    self.joy_flag.fsm_state_command = "gotoSUPERMIMIC"
                elif self.joy_map.b == 1.0:
                    self.joy_flag.fsm_state_command = "gotoSWR"
                elif self.joy_map.a == 1.0:
                    self.joy_flag.fsm_state_command = "gotoAMPMLP"

                self.get_x_y_yaw_speed_command()

    def get_joy_flag(self) -> ControlFlag:
        """获取当前手柄标志"""
        with self.data_mutex:
            return self.joy_flag

    def init(self) -> int:
        """初始化手柄控制器"""
        return 0

    def check_button_pressed_nums(self, joy_map: YUNZHUOMap) -> int:
        """检查按下的按钮数量"""
        count = 0
        if joy_map.e != 0.0:
            count += 1
        if joy_map.f != 0.0:
            count += 1
        if joy_map.g != 0.0:
            count += 1
        if joy_map.h != 0.0:
            count += 1
        return count

    def get_x_y_yaw_speed_command(self):
        """获取当前速度命令"""
        # 一次性获取状态配置
        state_name = self.joy_flag.fsm_state_command.replace("goto", "")
        state_limits = self.state_speed_limits.get(state_name, {})

        # 一次性获取所有速度限制
        max_x_plus = state_limits.get("max_x_plus", self.max_x_plus_speed)
        max_x_minus = state_limits.get("max_x_minus", self.max_x_minus_speed)
        max_y = state_limits.get("max_y", self.max_y_speed)
        max_yaw = state_limits.get("max_yaw", self.max_yaw_speed)
        x_command_offset = state_limits.get("x_command_offset",
                                            self.x_command_offset)
        y_command_offset = state_limits.get("y_command_offset",
                                            self.y_command_offset)
        yaw_command_offset = state_limits.get("yaw_command_offset",
                                              self.yaw_command_offset)
        # 速度命令计算
        self.joy_flag.y_speed_command = (self.joy_map.x1 * -max_y +
                                         y_command_offset)

        # X速度 (前进/后退)
        if self.joy_map.y1 >= 0:
            self.joy_flag.x_speed_command = (self.joy_map.y1 * max_x_plus +
                                             x_command_offset)  # 前进快一点
        else:
            self.joy_flag.x_speed_command = self.joy_map.y1 * max_x_minus  # 后退慢一点

        # 偏航速度
        self.joy_flag.yaw_speed_command = (self.joy_map.x2 * -max_yaw +
                                           yaw_command_offset)
        if os.environ.get("XMIGCS_JOYSTICK_SPEED_DEBUG", "0") == "1":
            print(
                f"State: {state_name}, "
                f"x_cmd: {self.joy_flag.x_speed_command:.2f}, "
                f"y_cmd: {self.joy_flag.y_speed_command:.2f}, "
                f"yaw_cmd: {self.joy_flag.yaw_speed_command:.2f}"
            )

    def get_walk_height_command(self):
        """获取当前高度命令"""
        current_height_command = self.joy_flag.walk_height_command
        deadzone_height = 0.5
        # 高度命令计算
        if self.joy_map.x2 >= deadzone_height:
            # x2 下拨
            self.joy_flag.walk_height_command += -self.joy_map.x2 * (
                self.joy_flag.walk_height_command - self.min_height)
        if self.joy_map.x2 <= -deadzone_height:
            # x2 上拨
            self.joy_flag.walk_height_command += -self.joy_map.x2 * (
                self.max_height - self.joy_flag.walk_height_command)

        # 1s中高度变化3cm, step= 0.03 / 100 hz = 0.0003
        step = 0.03 / 100
        self.joy_flag.walk_height_command = np.clip(
            self.joy_flag.walk_height_command, current_height_command - step,
            current_height_command + step)

    def get_rpyz_command(self):
        """使用两个遥杆生成站立策略的roll/pitch/yaw/height命令"""

        def _scale_centered(axis_val: float, neg_limit: float,
                            pos_limit: float) -> float:
            axis = float(np.clip(axis_val, -1.0, 1.0))
            return axis * (pos_limit if axis >= 0 else abs(neg_limit))

        # 右摇杆控制姿态，左摇杆控制偏航和高度（与行走时的轴方向保持一致）
        self.joy_flag.waist_roll_command = _scale_centered(
            self.joy_map.x1, -0.3, 0.3)
        self.joy_flag.waist_pitch_command = _scale_centered(
            self.joy_map.y1, -0.2, 0.5)
        self.joy_flag.waist_yaw_command = float(
            np.clip(-self.joy_map.x2, -1.0, 1.0) * 2.0)
        # 高度：摇杆回中保持初始高度，推杆在 [-1,1] 内映射到 [0.5, 0.95]
        height_deadzone = 0.1
        height_axis_raw = float(np.clip(-self.joy_map.y2, -1.0, 1.0))
        if abs(height_axis_raw) < height_deadzone:
            height_axis = 0.0
        else:
            height_axis = height_axis_raw
        init_h = self.initial_height if hasattr(self,
                                                "initial_height") else 0.98
        self.joy_flag.waist_height_command = float(
            np.interp(height_axis, [-1.0, 0.0, 1.0], [0.4, init_h, 0.98]))
