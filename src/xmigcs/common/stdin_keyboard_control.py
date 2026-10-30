"""
Keyboard Control Module for SSH and Local Environments
Keyboard input handling for robot state management without external libraries
"""
import threading
import sys
import os
import select
import termios
import tty
import yaml
from .joystick import ControlFlag
import signal


class KeyboardFlag(ControlFlag):
    def __init__(self):
        super().__init__()
        self.x_speed_command: float = 0.0
        self.y_speed_command: float = 0.0
        self.yaw_speed_command: float = 0.0
        self.height_cmd: float = 0.89


class KeyboardController:
    """
    键盘控制器，纯Python实现，不依赖外部库

    按键映射:
        w/s     - 前进/后退（增量调节速度）
        a/d     - 左移/右移
        q/e     - 左转/右转
        r       - 重置速度命令为0
        z       - gotoZERO（进入站立零位）
        c       - gotoSTOP（紧急停止）
        m       - gotoSWR（进入SWR行走）
        n       - gotoAMPMLP（进入 AMPMLP）
        v       - gotoSUPERMIMIC（播放参考动作）
        i       - gotoTELEOP_YIBO（源部署770维GMR/NPZ策略，从AMPMLP或SWR进入）
        u       - gotoWHOLE_BODY_CONTROL（独立算法实验状态，从AMPMLP或SWR进入）
        x/Ctrl+C- 退出
        ←/→     - 调节高度
    """

    def __init__(self):
        self.keyboard_flag = KeyboardFlag()
        # 启动时不要触发任何状态转换
        self.keyboard_flag.fsm_state_command = ""
        self.data_mutex = threading.Lock()

        # 状态追踪
        self.current_height = 0.89
        self.target_height = 0.89
        self.height_step = 0.05

        # 速度限制
        self.initial_height = 0.89
        self.max_forward_speed = 1.0
        self.max_lateral_speed = 0.5
        self.max_rotation_speed = 0.5

        # 控制标志
        self.running = False
        self.input_thread = None
        self.original_terminal_settings = None

        self._load_config()

    def _load_config(self):
        """加载YAML配置文件"""
        try:
            current_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(current_dir, "config", "dex_config.yaml")
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            if not config:
                return
            keyboard_cfg = config.get("keyboard", {})
            self.initial_height = keyboard_cfg.get("initial_height", 0.89)
            self.height_step = keyboard_cfg.get("height_step", 0.05)
            self.max_forward_speed = keyboard_cfg.get("max_forward_speed", 1.0)
            self.max_lateral_speed = keyboard_cfg.get("max_lateral_speed", 0.5)
            self.max_rotation_speed = keyboard_cfg.get("max_rotation_speed", 0.5)
            self.current_height = self.initial_height
            self.target_height = self.initial_height
            self.keyboard_flag.height_cmd = self.current_height
        except Exception:
            pass

    def _print(self, msg: str):
        """在 raw 模式下安全打印"""
        sys.stdout.write(msg + "\r\n")
        sys.stdout.flush()

    def _print_speed(self):
        """打印当前速度"""
        f = self.keyboard_flag
        self._print(f"[keyboard] vx={f.x_speed_command:+.1f} vy={f.y_speed_command:+.1f} vyaw={f.yaw_speed_command:+.1f}")

    def start(self):
        """启动键盘监听线程"""
        if not sys.stdin.isatty():
            print("[KeyboardController] stdin 不是终端，键盘控制不可用，请在真实终端中运行")
            self.running = False
            return
        self.running = True
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        print("[KeyboardController] 已启动，按 z/m/n/v/i/u/c 控制状态（z=STANDZERO，m=SWR，n=AMPMLP，v=SUPERMIMIC，i=TELEOP_YIBO，u=WHOLE_BODY_CONTROL，c=STOP），w/s/a/d 控制速度")

    def stop(self):
        """停止键盘监听"""
        self.running = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        self._restore_terminal()

    def _restore_terminal(self):
        """恢复终端设置"""
        if self.original_terminal_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_terminal_settings)
                self.original_terminal_settings = None
            except Exception:
                pass

    def _input_loop(self):
        """主输入循环"""
        fd = sys.stdin.fileno()
        try:
            self.original_terminal_settings = termios.tcgetattr(fd)
        except termios.error as e:
            print(f"[KeyboardController] 无法获取终端设置: {e}")
            self.running = False
            return

        try:
            tty.setraw(fd)
            while self.running:
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if rlist:
                        data = os.read(fd, 1024)
                        if data:
                            key = data[0:1].decode("utf-8", errors="ignore")
                            if key:
                                self._process_key(key)
                except (OSError, IOError):
                    break
        except Exception as e:
            print(f"[KeyboardController] 输入循环异常: {e}")
        finally:
            self._restore_terminal()

    def _process_key(self, key):
        """处理按键输入"""
        key_map = {
            'w': self._on_forward,
            's': self._on_backward,
            'a': self._on_left,
            'd': self._on_right,
            'q': self._on_yaw_left,
            'e': self._on_yaw_right,
            'r': self._on_reset_speed,
            'z': self._on_goto_zero,
            'c': self._on_goto_stop,
            'm': self._on_goto_swr,
            'n': self._on_goto_ampmlp,
            'v': self._on_goto_supermimic,
            'i': self._on_goto_teleop_yibo,
            'u': self._on_goto_whole_body_control,
            'x': self._on_exit,
            '\x03': self._handle_ctrl_c,
            '\x1b': self._handle_arrow_key,
        }
        handler = key_map.get(key)
        if handler:
            handler()

    def _handle_arrow_key(self):
        """处理方向键序列"""
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                data = os.read(sys.stdin.fileno(), 2)
                if len(data) >= 2 and data[0:1] == b'[':
                    key3 = data[1:2].decode("utf-8", errors="ignore")
                    arrow_handlers = {
                        'D': self._increase_height,
                        'C': self._decrease_height,
                    }
                    handler = arrow_handlers.get(key3)
                    if handler:
                        with self.data_mutex:
                            handler()
        except (OSError, IOError):
            pass

    # ── 速度控制 ──

    def _reset_speed(self):
        """重置速度为0"""
        self.keyboard_flag.x_speed_command = 0.0
        self.keyboard_flag.y_speed_command = 0.0
        self.keyboard_flag.yaw_speed_command = 0.0

    def _on_forward(self):
        with self.data_mutex:
            self.keyboard_flag.x_speed_command = min(
                self.keyboard_flag.x_speed_command + 0.1, self.max_forward_speed)
            self._print_speed()

    def _on_backward(self):
        with self.data_mutex:
            self.keyboard_flag.x_speed_command = max(
                self.keyboard_flag.x_speed_command - 0.1, -self.max_forward_speed)
            self._print_speed()

    def _on_left(self):
        with self.data_mutex:
            self.keyboard_flag.y_speed_command = max(
                self.keyboard_flag.y_speed_command - 0.1, -self.max_lateral_speed)
            self._print_speed()

    def _on_right(self):
        with self.data_mutex:
            self.keyboard_flag.y_speed_command = min(
                self.keyboard_flag.y_speed_command + 0.1, self.max_lateral_speed)
            self._print_speed()

    def _on_yaw_left(self):
        with self.data_mutex:
            self.keyboard_flag.yaw_speed_command = max(
                self.keyboard_flag.yaw_speed_command - 0.1, -self.max_rotation_speed)
            self._print_speed()

    def _on_yaw_right(self):
        with self.data_mutex:
            self.keyboard_flag.yaw_speed_command = min(
                self.keyboard_flag.yaw_speed_command + 0.1, self.max_rotation_speed)
            self._print_speed()

    def _on_reset_speed(self):
        with self.data_mutex:
            self._reset_speed()
            self._print("[keyboard] speed reset")
            self._print_speed()

    # ── 状态切换 ──

    def _on_goto_zero(self):
        with self.data_mutex:
            self._reset_speed()
            self.keyboard_flag.fsm_state_command = "gotoZERO"
            self._print("[keyboard] -> gotoZERO")

    def _on_goto_stop(self):
        with self.data_mutex:
            self._reset_speed()
            self.keyboard_flag.fsm_state_command = "gotoSTOP"
            self._print("[keyboard] -> gotoSTOP")

    def _on_goto_swr(self):
        with self.data_mutex:
            self._reset_speed()
            self.keyboard_flag.fsm_state_command = "gotoSWR"
            self._print("[keyboard] -> gotoSWR")

    def _on_goto_ampmlp(self):
        with self.data_mutex:
            self._reset_speed()
            self.keyboard_flag.fsm_state_command = "gotoAMPMLP"
            self._print("[keyboard] -> gotoAMPMLP")

    def _on_goto_supermimic(self):
        with self.data_mutex:
            self._reset_speed()
            self.keyboard_flag.fsm_state_command = "gotoSUPERMIMIC"
            self._print("[keyboard] -> gotoSUPERMIMIC")

    # ── 高度控制 ──

    def _on_goto_teleop_yibo(self):
        with self.data_mutex:
            self._reset_speed()
            self.keyboard_flag.fsm_state_command = "gotoTELEOP_YIBO"
            self._print("[keyboard] -> gotoTELEOP_YIBO")

    def _on_goto_whole_body_control(self):
        with self.data_mutex:
            self._reset_speed()
            self.keyboard_flag.fsm_state_command = "gotoWHOLE_BODY_CONTROL"
            self._print("[keyboard] -> gotoWHOLE_BODY_CONTROL")

    def _increase_height(self):
        new_target = self.target_height + self.height_step
        if new_target <= 0.90:
            self.target_height = new_target
            self._print(f"[keyboard] height: {self.target_height:.2f}")

    def _decrease_height(self):
        new_target = self.target_height - self.height_step
        if new_target >= 0.65:
            self.target_height = new_target
            self._print(f"[keyboard] height: {self.target_height:.2f}")

    # ── 退出 ──

    def _on_exit(self):
        with self.data_mutex:
            self.running = False

    def _handle_ctrl_c(self):
        self.running = False
        self._restore_terminal()
        os.kill(os.getpid(), signal.SIGINT)

    # ── 公共接口 ──

    def update_flag(self):
        """更新控制标志（高度平滑调节）"""
        with self.data_mutex:
            if abs(self.current_height - self.target_height) > 0.0001:
                if self.current_height < self.target_height:
                    self.current_height += 0.0001
                else:
                    self.current_height -= 0.0001
            else:
                self.current_height = self.target_height
            self.keyboard_flag.height_cmd = self.current_height

    def get_keyboard_flag(self) -> KeyboardFlag:
        """获取当前键盘标志的副本"""
        with self.data_mutex:
            flag_copy = KeyboardFlag()
            flag_copy.__dict__.update(self.keyboard_flag.__dict__)
            return flag_copy

    def init(self) -> int:
        return 0
