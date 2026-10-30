import json
import yaml
from typing import Dict, Tuple
from xmigcs.FSM.fsm_base import FSMStateName, FSMState, RobotFSM
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData
import importlib
import functools
import time
from xmigcs.common.dynamic_model import DynamicModel
from xmigcs.utils.xlog_utils import xlog
from xmigcs.utils.logging_utils import get_logger

_logger = get_logger(__name__)

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


class RobotFSMImpl(RobotFSM):
    """基于配置的状态机"""

    def __init__(self, robot_data: RobotData, config: Dict):
        super().__init__(robot_data)
        self.current_state: FSMStateName = None
        self.target_state: FSMStateName = None
        # 每个 (事件, 源状态) 对应一个或多个目标状态；trigger 时按顺序取首个通过条件检查的
        self.transitions: Dict[str, Dict[FSMStateName, Tuple[FSMStateName, ...]]] = {}
        self.state_objects: Dict[FSMStateName, FSMState] = {}
        self.last_event_name = ""
        # 只记录当前活跃的拒绝类型，相同拒绝不重复刷日志
        self._active_reject_key = None
        self._last_motor_error_log_time = {}
        # 安全保护配置
        safety_cfg = config.get('safety', {}) if config else {}
        self._fall_protection = safety_cfg.get('fall_protection', False)
        self._fall_gravity_threshold = safety_cfg.get('fall_gravity_threshold', -0.3)
        if config:
            self.load_config(config)
        self.first_run = False
        # 接触检测器
        self.contact_detector = DynamicModel(
            robot_data=robot_data)

    def _log_info(self, message: str, *args) -> None:
        rendered = message % args if args else message
        xlog.info(f"{rendered}")
        _logger.info(rendered)


    def _log_error(self, message: str, *args) -> None:
        # xlog Python 绑定没有 error，统一映射到 warning
        rendered = message % args if args else message
        xlog.warning(f"{rendered}")
        _logger.warning(rendered)

    def _log_transition_rejected(
        self,
        reason: str,
        event_name: str,
        current_state: str,
        target_state: str | None = None,
    ) -> None:
        key = (reason, event_name, current_state, target_state or "")
        if self._active_reject_key == key:
            return

        message = (
            "fsm_transition_rejected_start reason=%s event=%s current=%s"
            if target_state is None
            else "fsm_transition_rejected_start reason=%s event=%s current=%s target=%s"
        )
        values = (
            (reason, event_name, current_state)
            if target_state is None
            else (reason, event_name, current_state, target_state)
        )
        self._log_error(message, *values)
        self._active_reject_key = key
        
    def load_config(self, config: Dict):
        # 解析配置
        # 扩展枚举（配置文件会覆盖硬编码）
        for name, value in config.get('states').items():
            FSMStateName.extend(name, value)

        for event_name, trans_list in config['transitions'].items():
            self.transitions[event_name] = {}
            for trans in trans_list:
                try:
                    from_state = FSMStateName[trans['from']]
                except KeyError:
                    continue
                to_raw = trans['to']
                name_seq = to_raw if isinstance(to_raw, (list, tuple)) else (to_raw,)
                to_states_list = []
                for x in name_seq:
                    key = str(x)
                    try:
                        to_states_list.append(FSMStateName[key])
                    except KeyError:
                        self._log_error(
                            "fsm_transition_skip_unknown_to event=%s from=%s to=%s",
                            event_name,
                            trans['from'],
                            key,
                        )
                if not to_states_list:
                    continue
                to_states = tuple(to_states_list)
                bucket = self.transitions[event_name]
                if from_state in bucket:
                    bucket[from_state] = bucket[from_state] + to_states
                else:
                    bucket[from_state] = to_states

        # 自动添加全局转换
        self._add_global_transitions(config)

        for state_name in config['states']:
            # 构建模块名和类名
            module_name = f"xmigcs.policy.{state_name.lower()}.fsm_{state_name.lower()}"
            class_name = f"FSMState{state_name}"
            try:
                # 动态导入
                module = importlib.import_module(module_name)
                state_class = getattr(module, class_name)

                # 创建实例
                state_enum = FSMStateName[state_name]
                self.state_objects[state_enum] = state_class(self.robot_data_)
                self._log_info("fsm_state_load_success state=%s module=%s class=%s", state_name, module_name, class_name)
            except Exception as exc:
                # 状态模块导入或实例化失败时记录错误
                self._log_error(
                    "fsm_state_load_failed state=%s module=%s class=%s error=%s",
                    state_name,
                    module_name,
                    class_name,
                    exc,
                )
                raise

        self.target_state = FSMStateName[config['init_state']]

    def _add_global_transitions(self, config: Dict):
        """
        添加全局转换规则
        如果配置中定义了特殊事件（如'gotoSTOP'），
        则为所有状态添加相应的转换
        """
        all_states = [FSMStateName[name] for name in config['states'].keys()]

        # 定义需要全局处理的事件及其目标状态
        global_events = {
            'gotoSTOP': 'STOP',
            'gotoDAMPING': 'DAMPING',
            # 可以在这里添加更多，例如：
            # 'gotoPAUSE': 'PAUSE',
            # 'gotoRESET': 'IDLE',
        }

        for event_name, target_state_name in global_events.items():
            if event_name in self.transitions:
                target_state = FSMStateName[target_state_name]

                for state in all_states:
                    # 如果该状态还没有定义到这个事件的转换，则自动添加
                    if state not in self.transitions[event_name]:
                        self.transitions[event_name][state] = (target_state,)

    def first_enter(self):
        """进入初始状态"""
        if not self.first_run:
            self.current_state = self.target_state
            self.state_objects[self.current_state].on_enter()
            # 记录初始进入状态
            self._log_info(
                "fsm_initial_enter state=%s robot_time=%.6f",
                self.current_state.name,
                float(getattr(self.robot_data_, "time_now_", 0.0)),
            )
            self.first_run = True

    def trigger(self, flag: ControlFlag) -> bool:
        """
        触发事件转换
        
        Args:
            flag: 控制标志        
        Returns:
            是否成功转换
        """
        event_name = flag.fsm_state_command
        self.last_event_name = event_name
        if not event_name:
            return False
        if event_name not in self.transitions:
            # 事件未定义：开始记录一段拒绝区间
            self._log_transition_rejected(
                "unknown_event",
                event_name,
                self.current_state.name if self.current_state is not None else "NONE",
            )
            return False

        if self.current_state not in self.transitions[event_name]:
            # 当前状态下没有这条转移边：开始记录一段拒绝区间
            self._log_transition_rejected(
                "no_transition",
                event_name,
                self.current_state.name if self.current_state is not None else "NONE",
            )
            return False
        
        candidates = self.transitions[event_name][self.current_state]
        if not isinstance(candidates, tuple):
            candidates = (candidates,)
        target_state = None
        for cand in candidates:
            if self._check_condition(cand):
                target_state = cand
                break
        if target_state is None:
            self._log_transition_rejected(
                "condition_failed",
                event_name,
                self.current_state.name if self.current_state is not None else "NONE",
                "|".join(c.name for c in candidates),
            )
            return False
        # 恢复正常切换时，清掉上一类拒绝状态
        self._active_reject_key = None
        # 可以转换
        self.target_state = target_state
        if target_state != self.current_state:
            self._log_info(
                "fsm_transition_candidate event=%s current=%s target=%s",
                event_name,
                self.current_state.name if self.current_state is not None else "NONE",
                target_state.name,
            )
        return True

    def transition(self):
        """执行状态转换"""
        # 执行转换
        if self.target_state != self.current_state:
            previous_state = self.current_state
            next_state = self.target_state
            self._log_info(
                "fsm_transition_start event=%s from=%s to=%s robot_time=%.6f",
                self.last_event_name,
                previous_state.name if previous_state is not None else "NONE",
                next_state.name,
                float(getattr(self.robot_data_, "time_now_", 0.0)),
            )
            try:
                self.state_objects[self.current_state].on_exit()
                self.current_state = next_state
                self.state_objects[self.current_state].on_enter()
            except Exception as exc:
                self._log_error(
                    "fsm_transition_failed event=%s from=%s to=%s error=%r",
                    self.last_event_name,
                    previous_state.name if previous_state is not None else "NONE",
                    next_state.name,
                    exc,
                )
                raise
            # 记录转换时间
            self.robot_data_.record_transition()
            # 记录状态切换成功
            self._log_info(
                "fsm_transition event=%s from=%s to=%s robot_time=%.6f",
                self.last_event_name,
                previous_state.name if previous_state is not None else "NONE",
                self.current_state.name,
                float(getattr(self.robot_data_, "time_now_", 0.0)),
            )
        return True

    def _check_condition(self, target_state:FSMStateName) -> bool:
        """检查转换条件：调用源状态和目标状态的 check_transition，两者都允许才通过。"""
        if self.current_state != target_state:
            exit_ok = self.state_objects[self.current_state].check_transition(action="exit", target_state=target_state)
            enter_ok = self.state_objects[target_state].check_transition(
                action="enter",
                target_state=target_state,
                source_state=self.current_state,
            )
            return bool(exit_ok) and bool(enter_ok)
        return True

    def get_current_state(self) -> FSMStateName:
        """获取当前状态"""
        return self.current_state

    @timing_decorator
    def run_fsm(self, flag: ControlFlag):
        """运行状态机"""
        self.first_enter()
        self.trigger(flag)
        self.safe_guard()
        self.transition()
        self.state_objects[self.current_state].run(flag)

    @timing_decorator
    def safe_guard(self):
        # 例如：检查机器人是否处于安全状态，获取当前robotdata等信息，
        #TODO:电量低，强制降速切换到全身走
        # 检测是否接触
        # result = self.contact_detector.is_contact()
        # if not result:
        #     xlog.warning(f"离地, 切换到STOP状态")
        #     self.target_state = FSMStateName['STOP']

        # 电机状态检查
        for motor_id, error in self.robot_data_.diagnostic_info['motor_status'].items():
            if error != 0 and self.current_state != FSMStateName['DAMPING']:
                now = time.perf_counter()
                last_log_time = self._last_motor_error_log_time.get((motor_id, error), 0.0)
                if now - last_log_time >= 1.0:
                    self._log_error(
                        "fsm_safe_guard_override reason=motor_error current=%s previous_target=%s new_target=DAMPING motor=%s error=%s",
                        self.current_state.name if self.current_state is not None else "NONE",
                        self.target_state.name if self.target_state is not None else "NONE",
                        motor_id,
                        error,
                    )
                    self._last_motor_error_log_time[(motor_id, error)] = now
                self.target_state = FSMStateName['DAMPING']

        # 2. 摔倒检测 → STOP（受 safety.fall_protection 开关控制）
        if self._fall_protection and self.current_state not in (FSMStateName['STOP'], FSMStateName['DAMPING']):
            try:
                gravity = self.robot_data_.get_project_gravity()
                # 投影重力：直立时 z≈-1.0，摔倒倾斜时z分量趋向0或正值
                if gravity[2] > self._fall_gravity_threshold:
                    self._log_error("摔倒检测: projected_gravity=%s, 切换到STOP状态", gravity)
                    self.target_state = FSMStateName['STOP']
            except Exception:
                pass

def get_robot_fsm(robot_data: RobotData, config: Dict) -> RobotFSM:
    """工厂函数，返回机器人FSM实例"""
    return RobotFSMImpl(robot_data, config)


# 配置文件示例 (config.yaml)
"""
states:
  STOP: 0
  ZERO: 1
  MLP: 2
  BASEBALLCATCH: 3
  DAMPING: 4
  NOGRAVITY: 5
  BEYONDZERO: 6
  BEYONDMIMIC: 7
  STAND: 8
  NAVIGATE: 9
  WALKAMP: 10
  NIUKUA: 11

init_state: STOP
transitions:
  to_stop:
    - {from: ZERO, to: STOP}
    - {from: MLP, to: STOP}
    - {from: STAND, to: STOP}
    - {from: WALKAMP, to: STOP}
  
  to_zero:
    - {from: STOP, to: ZERO}
    - {from: MLP, to: ZERO}
"""


if __name__ == '__main__':
    # 创建状态机实例
    state_machine = RobotFSMImpl('config.yaml')

    # 触发转换
    state_machine.trigger('gotoSTOP')
    state_machine.trigger('gotoZERO')
