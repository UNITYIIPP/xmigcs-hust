"""
Finite State Machine (FSM) Module
Python equivalent of the C++ FSM system
"""
from abc import ABC, abstractmethod
from enum import Enum

from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData

class FSMStateName(Enum):
    """FSM状态枚举"""
    STOP = 0  # 停止状态

    @classmethod
    def extend(cls, name: str, value: int):
        """Add a state once and reject ambiguous enum values."""
        value = int(value)
        if name in cls.__members__:
            member = cls.__members__[name]
            if member.value != value:
                raise ValueError(
                    f"FSM state {name!r} already uses value {member.value}, "
                    f"cannot change it to {value}"
                )
            return member

        existing = cls._value2member_map_.get(value)
        if existing is not None:
            raise ValueError(
                f"FSM state value {value} is already used by {existing.name!r}; "
                f"cannot assign it to {name!r}"
            )

        member = object.__new__(cls)
        member._value_ = value
        member._name_ = name
        cls._member_map_[name] = member
        cls._value2member_map_[value] = member
        return member


class FSMState(ABC):
    """FSM状态抽象基类"""

    def __init__(self, robot_data: RobotData):
        self.robot_data_ = robot_data

    @abstractmethod
    def on_enter(self):
        """进入状态时的行为"""
        pass

    @abstractmethod
    def run(self, flag: ControlFlag):
        """运行状态的正常行为"""
        pass

    @abstractmethod
    def on_exit(self):
        """退出状态时的行为"""
        pass

    def check_transition(self, action: str = "", target_state=None, **kwargs) -> bool:
        """检查状态转换是否允许。

        Args:
            action: "exit" 或 "enter"
            target_state: 目标状态枚举值
        Returns:
            True 表示允许切换（默认），False 表示拒绝
        """
        return True

class RobotFSM(ABC):
    """机器人FSM抽象基类"""

    def __init__(self, robot_data: RobotData):
        self.robot_data_ = robot_data
        # self.disable_joints_ = False

    @abstractmethod
    def run_fsm(self, flag: ControlFlag):
        """运行FSM"""
        pass

    @abstractmethod
    def get_current_state(self) -> FSMStateName:
        """获取当前FSM状态"""
        pass
