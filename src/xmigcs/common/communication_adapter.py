"""
Communication adapter interface for robot hardware.

Defines the abstract seam where behaviour can be altered without editing in place.
Each concrete adapter wraps a specific communication mode (ROS2, LPC, SHM RPC, TCP RPC).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class CommunicationAdapter(ABC):
    """Abstract interface for robot hardware communication.

    Concrete adapters wrap a specific communication mode and translate
    between the adapter's protocol and the unified queue-based interface
    consumed by RobotInterface.
    """

    @abstractmethod
    def init(self, node, robot_data, queues: dict, id_map) -> None:
        """Initialize the adapter (create publishers, subscribers, connections).

        Args:
            node: ROS2 node instance (for ROS-based adapters).
            robot_data: RobotData instance.
            queues: Dict of PeekableQueue instances for sensor data.
            id_map: BodyServoIdMap instance.
        """
        ...

    @abstractmethod
    def send_motor_commands(self, robot_data, current_state, control_status: dict, id_map) -> None:
        """Send motor commands to the robot.

        Args:
            robot_data: RobotData with desired commands filled in.
            current_state: Current FSM state enum.
            control_status: Dict of state→motor_id mappings.
            id_map: BodyServoIdMap instance.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the adapter is connected and ready."""
        ...


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, type[CommunicationAdapter]] = {}


def register_adapter(name: str, cls: type[CommunicationAdapter]) -> None:
    """Register a communication adapter class under a given name."""
    _ADAPTER_REGISTRY[name] = cls


def get_adapter(name: str) -> Optional[type[CommunicationAdapter]]:
    """Look up a registered adapter class by name."""
    return _ADAPTER_REGISTRY.get(name)


def list_adapters() -> list[str]:
    """Return the names of all registered adapters."""
    return list(_ADAPTER_REGISTRY.keys())
