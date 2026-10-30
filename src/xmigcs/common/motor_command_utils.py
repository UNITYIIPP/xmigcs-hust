"""
Motor command utilities extracted from RobotInterface.

Provides:
  - convert_to_motor_commands: transform desired joint commands to motor frame
  - Interpolation logic for state transitions
"""
from __future__ import annotations

import numpy as np
from xmigcs.FSM.fsm_base import FSMStateName


def convert_to_motor_commands(
    robot_data,
    floating_base_dof: int,
    zero_offset: np.ndarray,
    motor_dir: np.ndarray,
    zero_pos: np.ndarray,
    disable_joints: bool = False,
    clip_fn=None,
) -> None:
    """Transform desired joint commands from control frame to motor frame.

    Applies: q_motor = (q_desired - zero_offset) * motor_dir + zero_pos
    And similarly for velocity and torque.

    Args:
        robot_data: RobotData instance (modified in place).
        floating_base_dof: Number of floating base DOFs (typically 6).
        zero_offset: Joint zero offsets.
        motor_dir: Motor direction signs (+1 or -1).
        zero_pos: Motor zero positions.
        disable_joints: If True, zero out all gains and torques.
        clip_fn: Optional callable(motor_id, position) → (within_limit, clipped_pos).
    """
    q_d_reordered = robot_data.q_d_[floating_base_dof:].copy()
    qdot_d_reordered = robot_data.q_dot_d_[floating_base_dof:].copy()
    tor_d_reordered = robot_data.tau_d_[floating_base_dof:].copy()

    temp_q = np.empty_like(q_d_reordered)
    np.subtract(q_d_reordered, zero_offset, out=temp_q)
    np.multiply(temp_q, motor_dir, out=temp_q)
    np.add(temp_q, zero_pos, out=robot_data.q_d_[floating_base_dof:])

    np.multiply(qdot_d_reordered, motor_dir, out=robot_data.q_dot_d_[floating_base_dof:])
    np.multiply(tor_d_reordered, motor_dir, out=robot_data.tau_d_[floating_base_dof:])

    if disable_joints:
        robot_data.joint_kp_p_.fill(0.0)
        robot_data.joint_kd_p_.fill(0.0)
        robot_data.tau_d_.fill(0.0)


def interpolate_action(
    robot_data,
    current_state: FSMStateName,
    last_state: FSMStateName,
    state_motor_id_map: dict,
    floating_base_dof: int,
    transition_time: float,
    get_current_transition_time_fn,
    cached_transition_start_time,
    cached_last_state_serial_qd,
    cached_last_state_serial_kp,
    cached_last_state_serial_kd,
    dynamic_model=None,
    dynamic_inter_status: list = None,
) -> tuple:
    """Interpolate motor commands between FSM state transitions.

    Returns:
        Tuple of (updated cached_transition_start_time, cached_last_state_serial_qd,
                   cached_last_state_serial_kp, cached_last_state_serial_kd)
    """
    if current_state == FSMStateName.STOP or last_state == FSMStateName.STOP:
        return (cached_transition_start_time, cached_last_state_serial_qd,
                cached_last_state_serial_kp, cached_last_state_serial_kd)

    if robot_data.trans_start_time <= 0.01:
        return (cached_transition_start_time, cached_last_state_serial_qd,
                cached_last_state_serial_kp, cached_last_state_serial_kd)

    trans_time = get_current_transition_time_fn()
    if trans_time == 0.0:
        return (cached_transition_start_time, cached_last_state_serial_qd,
                cached_last_state_serial_kp, cached_last_state_serial_kd)

    alpha = (robot_data.time_now_ - robot_data.trans_start_time) / trans_time
    if alpha > 1.0:
        return (cached_transition_start_time, cached_last_state_serial_qd,
                cached_last_state_serial_kp, cached_last_state_serial_kd)

    alpha = np.clip(alpha, 0.0, 1.0)

    transition_start_time = robot_data.trans_start_time
    if cached_transition_start_time != transition_start_time:
        cached_transition_start_time = transition_start_time
        cached_last_state_serial_qd = robot_data.get_last_state_serial_qd()
        cached_last_state_serial_kp = robot_data.get_last_state_serial_kp()
        cached_last_state_serial_kd = robot_data.get_last_state_serial_kd()
        if dynamic_model is not None and last_state in (dynamic_inter_status or []):
            q_d_s_arm, joint_index = dynamic_model.compute_desired_joint_pos(
                cached_last_state_serial_kp, cached_last_state_serial_kd)
            cached_last_state_serial_qd[joint_index] = q_d_s_arm

    q_d_ = robot_data.get_desired_joint_pos()
    joint_kp = robot_data.get_joint_kp()
    joint_kd = robot_data.get_joint_kd()

    current_ids = state_motor_id_map.get(current_state, [])
    for motor_id in current_ids:
        idx = floating_base_dof + motor_id
        robot_data.q_d_[idx] = (cached_last_state_serial_qd[motor_id] * (1 - alpha) +
                                q_d_[motor_id] * alpha)
        robot_data.joint_kp_p_[motor_id] = (cached_last_state_serial_kp[motor_id] * (1 - alpha) +
                                             joint_kp[motor_id] * alpha)
        robot_data.joint_kd_p_[motor_id] = (cached_last_state_serial_kd[motor_id] * (1 - alpha) +
                                             joint_kd[motor_id] * alpha)

    return (cached_transition_start_time, cached_last_state_serial_qd,
            cached_last_state_serial_kp, cached_last_state_serial_kd)
