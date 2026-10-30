#!/usr/bin/env python3
"""
Analytical serial-parallel ankle conversion for the EVT parallel ankle.

This module keeps the public method names used by ``robot_interface.py`` while
replacing the neural-network FK in ``func_sp_trans_evt_opt.py`` with an
analytical/semi-analytical solver based on the geometry in
``/home/mig/Parallel_Ankle_Joint/dex_evt_papallel_ankle/parallel_ankle_ik.py``.

Conventions exposed by ``DexParallelAnkleSolver``:
- serial state order per side: [pitch, roll]
- internal parallel state order per side: [pitch-side crank, roll-side crank]
- Jacobian stored in this class maps [pitch_dot, roll_dot] to motor rates.

``AnalyticFuncSPTrans`` keeps the legacy public motor order used by
``func_sp_trans_evt_opt.py``. Per side that order is the reverse of the solver
geometry order, so the wrapper swaps the two motor rows at the API boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _foot_rotation(pitch: float, roll: float) -> np.ndarray:
    return _rotation_y(pitch) @ _rotation_x(roll)


@dataclass(frozen=True)
class LimbGeometry:
    name: str
    motor_origin: np.ndarray
    foot_anchor_zero: np.ndarray
    crank_vector_zero: np.ndarray
    rod_length: float


class DexParallelAnkleSolver:
    """Two-limb analytical solver for one ankle side."""

    def __init__(self, side: str) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side
        self.limbs = self._geometry(side)
        self._scalar_limbs = tuple(
            (
                float(limb.motor_origin[0]),
                float(limb.motor_origin[1]),
                float(limb.motor_origin[2]),
                float(limb.foot_anchor_zero[0]),
                float(limb.foot_anchor_zero[1]),
                float(limb.foot_anchor_zero[2]),
                float(limb.crank_vector_zero[0]),
                float(limb.crank_vector_zero[1]),
                float(limb.crank_vector_zero[2]),
                float(limb.rod_length**2),
            )
            for limb in self.limbs
        )

    @staticmethod
    def _geometry(side: str) -> tuple[LimbGeometry, LimbGeometry]:
        # Coordinates are in the ankle pitch frame, with the ankle roll joint at
        # the origin. The right side is the y-mirror of the left geometry; both
        # serial pitch and roll axes follow the URDF right-hand axes.
        y_sign = 1.0 if side == "left" else -1.0
        return (
            LimbGeometry(
                name=f"{side} pitch-side limb",
                motor_origin=np.array([0.024938, y_sign * 0.023567, 0.21876], dtype=np.float64),
                foot_anchor_zero=np.array([-0.03161, y_sign * 0.03602, 0.01512], dtype=np.float64),
                crank_vector_zero=np.array([-0.031643, y_sign * 0.0125, 0.014957], dtype=np.float64),
                rod_length=0.2200111579965889,
            ),
            LimbGeometry(
                name=f"{side} roll-side limb",
                motor_origin=np.array([0.030921, -y_sign * 0.023566, 0.2885], dtype=np.float64),
                foot_anchor_zero=np.array([-0.03161, -y_sign * 0.03602, 0.01512], dtype=np.float64),
                crank_vector_zero=np.array([-0.031634, -y_sign * 0.0125, 0.014976], dtype=np.float64),
                rod_length=0.29000656796183083,
            ),
        )

    @staticmethod
    def _choose_candidate(candidates: np.ndarray, reference: float) -> float:
        distances = np.abs(_wrap_to_pi(candidates - reference))
        return float(candidates[int(np.argmin(distances))])

    def _solve_single_limb(
        self,
        limb: LimbGeometry,
        foot_rotation: np.ndarray,
        reference_theta: float,
        tol: float = 1e-9,
    ) -> float:
        c_point = foot_rotation @ limb.foot_anchor_zero
        d = c_point - limb.motor_origin
        u = limb.crank_vector_zero

        p_term = d[0] * u[0] + d[2] * u[2]
        q_term = d[0] * u[2] - d[2] * u[0]
        s_term = 0.5 * (d @ d + u @ u - limb.rod_length**2) - d[1] * u[1]

        rho = math.hypot(float(p_term), float(q_term))
        if rho < tol:
            raise ValueError(f"{limb.name}: degenerate limb geometry, rho={rho}")

        ratio = float(s_term) / rho
        if ratio < -1.0 - tol or ratio > 1.0 + tol:
            raise ValueError(f"{limb.name}: pose is unreachable, |S/rho|={abs(ratio):.6f} > 1")

        phase = math.atan2(float(q_term), float(p_term))
        offset = math.acos(min(1.0, max(-1.0, ratio)))
        candidates = _wrap_to_pi(np.array([phase + offset, phase - offset], dtype=np.float64))
        return self._choose_candidate(candidates, reference_theta)

    def ik(
        self,
        pitch: float,
        roll: float,
        reference_theta: np.ndarray | tuple[float, float] | None = None,
    ) -> np.ndarray:
        if reference_theta is None:
            reference_theta = (0.0, 0.0)
        reference_theta = np.asarray(reference_theta, dtype=np.float64).reshape(2)

        foot_rotation = _foot_rotation(float(pitch), float(roll))
        theta1 = self._solve_single_limb(self.limbs[0], foot_rotation, float(reference_theta[0]))
        theta2 = self._solve_single_limb(self.limbs[1], foot_rotation, float(reference_theta[1]))
        return np.array([theta1, theta2], dtype=np.float64)

    @staticmethod
    def _solve_2x2(
        m00: float,
        m01: float,
        m10: float,
        m11: float,
        b0: float,
        b1: float,
        det_tol: float = 1e-12,
    ) -> np.ndarray:
        det = m00 * m11 - m01 * m10
        if abs(det) < det_tol:
            raise RuntimeError(f"2x2 system is singular, det={det}")
        inv_det = 1.0 / det
        return np.array(
            [(m11 * b0 - m01 * b1) * inv_det, (-m10 * b0 + m00 * b1) * inv_det],
            dtype=np.float64,
        )

    def _pose_residual_and_jacobian_scalar(
        self,
        pitch: float,
        roll: float,
        theta_target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        theta_target = np.asarray(theta_target, dtype=np.float64).reshape(2)
        cp = math.cos(float(pitch))
        sp = math.sin(float(pitch))
        cr = math.cos(float(roll))
        sr = math.sin(float(roll))

        residual = np.zeros(2, dtype=np.float64)
        jacobian = np.zeros((2, 2), dtype=np.float64)  # columns: [roll, pitch]

        for idx, (theta_i, limb_data) in enumerate(zip(theta_target, self._scalar_limbs)):
            ax, ay, az, cx0, cy0, cz0, ux, uy, uz, rod_length_sq = limb_data

            yz_term = sr * cy0 + cr * cz0
            cy = cr * cy0 - sr * cz0
            cx = cp * cx0 + sp * yz_term
            cz = -sp * cx0 + cp * yz_term

            ct = math.cos(float(theta_i))
            st = math.sin(float(theta_i))
            bx = ax + ct * ux + st * uz
            by = ay + uy
            bz = az - st * ux + ct * uz

            rrod_x = cx - bx
            rrod_y = cy - by
            rrod_z = cz - bz

            residual[idx] = 0.5 * (
                rrod_x * rrod_x + rrod_y * rrod_y + rrod_z * rrod_z - rod_length_sq
            )

            dcdroll_x = sp * cy
            dcdroll_y = -sp * cx - cp * cz
            dcdroll_z = cp * cy
            jacobian[idx, 0] = rrod_x * dcdroll_x + rrod_y * dcdroll_y + rrod_z * dcdroll_z
            jacobian[idx, 1] = rrod_x * cz - rrod_z * cx

        return residual, jacobian

    def jacobian_roll_pitch(
        self,
        pitch: float,
        roll: float,
        theta: np.ndarray | None = None,
        denominator_tol: float = 1e-10,
    ) -> np.ndarray:
        if theta is None:
            theta = self.ik(pitch, roll)
        theta = np.asarray(theta, dtype=np.float64).reshape(2)
        cp = math.cos(float(pitch))
        sp = math.sin(float(pitch))
        cr = math.cos(float(roll))
        sr = math.sin(float(roll))
        jacobian_reduced = np.zeros((2, 2), dtype=np.float64)  # columns: [roll, pitch]

        for idx, (theta_i, limb_data) in enumerate(zip(theta, self._scalar_limbs)):
            ax, ay, az, cx0, cy0, cz0, ux, uy, uz, _rod_length_sq = limb_data

            yz_term = sr * cy0 + cr * cz0
            cy = cr * cy0 - sr * cz0
            cx = cp * cx0 + sp * yz_term
            cz = -sp * cx0 + cp * yz_term

            ct = math.cos(float(theta_i))
            st = math.sin(float(theta_i))
            rbar_x = ct * ux + st * uz
            rbar_z = -st * ux + ct * uz
            bx = ax + rbar_x
            by = ay + uy
            bz = az + rbar_z

            rrod_x = cx - bx
            rrod_y = cy - by
            rrod_z = cz - bz

            numerator_roll = (
                rrod_x * (sp * cy)
                + rrod_y * (-sp * cx - cp * cz)
                + rrod_z * (cp * cy)
            )
            numerator_pitch = rrod_x * cz - rrod_z * cx
            denominator = rbar_z * rrod_x - rbar_x * rrod_z
            if abs(denominator) < denominator_tol:
                raise RuntimeError(f"{self.side} ankle Jacobian denominator too small: {denominator}")

            jacobian_reduced[idx, 0] = numerator_roll / denominator
            jacobian_reduced[idx, 1] = numerator_pitch / denominator

        if not np.all(np.isfinite(jacobian_reduced)):
            raise RuntimeError(f"{self.side} ankle Jacobian contains non-finite values")
        return jacobian_reduced

    def jacobian_pitch_roll(self, pitch: float, roll: float, theta: np.ndarray | None = None) -> np.ndarray:
        return self.jacobian_roll_pitch(pitch, roll, theta=theta)[:, [1, 0]]

    def forward_kinematics(
        self,
        theta_target: np.ndarray,
        initial_guess: tuple[float, float] | np.ndarray = (0.0, 0.0),
        max_iterations: int = 40,
        tolerance: float = 1e-10,
        step_tolerance: float = 1e-12,
    ) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        theta_target = np.asarray(theta_target, dtype=np.float64).reshape(2)
        initial_guess = np.asarray(initial_guess, dtype=np.float64).reshape(2)  # [pitch, roll]
        state = np.array([initial_guess[1], initial_guess[0]], dtype=np.float64)  # [roll, pitch]

        for iteration in range(max_iterations):
            roll = float(state[0])
            pitch = float(state[1])
            residual, jacobian = self._pose_residual_and_jacobian_scalar(pitch, roll, theta_target)
            residual_norm = float(math.hypot(float(residual[0]), float(residual[1])))

            if residual_norm < tolerance:
                theta_check = self.ik(pitch, roll, reference_theta=theta_target)
                theta_residual_norm = float(np.linalg.norm(_wrap_to_pi(theta_check - theta_target)))
                return (
                    np.array([pitch, roll], dtype=np.float64),
                    {
                        "iterations": iteration,
                        "converged": True,
                        "residual_norm": theta_residual_norm,
                        "closure_residual_norm": residual_norm,
                        "step_norm": 0.0,
                    },
                )

            step = self._solve_2x2(
                float(jacobian[0, 0]),
                float(jacobian[0, 1]),
                float(jacobian[1, 0]),
                float(jacobian[1, 1]),
                float(residual[0]),
                float(residual[1]),
            )
            step_norm = float(math.hypot(float(step[0]), float(step[1])))
            if step_norm < step_tolerance:
                raise RuntimeError(f"{self.side} ankle FK stalled before convergence")

            for scale in (1.0, 0.5, 0.25, 0.1, 0.05):
                candidate = state - scale * step
                candidate_residual, _ = self._pose_residual_and_jacobian_scalar(
                    float(candidate[1]), float(candidate[0]), theta_target
                )
                candidate_norm = float(
                    math.hypot(float(candidate_residual[0]), float(candidate_residual[1]))
                )
                if candidate_norm < residual_norm:
                    state = candidate
                    break
            else:
                raise RuntimeError(f"{self.side} ankle FK line search failed at iteration {iteration}")

        raise RuntimeError(f"{self.side} ankle FK failed to converge in {max_iterations} iterations")


class AnalyticFuncSPTrans:
    """Drop-in serial-parallel transformer with analytical ankle geometry."""

    _LEGACY_TO_SOLVER_ORDER = np.array([1, 0], dtype=np.int64)
    _SOLVER_TO_LEGACY_ORDER = np.array([1, 0], dtype=np.int64)
    ##参考以前的部署代码，并联电机顺序需要换一下
    
    def __init__(self, damping: float = 1e-4, torque_limit: float = 50.0) -> None:
        self.left = DexParallelAnkleSolver("left")
        self.right = DexParallelAnkleSolver("right")
        self.damping = float(damping)
        self.torque_limit = float(torque_limit)
        self.last_error = ""

        self.qP_l_est = np.zeros(2, dtype=np.float64)
        self.qP_r_est = np.zeros(2, dtype=np.float64)
        self.qDotP_l_est = np.zeros(2, dtype=np.float64)
        self.qDotP_r_est = np.zeros(2, dtype=np.float64)
        self.torP_l_est = np.zeros(2, dtype=np.float64)
        self.torP_r_est = np.zeros(2, dtype=np.float64)

        self.qS_l_est = np.zeros(2, dtype=np.float64)
        self.qS_r_est = np.zeros(2, dtype=np.float64)
        self.qDotS_l_est = np.zeros(2, dtype=np.float64)
        self.qDotS_r_est = np.zeros(2, dtype=np.float64)
        self.torS_l_est = np.zeros(2, dtype=np.float64)
        self.torS_r_est = np.zeros(2, dtype=np.float64)

        self.qS_l_ref = np.zeros(2, dtype=np.float64)
        self.qS_r_ref = np.zeros(2, dtype=np.float64)
        self.qDotS_l_ref = np.zeros(2, dtype=np.float64)
        self.qDotS_r_ref = np.zeros(2, dtype=np.float64)
        self.torS_l_ref = np.zeros(2, dtype=np.float64)
        self.torS_r_ref = np.zeros(2, dtype=np.float64)

        self.qP_l_ref = np.zeros(2, dtype=np.float64)
        self.qP_r_ref = np.zeros(2, dtype=np.float64)
        self.qDotP_l_ref = np.zeros(2, dtype=np.float64)
        self.qDotP_r_ref = np.zeros(2, dtype=np.float64)
        self.torP_l_ref = np.zeros(2, dtype=np.float64)
        self.torP_r_ref = np.zeros(2, dtype=np.float64)

        self.JLeft = self._jacobian_to_legacy_order(
            self.left.jacobian_pitch_roll(0.0, 0.0, theta=np.zeros(2))
        )
        self.JRight = self._jacobian_to_legacy_order(
            self.right.jacobian_pitch_roll(0.0, 0.0, theta=np.zeros(2))
        )

    @classmethod
    def _motors_to_solver_order(cls, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64).reshape(2)[cls._LEGACY_TO_SOLVER_ORDER]

    @classmethod
    def _motors_to_legacy_order(cls, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64).reshape(2)[cls._SOLVER_TO_LEGACY_ORDER]

    @classmethod
    def _jacobian_to_legacy_order(cls, jacobian: np.ndarray) -> np.ndarray:
        return np.asarray(jacobian, dtype=np.float64).reshape(2, 2)[cls._SOLVER_TO_LEGACY_ORDER, :]

    @staticmethod
    def _is_finite_vec(value: np.ndarray, size: int) -> bool:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        return arr.size == size and bool(np.all(np.isfinite(arr)))

    def _set_error(self, message: str) -> bool:
        self.last_error = message
        return False

    def _damped_solve(self, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        rhs = np.asarray(rhs, dtype=np.float64)
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
            raise np.linalg.LinAlgError("non-finite matrix or rhs")
        try:
            cond = np.linalg.cond(matrix)
            if np.isfinite(cond) and cond < 1.0 / max(self.damping, 1e-12):
                return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            pass

        lhs = matrix.T @ matrix + (self.damping * self.damping) * np.eye(matrix.shape[1])
        return np.linalg.solve(lhs, matrix.T @ rhs)

    def set_p_est(self, qP_est: np.ndarray, qDotP_est: np.ndarray, qTorP_est: np.ndarray) -> bool:
        if not (
            self._is_finite_vec(qP_est, 4)
            and self._is_finite_vec(qDotP_est, 4)
            and self._is_finite_vec(qTorP_est, 4)
        ):
            return self._set_error("set_p_est received non-finite or wrongly sized input")
        qP_est = np.asarray(qP_est, dtype=np.float64).reshape(4)
        qDotP_est = np.asarray(qDotP_est, dtype=np.float64).reshape(4)
        qTorP_est = np.asarray(qTorP_est, dtype=np.float64).reshape(4)
        self.qP_l_est[:] = qP_est[:2]
        self.qP_r_est[:] = qP_est[2:]
        self.qDotP_l_est[:] = qDotP_est[:2]
        self.qDotP_r_est[:] = qDotP_est[2:]
        self.torP_l_est[:] = qTorP_est[:2]
        self.torP_r_est[:] = qTorP_est[2:]
        self.last_error = ""
        return True

    def setPEst(self, qP_est: np.ndarray, qDotP_est: np.ndarray, qTorP_est: np.ndarray) -> bool:
        return self.set_p_est(qP_est, qDotP_est, qTorP_est)

    def calcFK(self) -> bool:
        ok = True
        try:
            self.qS_l_est[:] = self.left.forward_kinematics(
                self._motors_to_solver_order(self.qP_l_est), initial_guess=self.qS_l_est
            )[0]
        except Exception as exc:
            ok = self._set_error(f"left FK failed: {exc}")
        try:
            self.qS_r_est[:] = self.right.forward_kinematics(
                self._motors_to_solver_order(self.qP_r_est), initial_guess=self.qS_r_est
            )[0]
        except Exception as exc:
            ok = self._set_error(f"right FK failed: {exc}") and ok
        return ok

    def calc_fk(self) -> bool:
        return self.calcFK()

    def calcIK(self) -> bool:
        ok = True
        try:
            self.JLeft[:, :] = self._jacobian_to_legacy_order(
                self.left.jacobian_pitch_roll(
                    float(self.qS_l_est[0]),
                    float(self.qS_l_est[1]),
                    theta=self._motors_to_solver_order(self.qP_l_est),
                )
            )
            self.qDotS_l_est[:] = self._damped_solve(self.JLeft, self.qDotP_l_est)
            self.torS_l_est[:] = self.JLeft.T @ self.torP_l_est
        except Exception as exc:
            ok = self._set_error(f"left Jacobian transform failed: {exc}")

        try:
            self.JRight[:, :] = self._jacobian_to_legacy_order(
                self.right.jacobian_pitch_roll(
                    float(self.qS_r_est[0]),
                    float(self.qS_r_est[1]),
                    theta=self._motors_to_solver_order(self.qP_r_est),
                )
            )
            self.qDotS_r_est[:] = self._damped_solve(self.JRight, self.qDotP_r_est)
            self.torS_r_est[:] = self.JRight.T @ self.torP_r_est
        except Exception as exc:
            ok = self._set_error(f"right Jacobian transform failed: {exc}") and ok
        return ok

    def calc_ik(self) -> bool:
        return self.calcIK()

    def get_s_state(self, qS_est: np.ndarray, qDotS_est: np.ndarray, torS_est: np.ndarray) -> bool:
        qS_est[:2] = self.qS_l_est
        qS_est[2:] = self.qS_r_est
        qDotS_est[:2] = self.qDotS_l_est
        qDotS_est[2:] = self.qDotS_r_est
        torS_est[:2] = self.torS_l_est
        torS_est[2:] = self.torS_r_est
        return True

    def getSState(
        self,
        qS_est: np.ndarray | None = None,
        qDotS_est: np.ndarray | None = None,
        torS_est: np.ndarray | None = None,
    ):
        if qS_est is None or qDotS_est is None or torS_est is None:
            qS_est = np.zeros(4, dtype=np.float64)
            qDotS_est = np.zeros(4, dtype=np.float64)
            torS_est = np.zeros(4, dtype=np.float64)
            self.get_s_state(qS_est, qDotS_est, torS_est)
            return qS_est, qDotS_est, torS_est
        return self.get_s_state(qS_est, qDotS_est, torS_est)

    def set_s_des(self, qS_ref: np.ndarray, qDotS_ref: np.ndarray, torS_des: np.ndarray) -> bool:
        if not (
            self._is_finite_vec(qS_ref, 4)
            and self._is_finite_vec(qDotS_ref, 4)
            and self._is_finite_vec(torS_des, 4)
        ):
            return self._set_error("set_s_des received non-finite or wrongly sized input")
        qS_ref = np.asarray(qS_ref, dtype=np.float64).reshape(4)
        qDotS_ref = np.asarray(qDotS_ref, dtype=np.float64).reshape(4)
        torS_des = np.asarray(torS_des, dtype=np.float64).reshape(4)
        self.qS_l_ref[:] = qS_ref[:2]
        self.qS_r_ref[:] = qS_ref[2:]
        self.qDotS_l_ref[:] = qDotS_ref[:2]
        self.qDotS_r_ref[:] = qDotS_ref[2:]
        self.torS_l_ref[:] = torS_des[:2]
        self.torS_r_ref[:] = torS_des[2:]
        self.last_error = ""
        return True

    def setSDes(self, qS_ref: np.ndarray, qDotS_ref: np.ndarray, torS_des: np.ndarray) -> bool:
        return self.set_s_des(qS_ref, qDotS_ref, torS_des)

    def calc_joint_pos_ref(self) -> bool:
        ok = True
        try:
            self.qP_l_ref[:] = self._motors_to_legacy_order(
                self.left.ik(
                    float(self.qS_l_ref[0]),
                    float(self.qS_l_ref[1]),
                    reference_theta=self._motors_to_solver_order(self.qP_l_ref),
                )
            )
        except Exception as exc:
            ok = self._set_error(f"left IK failed: {exc}")
        try:
            self.qP_r_ref[:] = self._motors_to_legacy_order(
                self.right.ik(
                    float(self.qS_r_ref[0]),
                    float(self.qS_r_ref[1]),
                    reference_theta=self._motors_to_solver_order(self.qP_r_ref),
                )
            )
        except Exception as exc:
            ok = self._set_error(f"right IK failed: {exc}") and ok
        return ok

    def calcJointPosRef(self) -> bool:
        return self.calc_joint_pos_ref()

    def calc_joint_tor_des(self) -> bool:
        ok = True
        try:
            self.qDotP_l_ref[:] = self.JLeft @ self.qDotS_l_ref
            self.torP_l_ref[:] = self._damped_solve(self.JLeft.T, self.torS_l_ref)
        except Exception as exc:
            ok = self._set_error(f"left desired torque transform failed: {exc}")

        try:
            self.qDotP_r_ref[:] = self.JRight @ self.qDotS_r_ref
            self.torP_r_ref[:] = self._damped_solve(self.JRight.T, self.torS_r_ref)
        except Exception as exc:
            ok = self._set_error(f"right desired torque transform failed: {exc}") and ok

        np.clip(self.torP_l_ref, -self.torque_limit, self.torque_limit, out=self.torP_l_ref)
        np.clip(self.torP_r_ref, -self.torque_limit, self.torque_limit, out=self.torP_r_ref)
        return ok

    def calcJointTorDes(self) -> bool:
        return self.calc_joint_tor_des()

    def get_p_des(self, qP_des: np.ndarray, qDotP_des: np.ndarray, torP_des: np.ndarray) -> bool:
        qP_des[:2] = self.qP_l_ref
        qP_des[2:] = self.qP_r_ref
        qDotP_des[:2] = self.qDotP_l_ref
        qDotP_des[2:] = self.qDotP_r_ref
        torP_des[:2] = self.torP_l_ref
        torP_des[2:] = self.torP_r_ref
        return True

    def getPDes(
        self,
        qP_des: np.ndarray | None = None,
        qDotP_des: np.ndarray | None = None,
        torP_des: np.ndarray | None = None,
    ):
        if qP_des is None or qDotP_des is None or torP_des is None:
            qP_des = np.zeros(4, dtype=np.float64)
            qDotP_des = np.zeros(4, dtype=np.float64)
            torP_des = np.zeros(4, dtype=np.float64)
            self.get_p_des(qP_des, qDotP_des, torP_des)
            return qP_des, qDotP_des, torP_des
        return self.get_p_des(qP_des, qDotP_des, torP_des)


# Alias mirrors the naming used by the optimized transformer modules.
OptimizedFuncSPTrans = AnalyticFuncSPTrans
