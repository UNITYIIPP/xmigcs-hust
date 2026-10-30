"""Consolidated quaternion / rotation math utilities for xmigcs.

All quaternions use the **[w, x, y, z]** convention (scalar-first) unless
explicitly noted otherwise (e.g. ``quat_rotate_inverse_numpy`` accepts xyzw).

Functions
---------
euler_to_quaternion_scipy  – Euler angles → quaternion
matrix_from_quat           – Quaternion → 3×3 rotation matrix
quat_mul                   – Hamilton product of two quaternions
euler_single_axis_to_quat  – Single-axis angle → quaternion
quat_rotate_inverse_numpy  – Rotate vector by inverse of quaternion
yaw_quat                   – Extract yaw-only quaternion from full quaternion
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Euler ↔ Quaternion
# ---------------------------------------------------------------------------

def euler_to_quaternion_scipy(
    roll: float,
    pitch: float,
    yaw: float,
    degrees: bool = False,
    out_xyzw: bool = False,
) -> np.ndarray:
    """Convert Euler angles (XYZ / roll-pitch-yaw) to a quaternion.

    Uses ``scipy.spatial.transform.Rotation.from_euler`` internally.

    Parameters
    ----------
    roll : float
        Rotation angle about the X axis.
    pitch : float
        Rotation angle about the Y axis.
    yaw : float
        Rotation angle about the Z axis.
    degrees : bool, optional
        If *True*, the input angles are interpreted as degrees instead of
        radians.  Default is *False*.
    out_xyzw : bool, optional
        If *True*, return the quaternion in **[x, y, z, w]** order (the
        convention used by scipy and many graphics libraries).  Default is
        *False*, which returns **[w, x, y, z]**.

    Returns
    -------
    np.ndarray
        Quaternion as a 1-D array, dtype ``np.float32``.
    """
    r = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=degrees)
    q = r.as_quat()  # scipy convention: [x, y, z, w]
    if out_xyzw:
        return np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Quaternion → Matrix
# ---------------------------------------------------------------------------

def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Compute a 3×3 rotation matrix from a quaternion.

    Parameters
    ----------
    q : np.ndarray
        Quaternion in **[w, x, y, z]** order.

    Returns
    -------
    np.ndarray
        3×3 rotation matrix.
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


# ---------------------------------------------------------------------------
# Quaternion multiplication
# ---------------------------------------------------------------------------

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions.

    Both inputs and the output use **[w, x, y, z]** order.

    Parameters
    ----------
    q1, q2 : np.ndarray
        Quaternions in **[w, x, y, z]** order.

    Returns
    -------
    np.ndarray
        Product quaternion ``q1 ⊗ q2`` in **[w, x, y, z]** order.
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.array([w, x, y, z])


# ---------------------------------------------------------------------------
# Single-axis angle → Quaternion
# ---------------------------------------------------------------------------

def euler_single_axis_to_quat(
    angle: float,
    axis: str | np.ndarray,
    degrees: bool = False,
) -> np.ndarray:
    """Convert a single-axis rotation to a quaternion.

    Parameters
    ----------
    angle : float
        Rotation angle (radians by default, or degrees if *degrees=True*).
    axis : str | np.ndarray
        Rotation axis.  Accepted string values are ``'x'``, ``'y'``, ``'z'``
        (case-sensitive).  Alternatively, a 3-element array-like specifying
        an arbitrary axis vector (will be normalised internally).
    degrees : bool, optional
        If *True*, *angle* is interpreted as degrees.  Default is *False*.

    Returns
    -------
    np.ndarray
        Quaternion in **[w, x, y, z]** order.
    """
    if degrees:
        angle = np.radians(angle)
    ha = angle * 0.5
    c, s = np.cos(ha), np.sin(ha)

    # Fast path for canonical axes
    if isinstance(axis, str):
        components = {
            "x": (c, s, 0.0, 0.0),
            "y": (c, 0.0, s, 0.0),
            "z": (c, 0.0, 0.0, s),
        }
        if axis not in components:
            raise ValueError(f"Unknown axis: {axis!r}. Must be 'x', 'y', or 'z'.")
        return np.array(components[axis])

    # General axis vector
    axis_vec = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis_vec)
    if norm == 0:
        raise ValueError("Axis vector must not be the zero vector.")
    axis_vec = axis_vec / norm
    return np.array([c, *(s * axis_vec)])


# ---------------------------------------------------------------------------
# Inverse quaternion rotation (xyzw convention)
# ---------------------------------------------------------------------------

def quat_rotate_inverse_numpy(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by the **inverse** of a quaternion.

    This computes ``q⁻¹ ⊗ v ⊗ q`` using the vector formula, avoiding an
    explicit quaternion inverse.

    Parameters
    ----------
    q_xyzw : np.ndarray
        Quaternion in **[x, y, z, w]** order (scalar-last, matching the
        convention of scipy and many physics engines).
    v : np.ndarray
        3-D vector to rotate.

    Returns
    -------
    np.ndarray
        Rotated 3-D vector.
    """
    q_w = q_xyzw[3]
    q_v = q_xyzw[:3]
    return (
        v * (2.0 * q_w * q_w - 1.0)
        - np.cross(q_v, v) * (2.0 * q_w)
        + q_v * (2.0 * np.dot(q_v, v))
    )


# ---------------------------------------------------------------------------
# Yaw-only quaternion
# ---------------------------------------------------------------------------

def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Extract the yaw (Z-rotation) component of a quaternion.

    Given a quaternion **q** that encodes a full 3-D rotation, this function
    returns a new quaternion that preserves only the rotation about the Z
    (yaw) axis.

    The approach uses the forward direction of the rotation (the transformed
    X-axis) projected onto the XY-plane, then computes the yaw angle via
    ``arctan2`` and packs it back into a quaternion.

    Parameters
    ----------
    q : np.ndarray
        Quaternion in **[w, x, y, z]** order.

    Returns
    -------
    np.ndarray
        Yaw-only quaternion in **[w, x, y, z]** order.
    """
    # Forward direction (X-axis) under the rotation q
    w, x, y, z = q[0], q[1], q[2], q[3]
    # R * [1, 0, 0] – first column of the rotation matrix
    fwd_x = 1 - 2 * (y**2 + z**2)
    fwd_y = 2 * (x * y + z * w)

    yaw = np.arctan2(fwd_y, fwd_x)
    ha = yaw * 0.5
    return np.array([np.cos(ha), 0.0, 0.0, np.sin(ha)])
