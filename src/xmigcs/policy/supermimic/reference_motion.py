"""SuperMimic reference goals matching tools/sim2sim.py and MotionLib.goal."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _rotation_6d(matrix: np.ndarray, column_major: bool = False) -> np.ndarray:
    columns = matrix[..., :, :2]
    if column_major:
        columns = columns.swapaxes(-1, -2)
    return columns.reshape(matrix.shape[:-2] + (6,))


def _yaw_matrix(matrix: np.ndarray) -> np.ndarray:
    forward = matrix[:2, 0]
    norm = np.sqrt(np.dot(forward, forward) + 1e-9)
    cosine, sine = forward / norm
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Normalize wxyz and use the same float32 operations as native sim2sim."""
    quat = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if not np.isfinite(quat).all() or np.any(norm < 1e-12):
        raise ValueError("Reference contains a nonfinite or zero quaternion")
    w, x, y, z = np.moveaxis(quat / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1).reshape(quat.shape[:-1]+(3, 3))


def _strings(array):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(array).reshape(-1)]


class ReferenceMotion:
    """Read compact libraries/exports or prepared root_p/root_quat/q/qd clips."""

    def __init__(self, path: str, joint_names: list[str], goal_frames: int,
                 encoding: str = "reference_relative", clip_name: str | None = None):
        if encoding not in ("reference_relative", "raw_reference", "current_ref"):
            raise ValueError(f"Unsupported SuperMimic goal encoding: {encoding}")
        if goal_frames < 1:
            raise ValueError("goal_frames must be positive")
        self.encoding, self.goal_frames = encoding, goal_frames
        with np.load(path, allow_pickle=False) as data:
            if not np.isclose(float(data["fps"].item()), 50.0):
                raise ValueError(f"SuperMimic reference must be 50 Hz: {path}")
            names = _strings(data["joint_names"])
            if len(names) != len(joint_names) or set(names) != set(joint_names):
                raise ValueError(f"Reference joints do not match policy: {path}")
            perm = [names.index(name) for name in joint_names]
            if "robot" in data and str(data["robot"].item()) not in ("tiangong", "tiangong_evt2", "evt"):
                raise ValueError(f"Reference robot is not EVT2: {path}")
            if "root_pos" in data and "rest" in data:
                if "storage" in data and str(data["storage"].item()) != "compact_v1":
                    raise ValueError(f"Unsupported motion storage: {path}")
                root_p = data["root_pos"]
                rest = data["rest"].astype(np.float32)
                if rest.shape != (len(root_p), 68):
                    raise ValueError(f"Invalid compact motion shape: {path}")
                root_quat, q = rest[:, :4], rest[:, 4:33]
                root_v, root_w, qd = rest[:, 33:36], rest[:, 36:39], rest[:, 39:68]
            else:
                root_p, root_quat, root_v, root_w, q, qd = (
                    data[k] for k in ("root_p", "root_quat", "root_v", "root_w", "q", "qd"))
            total = len(root_p)
            metadata = {"starts", "lengths", "names"}
            if metadata <= set(data.files):
                starts, lengths, clips = data["starts"], data["lengths"], _strings(data["names"])
                if (lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer)
                        or len(lengths) == 0 or np.any(lengths < 2) or len(clips) != len(lengths)
                        or len(set(clips)) != len(clips) or int(lengths.sum()) != total
                        or not np.array_equal(starts, np.r_[0, np.cumsum(lengths)[:-1]])):
                    raise ValueError(f"Invalid motion clip boundaries: {path}")
                if clip_name is None and len(clips) != 1:
                    raise ValueError(f"Select motion_clip from {clips}: {path}")
                if clip_name is not None and clip_name not in clips:
                    raise ValueError(f"Unknown motion_clip {clip_name!r}: {path}")
                index = clips.index(clip_name) if clip_name is not None else 0
                start, length, self.name = int(starts[index]), int(lengths[index]), clips[index]
            elif metadata.intersection(data.files):
                raise ValueError(f"Incomplete clip metadata: {path}")
            else:
                start, length = 0, total
                self.name = str(data["name"].item()) if "name" in data else Path(path).stem
                if clip_name is not None and clip_name != self.name:
                    raise ValueError(f"Unknown motion_clip {clip_name!r}: {path}")
            values = (root_p, root_quat, root_v, root_w, q, qd)
            for key, value, width in zip(("root_p", "root_quat", "root_v", "root_w", "q", "qd"),
                                         values, (3, 4, 3, 3, len(names), len(names))):
                if value.shape != (total, width) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {key} in reference: {path}")
            sl = slice(start, start + length)
            self.root_p, root_quat, self.root_v, self.root_w, self.q, self.qd = (
                np.asarray(a[sl], dtype=np.float32).copy() for a in values)
        self.length = length
        if self.length < 2:
            raise ValueError(f"Reference needs at least two frames: {path}")
        self.q, self.qd = self.q[:, perm], self.qd[:, perm]
        self.root_R = _quaternion_matrix(root_quat)
        root_R_inv = self.root_R.transpose(0, 2, 1)
        self.root_v_local = np.einsum("nij,nj->ni", root_R_inv, self.root_v)
        self.root_w_local = np.einsum("nij,nj->ni", root_R_inv, self.root_w)

    def goal(self, frame: int) -> np.ndarray:
        """Future t+1..t+K; actor padding repeats the complete final frame."""
        if not 0 <= frame < self.length:
            raise IndexError(frame)
        requested = frame + np.arange(1, self.goal_frames + 1)
        future = np.minimum(requested, self.length - 1)
        recorded = (requested < self.length)[:, None]
        if self.encoding in ("reference_relative", "current_ref"):
            current_yaw_inv = _yaw_matrix(self.root_R[frame]).T
            position = (self.root_p[future] - self.root_p[frame]) @ current_yaw_inv.T
            orientation = _rotation_6d(current_yaw_inv @ self.root_R[future],
                                       column_major=self.encoding == "current_ref")
            linear_velocity = self.root_v_local[future]
            angular_velocity = self.root_w_local[future]
        else:
            position = self.root_p[future]
            orientation = _rotation_6d(self.root_R[future])
            linear_velocity, angular_velocity = self.root_v[future], self.root_w[future]
        if self.encoding == "current_ref":
            # Actor training clamps future indices, retaining recorded velocities
            # even in repeated final frames (unlike the legacy SuperMimic format).
            return np.concatenate((position, orientation, self.q[future], self.qd[future],
                                   linear_velocity, angular_velocity), axis=-1).reshape(-1).astype(np.float32)
        return np.concatenate((position, orientation, np.where(recorded, linear_velocity, 0.0),
                               np.where(recorded, angular_velocity, 0.0), self.q[future],
                               np.where(recorded, self.qd[future] * 0.1, 0.0)), axis=-1).reshape(-1).astype(np.float32)
