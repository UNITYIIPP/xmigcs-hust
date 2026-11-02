"""Reference-relative SuperMimic goals and delayed live reference windows."""

from collections import deque

import numpy as np


def rotation(quaternion):
    """Normalized wxyz quaternion(s), local to world."""
    q = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if q.shape[-1] != 4 or not np.isfinite(q).all() or np.any(norm < 1e-12):
        raise ValueError("Reference requires finite, nonzero wxyz quaternions")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def heading(rot):
    c, s = rot[:2, 0] / np.sqrt(np.dot(rot[:2, 0], rot[:2, 0]) + 1e-9)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def joint_permutation(source_names, policy_names):
    if source_names is None:
        raise ValueError("Reference has no joint_names; configure its source joint order explicitly")
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in source_names]
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(policy_names):
        raise ValueError("Reference joint_names must match the 29 policy joints")
    return np.array([names.index(n) for n in policy_names])


def reference_frame(pos, quat, v, w, q, qd):
    frame = {key: np.asarray(value, dtype=np.float32).copy()
             for key, value in zip(('pos', 'quat', 'v', 'w', 'q', 'qd'), (pos, quat, v, w, q, qd))}
    for key, width in (('pos', 3), ('quat', 4), ('v', 3), ('w', 3), ('q', 29), ('qd', 29)):
        if frame[key].shape != (width,) or not np.isfinite(frame[key]).all():
            raise ValueError(f"Invalid reference {key}: expected finite ({width},)")
    frame['rot'] = rotation(frame['quat'])
    frame['quat'] /= np.linalg.norm(frame['quat'])
    return frame


def stack_frames(frames):
    return {key: np.stack([f[key] for f in frames]) for key in frames[0]}


def encode_goal(current, future, recorded=None):
    """Match sim2sim.PolicySpec.goal; only NPZ end padding zeros velocities."""
    h = heading(current['rot'])
    dp = (future['pos'] - current['pos']) @ h
    rot6 = (h.T @ future['rot'])[:, :, :2].reshape(-1, 6)
    v = np.einsum('kji,kj->ki', future['rot'], future['v'])
    w = np.einsum('kji,kj->ki', future['rot'], future['w'])
    mask = np.ones((len(dp), 1), dtype=bool) if recorded is None else np.asarray(recorded)[:, None]
    return np.concatenate((dp, rot6, np.where(mask, v, 0), np.where(mask, w, 0),
                           future['q'], np.where(mask, future['qd'] * .1, 0)), axis=-1).reshape(-1).astype(np.float32)


class NPZReference:
    """Keep old IsaacLab files intact; also accept sim2sim compact/prepared files."""

    def __init__(self, path, policy_names, fallback_names=None, anchor_index=0, clip_name=None):
        with np.load(path, allow_pickle=False) as d:
            if not np.isclose(float(d['fps'].item()), 50.):
                raise ValueError("NPZ reference must be 50 Hz; resample before deployment")
            perm = joint_permutation(d['joint_names'] if 'joint_names' in d else fallback_names, policy_names)
            if 'root_pos' in d and 'rest' in d:
                rest = d['rest']
                if rest.ndim != 2 or rest.shape[1] != 68:
                    raise ValueError("compact_v1 rest must have width 68")
                pos, quat, q = d['root_pos'], rest[:, :4], rest[:, 4:33]
                v, w, qd = rest[:, 33:36], rest[:, 36:39], rest[:, 39:]
            elif 'qpos' in d and 'qvel' in d:
                qp, qv = d['qpos'], d['qvel']
                if qp.ndim != 2 or qp.shape[1] != 36 or qv.shape != (len(qp), 35):
                    raise ValueError("Expected qpos [N,36] and qvel [N,35]")
                pos, quat, q, v, w, qd = qp[:, :3], qp[:, 3:7], qp[:, 7:], qv[:, :3], qv[:, 3:6], qv[:, 6:]
            elif all(k in d for k in ('root_p', 'root_quat', 'q', 'root_v', 'root_w', 'qd')):
                pos, quat, q, v, w, qd = (d[k] for k in ('root_p', 'root_quat', 'q', 'root_v', 'root_w', 'qd'))
            else:
                pos, quat, v, w = (d[k][:, anchor_index] for k in
                                   ('body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'))
                q, qd = d['joint_pos'], d['joint_vel']
            total = len(pos)
            start, length = 0, total
            clip_keys = {'starts', 'lengths', 'names'}
            if clip_keys.intersection(d.files):
                if not clip_keys.issubset(d.files):
                    raise ValueError("Incomplete NPZ clip metadata")
                names = [n.decode() if isinstance(n, bytes) else str(n) for n in d['names']]
                lengths, starts = d['lengths'], d['starts']
                if (lengths.ndim != 1 or len(names) != len(lengths) or len(set(names)) != len(names)
                        or not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths < 1)
                        or sum(lengths) != total or not np.array_equal(starts, np.r_[0, np.cumsum(lengths)[:-1]])):
                    raise ValueError("Invalid NPZ clip boundaries")
                if clip_name is None and len(names) != 1:
                    raise ValueError("Select motion_clip for a multi-clip NPZ")
                i = names.index(clip_name) if clip_name is not None else 0
                start, length = int(starts[i]), int(lengths[i])
            self.arrays = {}
            for key, value, width in zip(('pos', 'quat', 'v', 'w', 'q', 'qd'),
                                         (pos, quat, v, w, q, qd), (3, 4, 3, 3, 29, 29)):
                if value.shape != (total, width) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid NPZ {key}")
                self.arrays[key] = np.asarray(value[start:start+length], dtype=np.float32).copy()
        if length < 1:
            raise ValueError("Empty NPZ reference")
        self.length = length
        self.arrays['q'] = self.arrays['q'][:, perm]
        self.arrays['qd'] = self.arrays['qd'][:, perm]
        self.arrays['rot'] = rotation(self.arrays['quat'])
        self.arrays['quat'] /= np.linalg.norm(self.arrays['quat'], axis=-1, keepdims=True)

    def window(self, index, count, end=None):
        end = self.length if end is None else min(self.length, end)
        if end < 1:
            raise ValueError("Effective motion length must be positive")
        index = min(max(int(index), 0), end - 1)
        requested = index + np.arange(1, count + 1)
        future_index = np.minimum(requested, end - 1)
        current = {k: v[index] for k, v in self.arrays.items()}
        future = {k: v[future_index] for k, v in self.arrays.items()}
        return current, future, requested < end


class DelayedReferenceWindow:
    """Resample received references at policy dt, retaining current + K future frames.

    Startup repeats the latest sample into missing future slots. No clock-driven
    extrapolation: duplicate/out-of-order timestamps do not advance playback.
    A long interruption freezes the window; recovery appends one new sample and
    restarts the sampling clock, without synthesizing a trajectory across the gap.
    """

    def __init__(self, count, dt=.02, gap_reset_s=.1):
        if count < 1 or not np.isfinite(dt) or dt <= 0 or not np.isfinite(gap_reset_s) or gap_reset_s < dt:
            raise ValueError("Invalid GMR window timing")
        self.count, self.dt, self.gap_reset_s = count, dt, gap_reset_s
        self.clear()

    def clear(self):
        self.frames = deque(maxlen=self.count + 1)
        self.last_time = None
        self.last_frame = None
        self.next_time = None

    def push(self, timestamp, frame):
        t = float(timestamp)
        if not np.isfinite(t) or (self.last_time is not None and t <= self.last_time + 1e-9):
            return False
        if self.last_time is None or t - self.last_time > self.gap_reset_s:
            self.frames.append({k: v.copy() for k, v in frame.items()})
            self.next_time = t + self.dt
        else:
            while self.next_time <= t + 1e-9:
                alpha = float(np.clip((self.next_time - self.last_time) / (t - self.last_time), 0., 1.))
                a, b = self.last_frame, frame
                values = {k: (1-alpha)*a[k] + alpha*b[k] for k in ('pos', 'v', 'w', 'q', 'qd')}
                # Quaternion sign continuity is needed for shortest-path nlerp.
                quat_b = b['quat'] if np.dot(a['quat'], b['quat']) >= 0 else -b['quat']
                values['quat'] = (1-alpha)*a['quat'] + alpha*quat_b
                self.frames.append(reference_frame(**values))
                self.next_time += self.dt
        self.last_time = t
        self.last_frame = {k: v.copy() for k, v in frame.items()}
        return True

    def window(self):
        if not self.frames:
            return None
        frames = list(self.frames)
        future = frames[1:] + [frames[-1]] * (self.count + 1 - len(frames))
        return frames[0], stack_frames(future), None
