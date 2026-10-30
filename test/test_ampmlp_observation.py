import json
import logging
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

# Keep this unit test independent from the ROS runtime used on the robot.
sensor_msgs = types.ModuleType("sensor_msgs")
sensor_msgs.msg = types.ModuleType("sensor_msgs.msg")
sensor_msgs.msg.Joy = type("Joy", (), {})
sys.modules.setdefault("sensor_msgs", sensor_msgs)
sys.modules.setdefault("sensor_msgs.msg", sensor_msgs.msg)

yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda stream: {}
sys.modules.setdefault("yaml", yaml_stub)

logging_utils = types.ModuleType("xmigcs.utils.logging_utils")
logging_utils.get_logger = logging.getLogger
sys.modules.setdefault("xmigcs.utils.logging_utils", logging_utils)
xlog_utils = types.ModuleType("xmigcs.utils.xlog_utils")
xlog_utils.xlog = lambda *args, **kwargs: None
sys.modules.setdefault("xmigcs.utils.xlog_utils", xlog_utils)

ort_stub = types.ModuleType("onnxruntime")
ort_stub.SessionOptions = type("SessionOptions", (), {})
ort_stub.GraphOptimizationLevel = type("GraphOptimizationLevel", (), {"ORT_ENABLE_ALL": 1})
ort_stub.InferenceSession = lambda *args, **kwargs: None
sys.modules.setdefault("onnxruntime", ort_stub)


def _euler_to_quaternion(roll, pitch, yaw, degrees=False, out_xyzw=False):
    if degrees:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    q = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)
    return q[[1, 2, 3, 0]] if out_xyzw else q


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _single_axis_quat(angle, axis, degrees=False):
    if degrees:
        angle = np.radians(angle)
    half = angle / 2.0
    values = {
        "x": [np.cos(half), np.sin(half), 0.0, 0.0],
        "y": [np.cos(half), 0.0, np.sin(half), 0.0],
        "z": [np.cos(half), 0.0, 0.0, np.sin(half)],
    }
    return np.asarray(values[axis])


def _matrix_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_rotate_inverse(q_xyzw, value):
    q_w = q_xyzw[3]
    q_v = q_xyzw[:3]
    return value * (2.0 * q_w * q_w - 1.0) - np.cross(q_v, value) * (2.0 * q_w) + q_v * (2.0 * np.dot(q_v, value))


rotation = types.ModuleType("xmigcs.common.rotation")
rotation.euler_to_quaternion_scipy = _euler_to_quaternion
rotation.quat_mul = _quat_mul
rotation.euler_single_axis_to_quat = _single_axis_quat
rotation.matrix_from_quat = _matrix_from_quat
rotation.quat_rotate_inverse_numpy = _quat_rotate_inverse
sys.modules.setdefault("xmigcs.common.rotation", rotation)

joint_mapping = types.ModuleType("xmigcs.common.joint_mapping")
joint_mapping.MJ_JOINTS_31 = [f"joint_{index}" for index in range(31)]
sys.modules.setdefault("xmigcs.common.joint_mapping", joint_mapping)

from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData
from xmigcs.policy import onnx_policy_base
from xmigcs.policy.ampmlp.fsm_ampmlp import FSMStateAMPMLP


BASE_LAYOUT = [
    {"name": "base_ang_vel", "size": 3},
    {"name": "projected_gravity", "size": 3},
    {"name": "velocity_commands", "size": 3},
    {"name": "joint_pos", "size": 23},
    {"name": "joint_vel", "size": 23},
    {"name": "actions", "size": 23},
]


class _ConcretePolicy(onnx_policy_base.ONNXPolicyBase):
    def compute_observation(self, flag):
        pass

    def _apply_action_to_robot(self):
        pass

    def _apply_pd_gains(self):
        pass


class _FakeInput:
    name = "obs"
    shape = [None, 81]


class _WarmupSession:
    def __init__(self, *args, **kwargs):
        self.feed_shape = None

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, outputs, feed):
        self.feed_shape = feed["obs"].shape
        if self.feed_shape != (1, 81):
            raise ValueError(f"expected (1, 81), got {self.feed_shape}")
        return [np.zeros((1, 23), dtype=np.float32)]


class _MetadataSession:
    def __init__(self, metadata, input_dim):
        self._metadata = metadata
        self._input = type("Input", (), {"shape": [None, input_dim]})()

    def get_modelmeta(self):
        return type("Meta", (), {"custom_metadata_map": self._metadata})()

    def get_inputs(self):
        return [self._input]


class ONNXWarmupTests(unittest.TestCase):
    def test_warmup_uses_static_model_input_dimension(self):
        policy = _ConcretePolicy.__new__(_ConcretePolicy)
        policy.obs_size_ = 78
        policy.num_hist_ = 1

        with patch.object(onnx_policy_base.ort, "InferenceSession", _WarmupSession):
            policy._init_onnx_session(".", "policy.onnx")

        self.assertIsNotNone(policy.ort_session_)
        self.assertEqual(policy.ort_session_.feed_shape, (1, 81))


class AMPObservationTests(unittest.TestCase):
    def _new_policy(self, layout, input_dim, metadata=None):
        policy = FSMStateAMPMLP.__new__(FSMStateAMPMLP)
        policy.joint_seq = [f"joint_{index}" for index in range(23)]
        observation_metadata = (
            {"observation_layout": json.dumps(layout)}
            if layout is not None
            else {"observation_names": ",".join(term["name"] for term in BASE_LAYOUT)}
        )
        policy.ort_session_ = _MetadataSession(
            {
                **observation_metadata,
                "amp_walk_run_speed_threshold": "1.5",
                "amp_walking_uprightness": "0.7",
                **(metadata or {}),
            },
            input_dim,
        )
        policy._configure_observation_layout()
        policy.action_num_ = 23
        policy.clip_obs_ = 100.0
        policy._init_buffers()
        policy._init_term_history_buffers()
        return policy

    def test_old_layout_keeps_78_dim_observation(self):
        policy = self._new_policy(None, 78)

        self.assertEqual(policy.obs_size_, 78)
        self.assertEqual(policy.num_hist_, 1)
        self.assertEqual([term["name"] for term in policy._obs_layout], [term["name"] for term in BASE_LAYOUT])

    def test_upright_onehot_layout_adds_walk_run_fall_condition(self):
        policy = self._new_policy(BASE_LAYOUT + [{"name": "amp_condition", "size": 3}], 81)
        policy.robot_data_ = RobotData()
        policy.lab2mj = np.arange(23)
        policy.default_angles_lab = np.zeros(23, dtype=np.float32)
        policy.dof_pos_scale_ = 1.0
        policy.dof_vel_scale_ = 1.0
        policy.ang_vel_scale_ = 1.0
        policy.last_actions_ = np.zeros(23, dtype=np.float32)
        policy._anchor_waist_indices = (12, 13, 14)

        flag = ControlFlag()
        flag.x_speed_command = 1.6
        flag.y_speed_command = 0.0
        flag.yaw_speed_command = 0.0
        policy.compute_observation(flag)
        np.testing.assert_array_equal(policy.observations_[-3:], [0.0, 1.0, 0.0])

        policy.robot_data_.imu_data_[2] = np.pi / 2.0
        policy.compute_observation(flag)
        np.testing.assert_array_equal(policy.observations_[-3:], [0.0, 0.0, 1.0])

    def test_history_is_grouped_by_term_like_isaaclab(self):
        policy = self._new_policy(BASE_LAYOUT, 390)
        self.assertEqual(policy.num_hist_, 5)
        self.assertEqual([buf.shape for buf in policy._term_hist_bufs], [(5, term["size"]) for term in BASE_LAYOUT])


if __name__ == "__main__":
    unittest.main()
