"""Run separately from legacy tests that replace ONNX/ROS modules with stubs.

PYTHONPATH=src python -m unittest discover -s test -p test_supermimic_deploy.py -v
No ROS nodes or motor interfaces are started.
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    from sensor_msgs.msg import Joy
except ImportError:
    sensor_msgs = types.ModuleType('sensor_msgs')
    sensor_msgs.msg = types.ModuleType('sensor_msgs.msg')
    sensor_msgs.msg.Joy = type('Joy', (), {})
    sys.modules['sensor_msgs'] = sensor_msgs
    sys.modules['sensor_msgs.msg'] = sensor_msgs.msg

from xmigcs.common.robot_data import RobotData
from xmigcs.common.joystick import ControlFlag
from xmigcs.policy.supermimic.fsm_supermimic import FSMStateSUPERMIMIC
from xmigcs.policy.supermimic.reference_motion import ReferenceMotion

POLICY = Path(__file__).resolve().parents[1] / 'src/xmigcs/policy/supermimic'


class SuperMimicDeployTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = RobotData()
        cls.state = FSMStateSUPERMIMIC(cls.data)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'clip.npz'
        # Existing observation/terminal tests exercise one policy tick per call.
        for name, value in (('dt_', 0.02), ('decimation_', 1)):
            setting = patch.object(self.state, name, value)
            setting.start()
            self.addCleanup(setting.stop)
        s = self.state
        n = 4
        t = np.arange(n, dtype=np.float32)
        quat = Rotation.from_euler('xyz', np.c_[t * 0.04, t * -0.03, t * 0.2]).as_quat()[:, [3, 0, 1, 2]]
        rest = np.zeros((n, 68), dtype=np.float16)
        rest[:, :4] = quat
        rest[:, 4:33] = s.default_angles_lab + t[:, None] * 0.01
        rest[:, 33:36] = [0.2, -0.1, 0.3]
        rest[:, 36:39] = [0.1, 0.2, -0.3]
        rest[:, 39:] = np.arange(29) * 0.1 + 0.25
        self.fixture = dict(root_pos=np.c_[t * 0.1, t * -0.03, 1 + t * 0.01], rest=rest,
                            fps=np.array(50.), joint_names=np.array(s.joint_seq))
        np.savez(self.path, **self.fixture)
        self.reference = ReferenceMotion(str(self.path), s.joint_seq, 10)

    def configured_state(self, dt, decimation, warm_start_time=0.0):
        cfg = yaml.safe_load((POLICY / 'config/supermimic.yaml').read_text())
        cfg['dt'] = dt
        cfg['control'].update(decimation=decimation, warm_start_time=warm_start_time)
        config_path = Path(self.temp.name) / 'config.yaml'
        config_path.write_text(yaml.safe_dump(cfg))
        rd = RobotData()
        state = FSMStateSUPERMIMIC(rd, config_path=str(config_path), motion_path=str(self.path))
        rd.q_a_[6+state.lab2mj] = state.default_angles_lab
        rd.q_dot_a_[6+state.lab2mj] = 0
        rd.imu_data_[:6] = 0
        # Scheduling must not depend on the global clock's phase or precision.
        rd.time_now_ = 123.457
        state.on_enter()
        return state, rd

    def test_decimation_preserves_policy_history_actions_and_reference_timing(self):
        expected = []
        for dt, decimation in ((0.02, 1), (0.01, 2), (0.0025, 8)):
            with self.subTest(dt=dt, decimation=decimation):
                s, rd = self.configured_state(dt, decimation)
                with patch.object(s, 'compute_actions', wraps=s.compute_actions) as infer:
                    for tick in range(2 * decimation):
                        # Every tick must rewrite held commands and apply PD/fixed joints.
                        rd.q_d_[:] = 99
                        rd.joint_kp_p_[:] = 0
                        s.run(ControlFlag())
                        frame = tick // decimation
                        self.assertEqual(s._frame, frame + 1)
                        self.assertEqual(infer.call_count, frame + 1)
                        actual = (s.observations_.copy(), s.future_goal_.copy(),
                                  s.last_actions_.copy(), rd.q_d_.copy())
                        if decimation == 1:
                            expected.append(actual)
                        for value, baseline in zip(actual, expected[frame]):
                            np.testing.assert_allclose(value, baseline, atol=1e-6)
                        np.testing.assert_array_equal(rd.joint_kp_p_[s.lab2mj],
                                                      s.stiffness_array_seq)
                    flag = ControlFlag()
                    s.run(flag)
                    self.assertTrue(s._finished)
                    self.assertEqual(flag.fsm_state_command, 'gotoSTOP')
                    last = s._last_joint_target.copy()
                    for _ in range(decimation):
                        rd.q_d_[:] = 99
                        s.run(ControlFlag())
                        np.testing.assert_array_equal(rd.q_d_[6+s.lab2mj], last)
                    self.assertEqual(infer.call_count, 3)
                    s.on_enter()
                    s.run(ControlFlag())
                    self.assertEqual(s._frame, 1)
                    self.assertEqual(infer.call_count, 4)

    def test_decimated_warm_start_uses_fsm_tick_duration(self):
        s, rd = self.configured_state(0.01, 2, warm_start_time=0.04)
        self.assertEqual(s._warm_start_steps, 4)
        start = s._warm_start_pose.copy()
        with patch.object(s, 'compute_actions', wraps=s.compute_actions) as infer:
            for tick in range(4):
                s.run(ControlFlag())
                np.testing.assert_allclose(rd.q_d_[6+s.lab2mj],
                                           start + (tick+1)/4 * (s.motion_.q[0]-start), atol=1e-6)
                self.assertEqual(s._frame, 0)
                self.assertTrue(s.is_first_obs_)
            infer.assert_not_called()
            s.run(ControlFlag())
            s.run(ControlFlag())
            self.assertEqual(infer.call_count, 1)
            self.assertEqual(s._frame, 1)

    def test_invalid_policy_period_or_decimation_is_rejected(self):
        for dt, decimation in ((0.01, 1), (0.02, 2), (0, 1), (-0.02, 1),
                               (float('nan'), 1), (float('inf'), 1),
                               (0.02, 0), (0.02, -1), (0.01, 2.0), (0.02, True)):
            with self.subTest(dt=dt, decimation=decimation):
                with self.assertRaisesRegex(ValueError, 'dt \\* decimation = 0.02 s'):
                    self.configured_state(dt, decimation)

    def test_loads_exact_onnx_without_json_and_keeps_original_model(self):
        self.assertEqual(self.state._obs_input_name, 'observation_history')
        self.assertEqual(self.state.goal_frames_, 10)
        self.assertEqual(Path(self.state.model_path).name, 'policy_sim2sim_018000.onnx')
        self.assertFalse(Path(self.state.model_path + '.json').exists())
        self.assertTrue((POLICY / 'model/policy.onnx').is_file())

    def test_compact_prepared_and_reordered_joints_match(self):
        r = self.reference
        prepared = Path(self.temp.name) / 'prepared.npz'
        data = self.fixture
        rest = data['rest'].astype(np.float32)
        np.savez(prepared, root_p=data['root_pos'], root_quat=rest[:, :4], q=rest[:, 4:33],
                 root_v=rest[:, 33:36], root_w=rest[:, 36:39], qd=rest[:, 39:],
                 fps=[50.], joint_names=data['joint_names'])
        p = ReferenceMotion(str(prepared), self.state.joint_seq, 10)
        reverse = data['rest'].copy()
        reverse[:, 4:33] = reverse[:, 4:33][:, ::-1]
        reverse[:, 39:] = reverse[:, 39:][:, ::-1]
        np.savez(self.path, **dict(data, rest=reverse, joint_names=data['joint_names'][::-1]))
        q = ReferenceMotion(str(self.path), self.state.joint_seq, 10)
        for f in range(4):
            np.testing.assert_array_equal(r.goal(f), p.goal(f))
            np.testing.assert_array_equal(r.goal(f), q.goal(f))

    def test_recorded_last_velocity_is_kept_only_padding_is_zero(self):
        r = self.reference
        goal = r.goal(2).reshape(10, 73)
        np.testing.assert_allclose(goal[0, 9:12], r.root_v_local[3])
        np.testing.assert_allclose(goal[0, 44:], r.qd[3] * 0.1)
        np.testing.assert_array_equal(goal[1:, 9:15], 0)
        np.testing.assert_array_equal(goal[1:, 44:], 0)
        np.testing.assert_array_equal(r.goal(3).reshape(10, 73)[:, 9:15], 0)

    def test_zero_quaternion_and_partial_library_are_rejected(self):
        data = self.fixture.copy()
        data['rest'] = data['rest'].copy()
        data['rest'][0, :4] = 0
        np.savez(self.path, **data)
        with self.assertRaisesRegex(ValueError, 'quaternion'):
            ReferenceMotion(str(self.path), self.state.joint_seq, 10)
        np.savez(self.path, **dict(self.fixture, starts=np.array([0])))
        with self.assertRaisesRegex(ValueError, 'metadata'):
            ReferenceMotion(str(self.path), self.state.joint_seq, 10)

    def test_warm_start_history_final_stop_and_reentry(self):
        s, rd = self.state, self.data
        original = s.motion_, s._warm_start_steps, s._align_heading
        try:
            s.motion_, s._warm_start_steps, s._align_heading = self.reference, 2, False
            rd.q_a_[6+s.lab2mj] = s.default_angles_lab
            rd.q_dot_a_[6+s.lab2mj] = 0
            rd.imu_data_[:6] = 0
            s.on_enter()
            start = s._warm_start_pose.copy()
            for k in range(2):
                s.run(ControlFlag())
                np.testing.assert_allclose(rd.q_d_[6+s.lab2mj],
                                           start + (k+1)/2 * (s.motion_.q[0]-start), atol=1e-6)
                self.assertEqual(s._frame, 0)
                self.assertTrue(s.is_first_obs_)
            s.run(ControlFlag())
            obs = s.observations_.reshape(10, 99)
            np.testing.assert_allclose(obs, np.tile(obs[0], (10, 1)))
            self.assertEqual(s._frame, 1)
            s._frame = 2
            flag = ControlFlag()
            s.run(flag)
            self.assertTrue(s._finished)
            self.assertEqual(flag.fsm_state_command, 'gotoSTOP')
            last = s._last_joint_target.copy()
            rd.q_d_[:] = 99
            with patch.object(s, 'compute_actions') as infer:
                s.run(ControlFlag())
            infer.assert_not_called()
            np.testing.assert_array_equal(rd.q_d_[6+s.lab2mj], last)
            s.on_enter()
            self.assertFalse(s._finished)
            self.assertEqual(s._frame, 0)
            self.assertTrue(s.is_first_obs_)
            np.testing.assert_array_equal(s.last_actions_, 0)
        finally:
            s.motion_, s._warm_start_steps, s._align_heading = original

    def test_sensor_rotation_maps_orientation_and_gyro_to_pelvis(self):
        s, rd = self.state, self.data
        original = s._sensor_in_pelvis.copy()
        try:
            mount = Rotation.from_euler('xyz', [-3.1416, 0, -1.5695]).as_matrix().astype(np.float32)
            pelvis = Rotation.from_euler('xyz', [0.2, -0.1, 0.5]).as_matrix()
            sensor = pelvis @ mount
            rpy = Rotation.from_matrix(sensor).as_euler('xyz')
            rd.imu_data_[:3] = rpy[::-1]
            omega = np.array([0.2, -0.4, 0.3])
            rd.imu_data_[3:6] = mount.T @ omega
            s._sensor_in_pelvis = mount
            actual_r, actual_w = s._pelvis_rotation_and_gyro()
            np.testing.assert_allclose(actual_r, pelvis, atol=4e-7)
            np.testing.assert_allclose(actual_w, omega, atol=1e-7)
        finally:
            s._sensor_in_pelvis = original


if __name__ == '__main__':
    unittest.main()
