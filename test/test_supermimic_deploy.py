"""Run separately from legacy tests that replace ONNX/ROS modules with stubs.

PYTHONPATH=src python -m unittest discover -s test -p test_supermimic_deploy.py -v
No ROS nodes or motor interfaces are started.
"""
import importlib.util
import copy
import json
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
LEGACY_MODEL = str(POLICY / 'model/policydexevt.onnx')


class SuperMimicDeployTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = RobotData()
        cls.state = FSMStateSUPERMIMIC(cls.data, model_path=LEGACY_MODEL)

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
        state = FSMStateSUPERMIMIC(rd, config_path=str(config_path), model_path=LEGACY_MODEL,
                                  motion_path=str(self.path))
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
        self.assertEqual(Path(self.state.model_path).name, 'policydexevt.onnx')
        with patch('xmigcs.policy.supermimic.fsm_supermimic.json.load',
                   side_effect=AssertionError('legacy model must not read a JSON sidecar')):
            state = FSMStateSUPERMIMIC(RobotData(), model_path=LEGACY_MODEL)
        self.assertEqual(state.goal_encoding_, 'reference_relative')
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


class ActorDeployTest(unittest.TestCase):
    """Verify the real actor against independently assembled inputs and targets."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.meta = json.loads((POLICY / 'model/actor_0100000.json').read_text())
        self.names = self.meta['control']['joint_names']
        t = np.arange(14, dtype=np.float32)
        self.pos = np.c_[.07*t, -.02*t, 1.+.005*t].astype(np.float32)
        self.rot = Rotation.from_euler('xyz', np.c_[.02*t, -.03*t, .15+.06*t]).as_matrix()
        quat = Rotation.from_matrix(self.rot).as_quat()[:, [3, 0, 1, 2]]
        self.q = (np.asarray(self.meta['control']['default_joint_positions_rad'])
                  + .01*t[:, None] + np.arange(29)*.002).astype(np.float32)
        self.qd = np.broadcast_to(np.arange(29)*.015+.1, self.q.shape).astype(np.float32).copy()
        self.v = np.tile([.2, -.1, .04], (len(t), 1)).astype(np.float32)
        self.w = np.tile([.03, -.04, .1], (len(t), 1)).astype(np.float32)
        self.path = Path(self.temp.name) / 'reference.npz'
        # Reverse reference joint order so the real state also exercises name mapping.
        np.savez(self.path, root_p=self.pos, root_quat=quat, q=self.q[:, ::-1],
                 qd=self.qd[:, ::-1], root_v=self.v, root_w=self.w,
                 fps=50., joint_names=np.array(self.names[::-1]), robot='evt')
        self.data = RobotData()
        self.state = FSMStateSUPERMIMIC(self.data, motion_path=str(self.path))

    def oracle_goal(self, frame):
        indices = np.minimum(frame + np.arange(1, 11), len(self.q)-1)
        yaw = np.arctan2(self.rot[frame, 1, 0], self.rot[frame, 0, 0])
        h = Rotation.from_euler('z', yaw).as_matrix()
        relative = h.T @ self.rot[indices]
        rot6 = np.concatenate((relative[:, :, 0], relative[:, :, 1]), axis=1)
        velocity = np.einsum('nji,nj->ni', self.rot[indices], self.v[indices])
        omega = np.einsum('nji,nj->ni', self.rot[indices], self.w[indices])
        return np.c_[(self.pos[indices]-self.pos[frame]) @ h, rot6, self.q[indices],
                     self.qd[indices], velocity, omega].astype(np.float32).reshape(1, 730)

    def test_current_ref_goal_matches_actor_contract_including_end_padding(self):
        for frame in (0, 4, 12, 13):
            with self.subTest(frame=frame):
                np.testing.assert_allclose(self.state.motion_.goal(frame).reshape(1, 730),
                                           self.oracle_goal(frame), atol=5e-7)
        # Actor training repeats the full last frame, including nonzero velocities.
        final_goal = self.state.motion_.goal(13).reshape(10, 73)
        np.testing.assert_array_equal(final_goal[:, 38:67], np.tile(self.qd[-1], (10, 1)))

    def test_real_actor_history_actions_targets_timing_stop_and_reentry(self):
        s, rd = self.state, self.data
        self.assertEqual(Path(s.model_path).name, 'actor_0100000.onnx')
        self.assertEqual(s._obs_input_shape, (1, 990))
        self.assertEqual(s.goal_encoding_, 'current_ref')
        rd.q_a_[6+s.lab2mj] = self.q[0]
        # Initial yaw differs from the reference, so alignment must be applied.
        initial_rpy = np.array([.03, -.02, .7])
        rd.imu_data_[:3] = initial_rpy[::-1]
        s.on_enter()
        alignment = Rotation.from_euler('z', initial_rpy[2]-.15).as_matrix()
        previous_action = np.zeros(29, dtype=np.float32)
        history = None
        controls = self.meta['control']
        default = np.asarray(controls['default_joint_positions_rad'], dtype=np.float32)
        scale = np.asarray(controls['action_scale_rad'], dtype=np.float32)
        gain = np.asarray(controls['policy_action_gain'], dtype=np.float32)
        mask = np.asarray(controls['reference_mask'], dtype=np.float32)
        for frame in range(len(self.q)-1):
            rpy = initial_rpy + np.array([.005, -.003, .01])*frame
            robot_rot = Rotation.from_euler('xyz', rpy).as_matrix()
            rd.imu_data_[:3] = rpy[::-1]
            rd.imu_data_[3:6] = [.1, -.2, .3]
            rd.q_a_[6+s.lab2mj] = self.q[frame] + .003
            rd.q_dot_a_[6+s.lab2mj] = self.qd[frame] - .01
            rel = robot_rot.T @ alignment @ self.rot[frame]
            obs = np.r_[self.q[frame]+.003-default, self.qd[frame]-.01,
                        [.1, -.2, .3], -robot_rot[2], previous_action, rel[:, 0], rel[:, 1]].astype(np.float32)
            history = (np.tile(obs, (10, 1)) if history is None
                       else np.vstack((history[1:], obs)))
            goal = self.oracle_goal(frame)
            expected_action, = s.ort_session_.run(['action'], {'obs_history': history.reshape(1, 990), 'goal': goal})
            target = default + scale*gain*expected_action[0] + mask*(self.q[frame+1]-default)
            flag = ControlFlag()
            with patch.object(s, 'compute_actions', wraps=s.compute_actions) as infer:
                for tick in range(4):
                    rd.q_d_[:] = 99
                    rd.joint_kp_p_[:] = 0
                    s.run(flag)
                    self.assertEqual(s._frame, frame+1)
                    self.assertEqual(infer.call_count, 1)
                    np.testing.assert_allclose(rd.q_d_[6+s.lab2mj], target, atol=3e-5)
                    np.testing.assert_array_equal(rd.q_d_[35:], 0)
                    np.testing.assert_array_equal(rd.joint_kp_p_[29:], 0)
            np.testing.assert_allclose(s.observations_.reshape(10, 99), history, atol=6e-7)
            np.testing.assert_allclose(s.actions_, expected_action[0], atol=1e-4, rtol=1e-5)
            # Use the actual previous raw action for the next independent observation;
            # this prevents floating-point differences from accumulating in the oracle.
            previous_action[:] = s.actions_
            if frame == len(self.q)-2:
                self.assertTrue(s._finished)
                self.assertEqual(flag.fsm_state_command, 'gotoSTOP')
        rows = {r['joint']: r for r in controls['actuators']}
        np.testing.assert_array_equal(rd.joint_kp_p_[s.lab2mj],
                                      np.array([rows[n]['stiffness'] for n in s.joint_seq], dtype=np.float32))
        np.testing.assert_array_equal(rd.joint_kd_p_[s.lab2mj],
                                      np.array([rows[n]['damping'] for n in s.joint_seq], dtype=np.float32))
        s.on_exit()
        s.on_enter()
        self.assertFalse(s._finished)
        self.assertEqual(s._frame, 0)
        np.testing.assert_array_equal(s.last_actions_, 0)
        s.run(ControlFlag())
        np.testing.assert_array_equal(s.observations_.reshape(10, 99),
                                      np.tile(s.observations_[:99], (10, 1)))

    def test_default_actor_and_packaged_reference_run_offline(self):
        rd = RobotData()
        state = FSMStateSUPERMIMIC(rd)
        self.assertEqual(state.motion_.name, 'walk1_subject1')
        rd.q_a_[6+state.lab2mj] = state.motion_.q[0]
        rd.q_dot_a_[6+state.lab2mj] = state.motion_.qd[0]
        state.on_enter()
        for _ in range(12):
            state.run(ControlFlag())
        self.assertEqual(state._frame, 3)
        self.assertTrue(np.isfinite(rd.q_d_).all())
        self.assertTrue(np.isfinite(state.actions_).all())

    def test_actor_metadata_rejects_incompatible_contracts(self):
        def replace_value(spec, path, value):
            for key in path[:-1]:
                spec = spec[key]
            spec[path[-1]] = value

        cases = [
            (['model_file'], 'other.onnx'),
            (['history_frames'], 9),
            (['future_frame_offsets'], list(range(2, 12))),
            (['task_config', 'actor_ref_orientation'], False),
            (['task_config', 'goal_pos'], 'robot'),
            (['inputs', 'obs_history', 'reference_orientation', 'rotation_6d_order'],
             ['R00', 'R01', 'R10', 'R11', 'R20', 'R21']),
            (['control', 'reference_mask'], [0]*29),
            (['control', 'policy_action_gain'], [2.]*29),
            (['control', 'actuators'], self.meta['control']['actuators'][:-1]),
        ]
        for path, value in cases:
            with self.subTest(path=path):
                spec = copy.deepcopy(self.meta)
                replace_value(spec, path, value)
                with patch('xmigcs.policy.supermimic.fsm_supermimic.json.load', return_value=spec):
                    with self.assertRaisesRegex(ValueError, 'actor metadata'):
                        self.state._configure_actor_policy()

    def test_missing_actor_sidecar_is_rejected(self):
        with patch('xmigcs.policy.supermimic.fsm_supermimic.Path.open', side_effect=FileNotFoundError('missing')):
            with self.assertRaisesRegex(ValueError, 'actor metadata'):
                self.state._configure_actor_policy()


if __name__ == '__main__':
    unittest.main()
