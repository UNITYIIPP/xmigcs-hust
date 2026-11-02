"""U SuperMimic encoding, real ONNX, and timestamped GMR tests; no motors started.

PYTHONPATH="src:$PYTHONPATH" .venv/bin/python -m unittest discover -s test -p test_whole_body_control.py -v
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from xmigcs.common.robot_data import RobotData
from xmigcs.common.robot_interface import RobotInterfaceImpl
from xmigcs.common.joystick import ControlFlag
from xmigcs.policy.whole_body_control.fsm_whole_body_control import FSMStateWHOLE_BODY_CONTROL
from xmigcs.policy.whole_body_control.reference_window import (
    DelayedReferenceWindow, NPZReference, encode_goal, reference_frame, rotation,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'src/xmigcs/policy/whole_body_control'
ORACLE_PATH = Path(os.environ.get('XMIGCS_SIM2SIM', '/home/mig/dou/supermimic_vibed/tools/sim2sim.py'))


def frame(index):
    quat = Rotation.from_euler('xyz', [.03*index, -.02*index, .07*index]).as_quat()[[3, 0, 1, 2]]
    return reference_frame(np.array([.1, -.03, .02])*index, quat,
                           [1, 2, 3], [-.2, .3, .4], np.arange(29)*.01+index, np.arange(29)*.02+1)


def load_oracle():
    spec = importlib.util.spec_from_file_location('wbc_sim2sim_oracle', ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    # Only pure NumPy/model routines are called; MuJoCo physics is not mocked or tested.
    try:
        import mujoco
    except ImportError:
        mujoco = SimpleNamespace()
    with patch.dict(sys.modules, {'wbc_sim2sim_oracle': module, 'mujoco': mujoco}):
        spec.loader.exec_module(module)
    return module


class WindowTests(unittest.TestCase):
    def test_progressive_fill_full_delay_and_duplicate_reads(self):
        window = DelayedReferenceWindow(10)
        for i in range(13):
            self.assertTrue(window.push(10+i*.02, frame(i)))
            current, future, mask = window.window()
            first = max(0, i-10)
            np.testing.assert_allclose(current['q'], frame(first)['q'])
            expected = [min(first+j, i) for j in range(1, 11)]
            np.testing.assert_allclose(future['q'], np.stack([frame(j)['q'] for j in expected]))
            np.testing.assert_allclose(future['v'], np.tile([1, 2, 3], (10, 1)))
            self.assertIsNone(mask)  # Startup copies velocities; not stationary NPZ end padding.
            snapshot = encode_goal(current, future).copy()
            self.assertFalse(window.push(10+i*.02, frame(999)))
            self.assertFalse(window.push(9., frame(999)))
            np.testing.assert_array_equal(encode_goal(*window.window()), snapshot)
        self.assertEqual(len(window.frames), 11)

    def test_resampling_100hz_irregular_arrivals_and_quaternion_sign(self):
        w = DelayedReferenceWindow(3)
        a, b = frame(0), frame(2)
        a['quat'][:] = [1, 0, 0, 0]; a['rot'] = rotation(a['quat'])
        b['quat'][:] = [-1, 0, 0, 0]; b['rot'] = rotation(b['quat'])
        w.push(1., a)
        w.push(1.01, a)
        self.assertEqual(len(w.frames), 1)
        w.push(1.03, b)
        self.assertEqual(len(w.frames), 2)
        np.testing.assert_allclose(w.frames[-1]['q'], (a['q']+b['q'])/2, atol=1e-6)
        np.testing.assert_allclose(w.frames[-1]['rot'], np.eye(3), atol=1e-7)
        w.push(1.04, b)
        self.assertEqual(len(w.frames), 3)

    def test_stall_freezes_recovery_does_not_interpolate_missing_seconds(self):
        w = DelayedReferenceWindow(3)
        for i in range(5):
            w.push(i*.02, frame(i))
        before = encode_goal(*w.window())
        for _ in range(100):
            w.push(.08, frame(99))
        np.testing.assert_array_equal(encode_goal(*w.window()), before)
        w.push(10., frame(5))
        np.testing.assert_allclose(w.window()[0]['q'], frame(2)['q'])
        np.testing.assert_allclose(w.window()[1]['q'][-1], frame(5)['q'])
        w.clear()
        self.assertIsNone(w.window())
        w.push(11., frame(6))
        np.testing.assert_allclose(w.window()[1]['q'], np.tile(frame(6)['q'], (3, 1)))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def make_state(self, mode='npz', **overrides):
        config = yaml.safe_load((POLICY/'config/whole_body_control.yaml').read_text())
        config['onnx_path'] = str(POLICY/'model'/config['onnx_path'])
        config['motion_file'] = str((POLICY/'model'/config['motion_file']).resolve())
        config['ref_source'] = mode
        config['warn_on_missing_ref'] = False
        config.update(overrides)
        path = Path(self.temp.name)/'config.yaml'
        path.write_text(yaml.safe_dump(config))
        data = RobotData()
        state = FSMStateWHOLE_BODY_CONTROL(data, str(path))
        data.q_a_[6:35] = state.joint_pos_array
        data.q_d_[35:] = [.12, -.23]
        data.joint_kp_p_[29:] = [1.2, 2.3]
        data.time_now_ = 13.457
        state.on_enter()
        return data, state

    def legacy_motion(self):
        """A nameless IsaacLab-format fixture independent of the selected user clip."""
        path = Path(self.temp.name) / 'legacy.npz'
        joint_pos = np.arange(6 * 29, dtype=np.float32).reshape(6, 29) / 100
        joint_vel = joint_pos * .1
        body_pos = np.zeros((6, 1, 3), dtype=np.float32)
        body_quat = np.tile([1., 0., 0., 0.], (6, 1, 1))
        body_vel = np.arange(18, dtype=np.float32).reshape(6, 1, 3) / 10
        np.savez(path, fps=np.array([50]), joint_pos=joint_pos, joint_vel=joint_vel,
                 body_pos_w=body_pos, body_quat_w=body_quat,
                 body_lin_vel_w=body_vel, body_ang_vel_w=body_vel)
        return path

    def publish_reference(self, data, state, index, timestamp):
        f = frame(index)
        # The public reference uses source order; the state must map by name.
        jp, jv = np.empty(29), np.empty(29)
        jp[state._gmr_perm], jv[state._gmr_perm] = f['q'], f['qd']
        data.ref_joint_pos, data.ref_joint_vel = jp[None], jv[None]
        for attr, key in (('ref_body_pos_w','pos'), ('ref_body_quat_w','quat'),
                          ('ref_body_lin_vel_w','v'), ('ref_body_ang_vel_w','w')):
            setattr(data, attr, np.tile(f[key], (1, 24, 1)))
        data.ref_sample_time = data.ref_received_time = timestamp

    def test_real_model_metadata_action_mapping_history_and_head_unchanged(self):
        data, state = self.make_state()
        meta = json.loads(state.ort_session.get_modelmeta().custom_metadata_map['supermimic_interface'])
        np.testing.assert_array_equal(state.action_scale_lab, np.array(meta['action_scales'], dtype=np.float32))
        np.testing.assert_array_equal(state.default_angles_lab, np.array(meta['default_joint_positions'], dtype=np.float32))
        for _ in range(2):
            state.inner_run()
        self.assertFalse(state._history_initialized)
        self.assertEqual(state._npz_frame, 2)
        state.inner_run()
        np.testing.assert_array_equal(state.history, np.tile(state.obs, (state.history_length, 1)))
        np.testing.assert_array_equal(state.obs[64:93], 0)
        np.testing.assert_allclose(data.q_d_[6+state._motor_indices],
                                   state.default_angles_lab+state.action_scale_lab*state.action)
        previous_action, previous_history = state.action.copy(), state.history.copy()
        data.q_dot_a_[6:35] = np.arange(29)*.01
        state.inner_run()
        np.testing.assert_array_equal(state.obs[64:93], previous_action)
        np.testing.assert_array_equal(state.history[:-1], previous_history[1:])
        np.testing.assert_array_equal(state.obs[29:58], data.get_joint_vel()[state._motor_indices])
        state.set_kp_kd()
        rows = {r['name']: r for r in meta['robot_config']['joints']}
        np.testing.assert_allclose(data.joint_kp_p_[state._motor_indices], [rows[n]['stiffness'] for n in state.joint_seq], rtol=1e-6)
        np.testing.assert_allclose(data.joint_kd_p_[state._motor_indices], [rows[n]['damping'] for n in state.joint_seq], rtol=1e-6)
        np.testing.assert_array_equal(data.q_d_[35:], [.12, -.23])
        np.testing.assert_array_equal(data.joint_kp_p_[29:], [1.2, 2.3])
        self.assertTrue((POLICY/'model/policy47500_50hz.onnx').is_file())
        self.assertFalse(Path(state.onnx_path).samefile(ROOT/'src/xmigcs/policy/supermimic/model/policydexevt.onnx'))

    def test_50hz_scheduler_keeps_original_non_inference_behavior(self):
        data, state = self.make_state()
        calls = []
        state.inner_run = lambda: calls.append(data.time_now_)
        for offset in (0, .01, .02, .02, .039, .04):
            data.time_now_ = state._enter_robot_time + offset
            state.run(ControlFlag())
        np.testing.assert_allclose(np.array(calls)-state._enter_robot_time, [0, .02, .04])
        data.q_d_[6:35] = 7
        data.time_now_ = state._enter_robot_time+.05
        state.run(ControlFlag())
        np.testing.assert_array_equal(data.q_d_[6:35], 7)

    def test_npz_mapping_and_end_padding_with_optional_truncation(self):
        motion_path = self.legacy_motion()
        _, state = self.make_state(motion_file=str(motion_path))
        cfg = yaml.safe_load((POLICY/'config/whole_body_control.yaml').read_text())
        with np.load(motion_path) as d:
            perm = [cfg['npz_joint_names'].index(n) for n in state.joint_seq]
            np.testing.assert_array_equal(state.motion.arrays['q'], d['joint_pos'][:, perm])
            np.testing.assert_array_equal(state.motion.arrays['v'], d['body_lin_vel_w'][:, 0])
        state.hold_final_reference, state.motion_length, state._npz_frame = True, 4, 2
        current, future, mask = state._reference_window()
        np.testing.assert_array_equal(mask, [True]+[False]*9)
        goal = encode_goal(current, future, mask).reshape(10, 73)
        np.testing.assert_allclose(goal[0, 44:], state.motion.arrays['qd'][3]*.1)
        np.testing.assert_array_equal(goal[1:, 44:], 0)
        np.testing.assert_array_equal(goal[1:, 9:15], 0)
        state._npz_frame = state.motion.length+100
        np.testing.assert_array_equal(state._reference_window()[0]['q'], state.motion.arrays['q'][3])
        state.hold_final_reference = False
        np.testing.assert_array_equal(state._reference_window()[0]['q'], state.motion.arrays['q'][-1])

    def test_gmr_state_startup_freeze_recovery_reentry_and_live_observations(self):
        data, state = self.make_state('gmr')
        data.q_d_[6:35] = .123
        state.inner_run()
        self.assertEqual(state.counter_step, 0)
        np.testing.assert_array_equal(data.q_d_[6:35], .123)
        for i in range(13):
            self.publish_reference(data, state, i, 10.+i*.02)
            state.inner_run()
            np.testing.assert_allclose(state.current_reference['q'], frame(max(0,i-10))['q'])
        self.assertTrue(state._history_initialized)
        goal, history = state.future_goal.copy(), state.history.copy()
        data.q_a_[6:35] += .1
        state.inner_run()
        np.testing.assert_array_equal(state.future_goal, goal)
        self.assertFalse(np.array_equal(state.history, history))
        state.on_exit()
        self.assertIsNone(state.gmr_window.window())
        self.assertFalse(state._history_initialized)
        state.on_enter()
        self.assertEqual(len(state.gmr_window.frames), 1)
        np.testing.assert_array_equal(state.action, 0)
        self.assertEqual(state._alignment_ticks, 0)

    def test_policy_and_reference_validation(self):
        with self.assertRaisesRegex(ValueError, '770'):
            self.make_state(onnx_path=str(POLICY/'model/policy47500_50hz.onnx'))
        with self.assertRaisesRegex(ValueError, '0.02'):
            self.make_state(physical_dt=.005, decimation=2)
        with self.assertRaisesRegex(ValueError, 'joint_names'):
            self.make_state(npz_joint_names=None, motion_file=str(self.legacy_motion()))
        with self.assertRaisesRegex(ValueError, 'joint_names'):
            self.make_state('gmr', gmr_joint_names=['bad']*29)

    def test_optional_torque_cap_and_nonidentity_motor_mapping(self):
        mapping = list(range(29))[::-1]
        data, state = self.make_state(tau_limit_enabled=True, tau_limit=[1.]*29, locked_joint_map=mapping)
        for _ in range(3):
            state.inner_run()
        q = data.get_joint_pos()[state._motor_indices]
        qd = data.get_joint_vel()[state._motor_indices]
        target = data.q_d_[6+state._motor_indices]
        tau = state.stiffness_array_seq*(target-q)-state.damping_array_seq*qd
        self.assertLessEqual(float(np.abs(tau).max()), 1.0001)
        state.set_kp_kd()
        np.testing.assert_array_equal(data.joint_kp_p_[state._motor_indices], state.stiffness_array_seq)

    @unittest.skipUnless(ORACLE_PATH.is_file(), 'external sim2sim oracle unavailable')
    def test_numpy_observation_goal_and_real_actions_match_sim2sim(self):
        oracle = load_oracle()
        data, state = self.make_state()
        spec, act = oracle.load_policy(state.onnx_path)
        cfg = yaml.safe_load((POLICY/'config/whole_body_control.yaml').read_text())
        clip = oracle.catalog([str((POLICY/'model'/cfg['motion_file']).resolve())])[0]
        motion = oracle.Motion(clip, spec, cfg['npz_joint_names'])
        for i in (0, 2, 127, motion.length-2, motion.length-1):
            a, b, mask = state.motion.window(i, state.goal_frames)
            expected = spec.goal(motion.frame(i), motion.frame(i+np.arange(1, 11)), i+np.arange(1, 11)<motion.length)
            np.testing.assert_allclose(encode_goal(a, b, mask), expected, rtol=1e-6, atol=2e-6)
        state.inner_run(); state.inner_run()
        state.init_to_world = np.eye(3)
        data.imu_data_[:6] = [.3, -.1, .2, .1, -.2, .3]
        data.q_dot_a_[6:35] = np.arange(29)*.01
        previous = np.arange(29, dtype=np.float32)*.005
        state.action = previous.copy()
        robot_rot = rotation(data.get_robot_quat())
        simulation = object.__new__(oracle.Simulation)
        simulation.spec, simulation.motion, simulation.frame = spec, motion, 2
        simulation.qa = np.arange(29)
        simulation.qva = np.arange(29)
        simulation.va = 26  # qvel[29:32] supplies the local angular velocity.
        simulation.data = SimpleNamespace(qpos=data.get_joint_pos()[state._motor_indices],
            qvel=np.r_[data.get_joint_vel()[state._motor_indices], data.get_angular_velocity()])
        simulation.pose = lambda: (np.zeros(3), robot_rot)
        simulation.prev_action = previous
        expected_obs = simulation.observation()
        state.inner_run()
        np.testing.assert_allclose(state.obs, expected_obs, atol=2e-6, rtol=1e-6)
        np.testing.assert_allclose(state.action, act(state.history, state.future_goal), atol=2e-6, rtol=1e-6)

    def test_raw_imu_mount_transform(self):
        data, state = self.make_state(imu_frame='imu_waist_link')
        data.imu_data_[:6] = [.3, -.1, .2, .1, -.2, .3]
        for _ in range(3):
            state.inner_run()
        pelvis = rotation(data.get_robot_quat()) @ state._sensor_in_pelvis.T
        np.testing.assert_allclose(state.obs[58:61], state._sensor_in_pelvis @ data.get_angular_velocity())
        np.testing.assert_allclose(state.obs[61:64], pelvis.T @ [0, 0, -1], atol=1e-7)


class InterfaceTimestampTests(unittest.TestCase):
    @patch('xmigcs.common.robot_interface.xlog')
    def test_timestamp_tracks_interpolated_sample_then_freezes(self, _log):
        data = RobotData()
        interface = RobotInterfaceImpl(data, str(ROOT/'src/xmigcs/config/dex_config.yaml'))
        payload = np.zeros(232, dtype=np.float32)
        payload[72:168] = np.tile([1., 0, 0, 0], 24)
        for t in (10., 10.02):
            with patch('xmigcs.common.robot_interface.time.monotonic', return_value=t):
                interface._gmr_callback(SimpleNamespace(data=payload))
                interface.get_realtime_gmr_data()
        self.assertAlmostEqual(data.ref_sample_time, 10.008)
        self.assertAlmostEqual(data.ref_received_time, 10.02)
        for t in (10.04, 11., 12.):
            with patch('xmigcs.common.robot_interface.time.monotonic', return_value=t):
                interface.get_realtime_gmr_data()
            self.assertAlmostEqual(data.ref_sample_time, 10.02)
            self.assertAlmostEqual(data.ref_received_time, 10.02)


if __name__ == '__main__':
    unittest.main()
