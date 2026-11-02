"""ALLMIMIC current-reference actor and FSM tests; no motors started."""
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from xmigcs.common.robot_data import RobotData
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.stdin_keyboard_control import KeyboardController
from xmigcs.FSM.fsm_base import FSMStateName
from xmigcs.FSM.robot_fsm import RobotFSMImpl
from xmigcs.policy.allmimic.fsm_allmimic import FSMStateALLMIMIC, encode_reference_goal
from xmigcs.policy.whole_body_control.reference_window import reference_frame, stack_frames

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'src/xmigcs/policy/allmimic'
ORACLE = Path('/home/mig/dou/supermimic/tools/sim2sim.py')


def frame(i):
    q = Rotation.from_euler('xyz', [.03*i, -.02*i, .1*i]).as_quat()[[3, 0, 1, 2]]
    return reference_frame([.1*i, -.05*i, .9+.01*i], q, [1, 2, 3], [.1, -.2, .3],
                           np.arange(29)*.01+i*.05, np.arange(29)*.02+i*.01)


def oracle_class(joint_names):
    tree = ast.parse(ORACLE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PolicySpec')
    ns = {'dataclass': dataclass, 'np': np, 'robot': SimpleNamespace(JOINT_NAMES=joint_names)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ORACLE), 'exec'), ns)
    return ns['PolicySpec']


class AllMimicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def state(self, mode='npz', **overrides):
        config = yaml.safe_load((POLICY/'config/allmimic.yaml').read_text())
        config['onnx_path'] = str((POLICY/'model'/config['onnx_path']).resolve())
        config['motion_file'] = str((POLICY/'model'/config['motion_file']).resolve())
        config['ref_source'] = mode
        config['warn_on_missing_ref'] = False
        config.update(overrides)
        path = Path(self.tmp.name)/'state.yaml'
        path.write_text(yaml.safe_dump(config))
        data = RobotData()
        state = FSMStateALLMIMIC(data, str(path))
        data.q_a_[6:35] = state.joint_pos_array
        data.time_now_ = 10.
        state.on_enter()
        return data, state

    def test_current_reference_goal_matches_export_semantics(self):
        current = frame(4)
        future = stack_frames([frame(i) for i in range(5, 15)])
        actual = encode_reference_goal(current, future).reshape(10, 73)
        # Build independently with SciPy rotations to verify order and frame.
        r = current['rot']
        yaw = np.arctan2(r[1, 0], r[0, 0])
        h = Rotation.from_euler('z', yaw).as_matrix()
        for i in range(10):
            pos = h.T @ (future['pos'][i] - current['pos'])
            rel = h.T @ future['rot'][i]
            rot6 = np.r_[rel[:, 0], rel[:, 1]]
            v = future['rot'][i].T @ future['v'][i]
            w = future['rot'][i].T @ future['w'][i]
            expected = np.r_[pos, rot6, future['q'][i], future['qd'][i], v, w]
            np.testing.assert_allclose(actual[i], expected, atol=3e-7, rtol=1e-6)
        self.assertGreater(np.linalg.norm(actual[0, :3]), 0.)  # t+1, not first future as origin
        self.assertAlmostEqual(float(actual[0, 2]), float(future['pos'][0, 2]-current['pos'][2]), places=6)
        np.testing.assert_allclose(actual[:, 38:67], future['qd'])  # no 0.1 scale

    def test_current_reference_goal_invariant_to_global_yaw_and_translation(self):
        current = frame(4)
        future_frames = [frame(i) for i in range(5, 15)]
        original = encode_reference_goal(current, stack_frames(future_frames))
        global_rot = Rotation.from_euler('z', .83).as_matrix()
        translation = np.array([2., -3., 1.])
        def moved(f):
            r = global_rot @ f['rot']
            quat = Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]
            return reference_frame(global_rot @ f['pos'] + translation, quat,
                                   global_rot @ f['v'], global_rot @ f['w'], f['q'], f['qd'])
        transformed = encode_reference_goal(moved(current), stack_frames([moved(f) for f in future_frames]))
        np.testing.assert_allclose(transformed, original, atol=6e-7, rtol=1e-6)

    def test_real_model_npz_inputs_and_hybrid_residual_targets(self):
        data, state = self.state()
        self.assertEqual(Path(state.onnx_path).name, 'actor_0056000.onnx')
        self.assertEqual([(i.name, i.shape[-1]) for i in state.ort_session.get_inputs()],
                         [('obs_history', 930), ('goal', 730)])
        meta = json.loads(Path(state.onnx_path).with_suffix('.json').read_text())
        self.assertEqual(meta['task_config']['goal_pos'], 'current_ref')
        np.testing.assert_array_equal(state.action_scale_lab, np.array(meta['control']['action_scale_rad'], np.float32))
        rows = {r['joint']: r for r in meta['control']['actuators']}
        np.testing.assert_allclose(state.stiffness_array_seq, [rows[n]['stiffness'] for n in state.joint_seq])
        np.testing.assert_allclose(state.damping_array_seq, [rows[n]['damping'] for n in state.joint_seq])
        data.q_d_[35:] = [.12, -.23]
        for _ in range(3):
            state.inner_run()
        self.assertEqual(state.history.shape, (10, 93))
        np.testing.assert_array_equal(state.history, np.tile(state.obs, (10, 1)))
        current, future, _ = state.motion.window(2, state.goal_frames)
        np.testing.assert_allclose(state.future_goal, encode_reference_goal(current, future))
        direct = state.ort_session.run(['action'], {'obs_history': state.history.reshape(1, -1),
                                                     'goal': state.future_goal[None]})[0][0]
        np.testing.assert_allclose(state.action, direct, atol=1e-6)
        target = state.default_angles_lab + state.action_scale_lab*state.action_gain*state.action + state.reference_mask*(future['q'][0]-state.default_angles_lab)
        np.testing.assert_allclose(data.q_d_[6+state._motor_indices], target, atol=1e-7)
        np.testing.assert_array_equal(data.q_d_[35:], [.12, -.23])
        previous = state.action.copy()
        state.inner_run()
        np.testing.assert_array_equal(state.obs[-29:], previous)

    @unittest.skipUnless(ORACLE.is_file(), 'ALLMIMIC sim2sim source unavailable')
    def test_proprioception_and_residual_targets_match_training_sim2sim(self):
        data, state = self.state()
        data.imu_data_[:6] = [.3, -.1, .2, .1, -.2, .3]
        data.q_a_[6:35] += np.arange(29) * .001
        data.q_dot_a_[6:35] = np.arange(29) * .01
        state.inner_run(); state.inner_run()
        previous = np.arange(29, dtype=np.float32) * .005
        state.action = previous.copy()
        tree = ast.parse(ORACLE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Simulation')
        ns = {'np': np}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(ORACLE), 'exec'), ns)
        sim = object.__new__(ns['Simulation'])
        from xmigcs.policy.whole_body_control.reference_window import rotation
        robot_rot = rotation(data.get_robot_quat())
        sim.pose = lambda: (np.zeros(3), robot_rot)
        sim.spec = SimpleNamespace(q0=state.default_angles_lab)
        sim.qa = sim.qva = np.arange(29)
        sim.va = 26
        sim.data = SimpleNamespace(qpos=data.get_joint_pos()[state._motor_indices],
            qvel=np.r_[data.get_joint_vel()[state._motor_indices], data.get_angular_velocity()])
        sim.prev_action = previous
        expected = sim.observation()
        state.inner_run()
        np.testing.assert_allclose(state.obs, expected, atol=1e-7)
        spec = oracle_class(state.joint_seq)(state.joint_seq, state.default_angles_lab, state.action_scale_lab,
                                               action='res', res_legs='abs')
        rng = np.random.default_rng(314)
        for _ in range(8):
            action, next_q = rng.normal(size=(2,29)).astype(np.float32)
            target = (state.default_angles_lab + state.action_scale_lab*state.action_gain*action
                      + state.reference_mask*(next_q-state.default_angles_lab))
            np.testing.assert_allclose(target, spec.target(action, next_q), atol=2e-7)

    def test_gmr_window_and_current_reference_goal(self):
        data, state = self.state('gmr')
        state.inner_run()
        self.assertEqual(state.counter_step, 0)
        for i in range(12):
            f = frame(i)
            source_q, source_qd = np.empty(29), np.empty(29)
            source_q[state._gmr_perm], source_qd[state._gmr_perm] = f['q'], f['qd']
            data.ref_joint_pos, data.ref_joint_vel = source_q[None], source_qd[None]
            for attr, key in (('ref_body_pos_w', 'pos'), ('ref_body_quat_w', 'quat'),
                              ('ref_body_lin_vel_w', 'v'), ('ref_body_ang_vel_w', 'w')):
                setattr(data, attr, np.tile(f[key], (1, 24, 1)))
            data.ref_sample_time = 100+i*.02
            state._collect_gmr_reference()
        current, future, _ = state.gmr_window.window()
        np.testing.assert_allclose(current['q'], frame(1)['q'])
        np.testing.assert_allclose(future['q'], np.stack([frame(i)['q'] for i in range(2, 12)]))
        for _ in range(3):
            state.inner_run()
        np.testing.assert_allclose(state.future_goal, encode_reference_goal(current, future))
        self.assertTrue(np.isfinite(state.action).all())
        # Robot position has no influence on this model's goal.
        saved_goal = state.future_goal.copy()
        data.q_a_[:3] = [100., -200., 50.]
        data.imu_data_[:3] = [.4, -.2, .1]
        state.inner_run()
        np.testing.assert_array_equal(state.future_goal, saved_goal)
        state.on_exit()
        self.assertIsNone(state.gmr_window.window())

    def test_rejects_old_robot_relative_model(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported or mismatched'):
            self.state(onnx_path=str(POLICY/'model/actor_0016000.onnx'))

    @patch('xmigcs.FSM.robot_fsm.xlog')
    def test_o_key_registered_fsm_transitions(self, _log):
        cfg = yaml.safe_load((ROOT/'src/xmigcs/config/dex_config.yaml').read_text())
        fsm = RobotFSMImpl(RobotData(), cfg); fsm.first_enter()
        state = FSMStateName['ALLMIMIC']
        self.assertIsInstance(fsm.state_objects[state], FSMStateALLMIMIC)
        for origin in ('SWR', 'AMPMLP'):
            self.assertEqual(fsm.transitions['gotoALLMIMIC'][FSMStateName[origin]], (state,))
            fsm.current_state = fsm.target_state = FSMStateName[origin]
            fsm.run_fsm(ControlFlag('gotoALLMIMIC'))
            self.assertEqual(fsm.current_state, state)
            fsm.run_fsm(ControlFlag('gotoSTOP'))
            self.assertEqual(fsm.current_state, FSMStateName.STOP)
        keyboard = KeyboardController(); keyboard._print = lambda *_: None
        keyboard._process_key('o')
        self.assertEqual(keyboard.keyboard_flag.fsm_state_command, 'gotoALLMIMIC')


if __name__ == '__main__':
    unittest.main()
