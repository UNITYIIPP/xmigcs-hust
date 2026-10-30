"""Source-deployment migration tests. No motor interfaces are initialized.

PYTHONPATH="src:$PYTHONPATH" .venv/bin/python -m unittest discover -s test -p test_teleop_yibo.py -v
Enable the reference-only ROS subprocess test with XMIGCS_ROS_TEST=1 in an isolated ROS domain.
"""
import importlib.util
import itertools
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from xmigcs.common.gmr_shared_receiver import GMRSharedReceiver
from xmigcs.common.joystick import ControlFlag
from xmigcs.common.robot_data import RobotData
from xmigcs.common.robot_interface import RobotInterfaceImpl
from xmigcs.common.stdin_keyboard_control import KeyboardController
from xmigcs.FSM.fsm_base import FSMStateName
from xmigcs.FSM.robot_fsm import RobotFSMImpl
from xmigcs.policy.teleop_yibo.fsm_teleop_yibo import FSMStateTELEOP_YIBO
from xmigcs.rl_control_node import XMIGCSControlNode


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'src/xmigcs/policy/teleop_yibo'
CONFIG = ROOT / 'src/xmigcs/config/dex_config.yaml'
SOURCE = Path(os.environ.get('XMIGCS_SOURCE_DEPLOY', '/home/mig/universal_control/xmigcs-deploy'))
SOURCE_POLICY = SOURCE / 'src/xmigcs/policy/teleop_yibo/fsm_teleop_yibo.py'


def payload(offset=0.0):
    data = np.zeros(232, dtype=np.float32)
    data[:72] = offset
    data[72:168] = np.tile([1., 0., 0., 0.], 24)
    data[168:197] = np.arange(29) * 0.01 + offset
    data[197:226] = np.arange(29) * -0.02 - offset
    data[226:] = np.arange(6) + offset
    return data


class PolicyTests(unittest.TestCase):
    policy_dir = POLICY
    policy_class = FSMStateTELEOP_YIBO
    config_filename = 'teleop_yibo.yaml'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def make_state(self, mode='npz', cls=None, **overrides):
        cls = cls or self.policy_class
        config = yaml.safe_load((self.policy_dir / 'config' / self.config_filename).read_text())
        config['onnx_path'] = str(self.policy_dir / 'model' / config['onnx_path'])
        config['motion_file'] = str((self.policy_dir / 'model' / config['motion_file']).resolve())
        config['ref_source'] = mode
        config.update(overrides)
        path = Path(self.temp.name) / 'policy.yaml'
        path.write_text(yaml.safe_dump(config))
        data = RobotData()
        state = cls(data, str(path))
        data.q_a_[6:35] = state.joint_pos_array
        data.q_d_[35:] = [0.12, -0.23]
        data.joint_kp_p_[29:] = [1.2, 2.3]
        data.time_now_ = 13.457
        state.on_enter()
        return data, state

    def test_default_assets_model_metadata_and_head_mapping(self):
        data, state = self.make_state()
        self.assertEqual(state.ort_session.get_inputs()[0].shape, [1, 770])
        self.assertEqual(state.ort_session.get_outputs()[0].shape, [1, 29])
        metadata = state.ort_session.get_modelmeta().custom_metadata_map
        scales = np.array([float(x) for x in metadata['action_scale'].split(',')])
        np.testing.assert_array_equal(state.action_scale_lab, scales)
        state.inner_run()
        state.inner_run()
        self.assertFalse(state._history_initialized)
        state.inner_run()
        expected = (state.action * scales + state.default_angles_lab).reshape(-1)
        np.testing.assert_allclose(data.q_d_[6 + state.mj2lab], expected)
        state.set_kp_kd()
        np.testing.assert_array_equal(data.joint_kp_p_[:29], state.stiffness_array)
        np.testing.assert_array_equal(data.joint_kd_p_[:29], state.damping_array)
        np.testing.assert_array_equal(data.q_d_[35:], [0.12, -0.23])
        np.testing.assert_array_equal(data.joint_kp_p_[29:], [1.2, 2.3])

    def test_history_is_grouped_by_term_and_uses_previous_raw_action(self):
        data, state = self.make_state()
        state.inner_run()
        state.inner_run()
        observations = []
        real_session = state.ort_session
        def infer(outputs, feed):
            observations.append(feed[state.input_name[0]].copy())
            return real_session.run(outputs, feed)
        state.ort_session = SimpleNamespace(run=infer)
        state.inner_run()
        first = observations[-1][0]
        cmd = np.r_[state.ref_joint_pos[0], state.ref_joint_vel[0]]
        np.testing.assert_allclose(first[:290].reshape(5, 58), np.tile(cmd, (5, 1)))
        np.testing.assert_array_equal(first[-145:], 0)
        previous_action = state.action.copy()
        data.q_a_[6:35] += np.arange(29) * 0.001
        state.inner_run()
        second = observations[-1][0]
        np.testing.assert_array_equal(second[:232], first[58:290])
        np.testing.assert_allclose(second[-29:], previous_action[0])
        np.testing.assert_allclose(second[335:480].reshape(5, 29)[-1],
                                   data.get_joint_pos()[state.mj2lab] - state.default_angles_lab)

    def test_scheduler_and_non_inference_ticks_match_source(self):
        data, state = self.make_state()
        calls = []
        state.inner_run = lambda: calls.append(data.time_now_)
        for offset in (0, .01, .02, .02, .039, .04):
            data.time_now_ = state._enter_robot_time + offset
            state.run(ControlFlag())
        np.testing.assert_allclose(np.array(calls) - state._enter_robot_time, [0, .02, .04])
        data.q_d_[6:35] = 7  # Source does not rewrite cached targets on held ticks.
        data.time_now_ = state._enter_robot_time + .05
        state.run(ControlFlag())
        np.testing.assert_array_equal(data.q_d_[6:35], 7)

    def test_missing_and_stalled_reference_and_reentry_follow_source(self):
        data, state = self.make_state('gmr')
        data.q_d_[6:35] = .123
        state.inner_run()
        self.assertEqual(state.counter_step, 0)
        np.testing.assert_array_equal(data.q_d_[6:35], .123)
        interface = RobotInterfaceImpl(data, str(CONFIG))
        interface._publish_gmr_reference(interface._decode_gmr_frame(payload()))
        for _ in range(4):
            state.inner_run()
        self.assertEqual(state.counter_step, 4)
        self.assertTrue(state._sync_ref_from_robot_data())
        state.on_exit()
        state.on_enter()
        self.assertEqual(state.counter_step, 0)
        self.assertFalse(state._history_initialized)
        self.assertTrue(state._ref_ever_received)

    def test_npz_completion_and_optional_truncation_match_source(self):
        _, state = self.make_state()
        state.counter_step = state._npz_num_frames + 10
        state._advance_npz_reference()
        np.testing.assert_array_equal(state.ref_joint_pos[0], state._npz_joint_pos[-1])
        state.hold_final_reference = True
        state.motion_length = 3
        state._advance_npz_reference()
        np.testing.assert_array_equal(state.ref_joint_pos[0], state._npz_joint_pos[2])

    @unittest.skipUnless(SOURCE_POLICY.is_file(), 'source deployment unavailable')
    def test_source_and_migrated_policy_observations_actions_commands_match(self):
        spec = importlib.util.spec_from_file_location('source_teleop_oracle', SOURCE_POLICY)
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        for mode in ('gmr', 'npz'):
            with self.subTest(mode=mode):
                a, migrated = self.make_state(mode)
                b, original = self.make_state(mode, source.FSMStateTELEOP_YIBO)
                sessions = []
                for state in (migrated, original):
                    spy = Mock(wraps=state.ort_session)
                    state.ort_session = spy
                    sessions.append(spy)
                for tick in range(60):
                    for data, state in ((a, migrated), (b, original)):
                        data.time_now_ = state._enter_robot_time + tick * .01
                        data.q_a_[6:35] = state.joint_pos_array + .01 * np.sin(tick + np.arange(29))
                        data.q_dot_a_[6:35] = .02 * np.cos(tick + np.arange(29))
                        data.imu_data_[:6] = [.01*tick, .02, -.01, .1, -.2, .3]
                        if mode == 'gmr' and tick < 40:
                            frame = payload(.001*tick)
                            data.ref_joint_pos = frame[168:197].reshape(1, 29)
                            data.ref_joint_vel = frame[197:226].reshape(1, 29)
                            data.ref_body_pos_w = frame[:72].reshape(1, 24, 3)
                            data.ref_body_quat_w = frame[72:168].reshape(1, 24, 4)
                        state.run(ControlFlag())
                    for attr in ('q_d_', 'q_dot_d_', 'tau_d_', 'joint_kp_p_', 'joint_kd_p_'):
                        np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
                    np.testing.assert_array_equal(migrated.action, original.action)
                    self.assertEqual(migrated.counter_step, original.counter_step)
                self.assertEqual(sessions[0].run.call_count, sessions[1].run.call_count)
                for left, right in zip(sessions[0].run.call_args_list, sessions[1].run.call_args_list):
                    np.testing.assert_array_equal(left.args[1]['obs'], right.args[1]['obs'])


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        log = patch('xmigcs.common.robot_interface.xlog')
        log.start()
        self.addCleanup(log.stop)
        self.data = RobotData()
        self.interface = RobotInterfaceImpl(self.data, str(CONFIG))

    def test_gmr_queue_interpolation_and_cached_frame(self):
        interface = self.interface
        with patch('xmigcs.common.robot_interface.time.monotonic', return_value=10.):
            interface._gmr_callback(SimpleNamespace(data=payload()))
            interface.get_realtime_gmr_data()
        np.testing.assert_array_equal(self.data.ref_joint_pos[0], payload()[168:197])
        with patch('xmigcs.common.robot_interface.time.monotonic', return_value=10.02):
            interface._gmr_callback(SimpleNamespace(data=payload(1)))
            interface.get_realtime_gmr_data()
        # (10.020 - .012 - 10.000) / .020 = .4
        np.testing.assert_allclose(self.data.ref_joint_pos[0], payload(.4)[168:197], atol=1e-6)
        np.testing.assert_allclose(self.data.ref_body_quat_w[0, 0], [1, 0, 0, 0])
        with patch('xmigcs.common.robot_interface.time.monotonic', return_value=11.):
            interface.get_realtime_gmr_data()
        np.testing.assert_array_equal(self.data.ref_joint_pos[0], payload(1)[168:197])
        interface._gmr_callback(SimpleNamespace(data=[1, 2]))
        self.assertTrue(interface.queue_realtime_gmr_state.empty())
        for index in range(40):
            interface._gmr_callback(SimpleNamespace(data=payload(index)))
        self.assertEqual(interface.queue_realtime_gmr_state.qsize(), 32)
        interface.get_realtime_gmr_data()
        self.assertTrue(interface.queue_realtime_gmr_state.empty())

    def test_shared_latest_slot_is_consumed_once(self):
        receiver = GMRSharedReceiver(232, 182)
        self.interface._gmr_shared_receiver = receiver
        with receiver._lock:
            receiver._gmr_view[:] = payload(.2)
            receiver._val_gmr_cnt.value = 1
            receiver._val_gmr_t.value = time.monotonic()
        self.interface.get_realtime_gmr_data()
        np.testing.assert_array_equal(self.data.ref_joint_pos[0], payload(.2)[168:197])
        self.assertIsNone(receiver.read_latest().gmr_data)

    def test_receiver_start_failure_falls_back_and_shutdown_stops_receiver(self):
        interface = self.interface
        interface._gmr_shared_enabled = True
        interface._gmr_shared_cfg = {}
        with patch('xmigcs.common.robot_interface.GMRSharedReceiver') as receiver:
            receiver.return_value.start.return_value = False
            interface._init_gmr_shared_receiver()
            receiver.return_value.stop.assert_called_once()
        self.assertFalse(interface._gmr_shared_enabled)
        self.assertIsNone(interface._gmr_shared_receiver)
        interface.node = Mock()
        interface.control_tool = None
        interface._init_ros_highlevel_interfaces()
        self.assertTrue(any(c.args[1] == '/gmr_info'
                            for c in interface.node.create_subscription.call_args_list))
        receiver = Mock()
        interface._gmr_shared_receiver = receiver
        interface.destroy_node()
        receiver.stop.assert_called_once()

    def prepare_sender(self, mode):
        interface = self.interface
        interface.initialized = True
        interface.sim = True
        interface.comm_mode = mode
        interface.interpolate_action = Mock()
        interface.backup_serial_cmd = Mock()
        interface.convert_to_motor_commands = Mock()
        interface.publish_motor_commands = Mock()
        interface._send_motor_commands = Mock()
        interface.publish_robot_cmd = Mock()
        return interface

    def test_sync_i_and_async_other_states_handoff_for_ros_and_sdk(self):
        for mode, state_name in itertools.product(
                ('ros2', 'ros2_bridge', 'lpc', 'shm_rpc', 'tcp_rpc'),
                ('TELEOP_YIBO', 'WHOLE_BODY_CONTROL')):
            with self.subTest(mode=mode, state=state_name):
                interface = self.prepare_sender(mode)
                sender = interface.publish_motor_commands if mode.startswith('ros2') else interface._send_motor_commands
                interface.current_state = SimpleNamespace(name='SUPERMIMIC')
                interface.send_motor_commands()
                sender.assert_not_called()
                interface._publish_latest_command()
                self.assertEqual(sender.call_count, 1)
                interface.current_state = SimpleNamespace(name=state_name)
                self.data.q_d_[6] = .2
                interface.send_motor_commands()
                self.assertEqual(sender.call_count, 2)
                np.testing.assert_equal(sender.call_args.args[0]['q_d'][6], .2)
                interface._publish_latest_command()
                interface._publish_background_robot_cmd()
                self.assertEqual(sender.call_count, 2)
                interface.publish_robot_cmd.assert_called_once()
                interface.current_state = SimpleNamespace(name='STOP')
                interface._publish_latest_command()  # Must not replay the old I frame.
                self.assertEqual(sender.call_count, 2)
                interface.send_motor_commands()
                interface._publish_latest_command()
                self.assertEqual(sender.call_count, 3)
                self.assertEqual(sender.call_args.args[0]['state'].name, 'STOP')

    def test_background_sender_cannot_duplicate_inflight_sync_frame(self):
        interface = self.prepare_sender('ros2')
        interface.current_state = SimpleNamespace(name='TELEOP_YIBO')
        entered, release = threading.Event(), threading.Event()
        def publish(frame):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test release missing')
        interface.publish_motor_commands.side_effect = publish
        sync = threading.Thread(target=interface.send_motor_commands)
        background = threading.Thread(target=interface._publish_latest_command)
        sync.start()
        try:
            self.assertTrue(entered.wait(2))
            background.start()
        finally:
            release.set()
            sync.join(2)
            if background.ident is not None:
                background.join(2)
        self.assertFalse(sync.is_alive())
        self.assertFalse(background.is_alive())
        interface.publish_motor_commands.assert_called_once()


class FSMTests(unittest.TestCase):
    @patch('xmigcs.rl_control_node.xlog')
    def test_control_loop_samples_reference_before_fsm_and_feedback_after_send(self, _log):
        events = []
        interface = Mock()
        interface.process_controller_data.side_effect = lambda: events.append('input')
        interface.get_realtime_gmr_data.side_effect = lambda: events.append('reference')
        interface.update_param.side_effect = lambda **_: events.append('state')
        interface.wait_for_initial_motor_state.return_value = True
        fsm = Mock()
        fsm.get_current_state.return_value = SimpleNamespace(name='TELEOP_YIBO')
        fsm.run_fsm.side_effect = lambda _: events.append('fsm')
        node = SimpleNamespace(
            robot_interface=interface, robot_fsm=fsm, control_running=True, sim=True,
            control_stats={'cycles': 0, 'last_cycle_s': 0., 'max_cycle_s': 0.,
                           'cycle_durations_s': [], 'deadline_misses': 0},
            _get_policy_tick_period_s=lambda: .01,
            _update_robot_data=lambda _: events.append('feedback'),
            _send_control_commands=lambda: events.append('send'),
        )
        waits = []
        def wait(duration):
            waits.append(duration)
            if len(waits) == 3:
                node.control_running = False
        node.control_stop_event = SimpleNamespace(wait=wait)
        with patch('xmigcs.rl_control_node.rclpy.ok', return_value=True), \
             patch('xmigcs.rl_control_node.time.perf_counter',
                   side_effect=(1000 + i * .0001 for i in range(100))):
            XMIGCSControlNode._rl_control_loop_impl(node)
        self.assertEqual(events, ['input', 'feedback'] +
                         ['input', 'reference', 'fsm', 'state', 'send', 'feedback'] * 3)
        self.assertEqual(node.control_stats['cycles'], 3)
        self.assertTrue(all(0 < duration <= .01 for duration in waits))

    @patch('xmigcs.FSM.robot_fsm.xlog')
    def test_all_states_load_and_i_v_transitions_remain_independent(self, _log):
        config = yaml.safe_load(CONFIG.read_text())
        data = RobotData()
        fsm = RobotFSMImpl(data, config)
        fsm.first_enter()
        teleop = FSMStateName['TELEOP_YIBO']
        supermimic = FSMStateName['SUPERMIMIC']
        self.assertIsInstance(fsm.state_objects[teleop], FSMStateTELEOP_YIBO)
        self.assertEqual(fsm.state_objects[supermimic].__class__.__name__, 'FSMStateSUPERMIMIC')
        self.assertNotIn(FSMStateName['STANDZERO'], fsm.transitions['gotoTELEOP_YIBO'])
        self.assertIn(FSMStateName['STANDZERO'], fsm.transitions['gotoSUPERMIMIC'])
        for state_name, origin in itertools.product(
                ('TELEOP_YIBO', 'WHOLE_BODY_CONTROL'), ('AMPMLP', 'SWR')):
            with self.subTest(state=state_name, origin=origin):
                fsm.current_state = fsm.target_state = FSMStateName[origin]
                fsm.run_fsm(ControlFlag('goto'+state_name))
                self.assertEqual(fsm.current_state, FSMStateName[state_name])
                fsm.run_fsm(ControlFlag('gotoSTOP'))
                self.assertEqual(fsm.current_state, FSMStateName.STOP)
        whole_body = FSMStateName['WHOLE_BODY_CONTROL']
        self.assertEqual(fsm.state_objects[whole_body].__class__.__name__, 'FSMStateWHOLE_BODY_CONTROL')
        self.assertIsNot(fsm.state_objects[whole_body], fsm.state_objects[teleop])
        # Use the real period resolver without starting a ROS control node.
        node = SimpleNamespace(robot_fsm=fsm, _default_non_policy_tick_period_s=.01)
        fsm.current_state = teleop
        self.assertEqual(XMIGCSControlNode._get_policy_tick_period_s(node), .01)
        fsm.current_state = whole_body
        self.assertEqual(XMIGCSControlNode._get_policy_tick_period_s(node), .01)
        fsm.current_state = supermimic
        self.assertEqual(XMIGCSControlNode._get_policy_tick_period_s(node),
                         fsm.state_objects[supermimic].dt_)
        self.assertAlmostEqual(fsm.state_objects[supermimic].dt_ *
                               fsm.state_objects[supermimic].decimation_, .02)
        keyboard = KeyboardController()
        keyboard._print = lambda *_: None
        for key, event in [('i', 'gotoTELEOP_YIBO'), ('u', 'gotoWHOLE_BODY_CONTROL'),
                           ('v', 'gotoSUPERMIMIC')]:
            keyboard._process_key(key)
            self.assertEqual(keyboard.keyboard_flag.fsm_state_command, event)


@unittest.skipUnless(os.environ.get('XMIGCS_ROS_TEST') == '1', 'reference-only ROS test is opt-in')
class ROSReceiverTests(unittest.TestCase):
    def test_subprocess_receives_reference_and_stops(self):
        import rclpy
        from std_msgs.msg import Float32MultiArray
        topic = f'/xmigcs_teleop_test_{os.getpid()}/gmr_info'
        receiver = GMRSharedReceiver(232, 182, gmr_topic=topic, gmr_6kp_topic=topic+'_6kp')
        self.addCleanup(receiver.stop)
        self.assertTrue(receiver.start())
        context = rclpy.Context()
        rclpy.init(context=context)
        self.addCleanup(context.shutdown)
        node = rclpy.create_node('teleop_reference_test', context=context)
        self.addCleanup(node.destroy_node)
        executor = rclpy.executors.SingleThreadedExecutor(context=context)
        executor.add_node(node)
        self.addCleanup(executor.shutdown)
        publisher = node.create_publisher(Float32MultiArray, topic, 10)
        deadline = time.monotonic() + 5
        snapshot = None
        expected = payload(.12)
        while time.monotonic() < deadline:
            publisher.publish(Float32MultiArray(data=expected.tolist()))
            executor.spin_once(timeout_sec=.02)
            snapshot = receiver.read_latest()
            if snapshot.gmr_data is not None:
                break
        self.assertIsNotNone(snapshot.gmr_data)
        np.testing.assert_array_equal(snapshot.gmr_data, expected)
        process = receiver._proc
        receiver.stop()
        self.assertFalse(process.is_alive())


if __name__ == '__main__':
    unittest.main()
