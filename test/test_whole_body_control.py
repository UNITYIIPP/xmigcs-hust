"""Exercise the independent U policy with the same behavioral tests as I.

PYTHONPATH="src:$PYTHONPATH" .venv/bin/python -m unittest discover -s test -p test_whole_body_control.py -v
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

import test_teleop_yibo as baseline
from xmigcs.common.robot_data import RobotData
from xmigcs.policy.whole_body_control.fsm_whole_body_control import FSMStateWHOLE_BODY_CONTROL


class WholeBodyPolicyTests(baseline.PolicyTests):
    policy_dir = baseline.ROOT / 'src/xmigcs/policy/whole_body_control'
    policy_class = FSMStateWHOLE_BODY_CONTROL
    config_filename = 'whole_body_control.yaml'

    def test_default_load_is_independent_of_i_files_and_buffers(self):
        real_open = open
        def independent_open(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve().is_relative_to(baseline.POLICY):
                raise AssertionError(f'U must not read I resources: {path}')
            return real_open(path, *args, **kwargs)
        with patch('builtins.open', side_effect=independent_open):
            state = self.policy_class(RobotData())
        config = yaml.safe_load((self.policy_dir / 'config' / self.config_filename).read_text())
        self.assertEqual(Path(state.onnx_path).parent, self.policy_dir / 'model')
        motion = (self.policy_dir / 'model' / config['motion_file']).resolve()
        self.assertTrue(motion.is_relative_to(self.policy_dir))
        self.assertFalse(Path(state.onnx_path).samefile(baseline.POLICY / 'model' / config['onnx_path']))
        self.assertFalse(motion.samefile((baseline.POLICY / 'model' / config['motion_file']).resolve()))
        other = baseline.FSMStateTELEOP_YIBO(RobotData())
        for name in ('_buf_cmd', '_buf_ori', '_buf_ang_vel', '_buf_joint_pos', '_buf_joint_vel', '_buf_actions'):
            self.assertFalse(np.shares_memory(getattr(state, name), getattr(other, name)))
        state._buf_cmd.fill(123)
        np.testing.assert_array_equal(other._buf_cmd, 0)


if __name__ == '__main__':
    import unittest
    unittest.main()
