"""Fixed-mode simulation startup/reset contract for a mixed policy."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, sys.argv.pop(1))
try:
    import mujoco
except ImportError:
    mujoco = None
else:
    import policy_deploy_toolkit.simulate as simulation


@unittest.skipIf(mujoco is None, 'MuJoCo is not installed')
class MixedModeSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names = [f'{joint}_{leg}' for leg in range(1, 5)
                     for joint in ('hip_roll', 'hip_pitch', 'knee_pitch')]
        # Reverse the MuJoCo actuator order to catch joint-mapping mistakes.
        names = cls.names[::-1]
        bodies = ''.join(f'<body><joint name="{name}"/><geom type="sphere" size=".02" mass="1"/></body>' for name in names)
        actuators = ''.join(f'<position joint="{name}" kp="10" kv="1"/>' for name in names)
        cls.model_xml = ('<mujoco><option timestep=".005"/><worldbody><body name="base_link">'
                         '<site name="imu_link"/><geom type="sphere" size=".1" mass="1"/>'
                         f'{bodies}</body></worldbody><actuator>{actuators}</actuator>'
                         '<sensor><gyro site="imu_link"/><accelerometer site="imu_link"/></sensor></mujoco>')
        cls.references = {name: np.arange(12)*.01 + i*.1 for i, name in enumerate(('bb', 'bf', 'fb', 'ff'))}
        cls.selectors = {'bb': [-1, -1], 'bf': [-1, 1], 'fb': [1, -1], 'ff': [1, 1]}

    def bridge(self, mode, *, mixed=True):
        owner = self

        class FakePolicy:
            def __init__(self, model_path, metadata_path, obs_group, robot):
                self.robot = robot
            def policyInfo(self):
                return SimpleNamespace(
                    mixed_modes=[SimpleNamespace(name=name, selectors=np.array(owner.selectors[name], dtype=float),
                                                 joint_reference_rad=owner.references[name])
                                 for name in owner.references] if mixed else [],
                    joint_default_pos=owner.references['bb'],
                    joint_id_policy_to_robot=[self.robot.joint_names.index(name) for name in owner.names],
                    control_dt=.02, action_size=12)
            def default_commands(self): return {}
            def sensor_specs(self): return []
            def run(self, inputs, outputs):
                self.seen_selectors = inputs.mode_selectors.copy()
                return True
        model = mujoco.MjModel.from_xml_string(self.model_xml)
        data = mujoco.MjData(model)
        with patch.object(simulation, 'OnnxPolicy', FakePolicy):
            return simulation.MjOnnxBridge(model, data, 'policy.onnx', 'metadata.yaml', mode=mode)

    def test_all_four_modes_start_and_reset_consistently(self):
        for mode, ref in self.references.items():
            with self.subTest(mode=mode):
                bridge = self.bridge(mode)
                policy_order = np.asarray(bridge.info.joint_id_policy_to_robot)
                np.testing.assert_allclose(bridge.data.qpos[bridge.qpos_ids][policy_order], ref)
                np.testing.assert_array_equal(bridge.inputs.mode_selectors, self.selectors[mode])
                bridge.evaluate()
                np.testing.assert_array_equal(bridge.policy.seen_selectors, self.selectors[mode])
                # A reset reloads the policy and restores the selected mode.
                with patch.object(simulation, 'OnnxPolicy', type(bridge.policy)):
                    bridge.reset()
                np.testing.assert_array_equal(bridge.inputs.mode_selectors, self.selectors[mode])
                bridge.evaluate()
                np.testing.assert_array_equal(bridge.policy.seen_selectors, self.selectors[mode])

    def test_explicit_mode_required_only_for_mixed_policy(self):
        with self.assertRaisesRegex(ValueError, 'explicit mode'):
            self.bridge(None)
        with self.assertRaisesRegex(ValueError, 'only supported for mixed'):
            self.bridge('bb', mixed=False)
        self.bridge(None, mixed=False)

    def test_cli_option(self):
        args = simulation.create_arg_parser().parse_args(['--mjcf', 'robot.xml', '--policy', 'policy.onnx', '--mode', 'fb'])
        self.assertEqual(args.mode, 'fb')


if __name__ == '__main__':
    unittest.main()
