"""Four teleop paths: keyboard→sim, leader→sim, leader→SO101, leader→PingTi.

No Isaac Sim and no USB. Sim joint dicts are synthetic stand-ins for
``joint_pos_target`` after ``env.step``. Live serial is skipped unless
``ISAAC_SO_ARM101_PINGTI_PORT`` / ``ISAAC_SO_ARM101_SO101_FOLLOWER_PORT`` are set
(those still do not send motion — they only check the port string is non-empty).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from isaac_so_arm101.devices.leader_map import (
    joints6_from_named,
    leader_action_from_state,
    leader_state_hold,
    map_signed_m100,
    pingti_follower_action_from_joints,
    pingti_follower_action_from_leader,
    pingti_joint_pos_from_leader,
    rad_to_gripper_0_100,
    rad_to_signed_m100,
)
from isaac_so_arm101.devices.pingti import (
    MockPingTiFollower,
    open_pingti_follower,
    pingti_motor_table,
)
from isaac_so_arm101.devices.pipeline import (
    pingti_action_from_sim_named,
    run_leader_hw_loop,
    step_leader_followers,
)
from isaac_so_arm101.devices.so101 import (
    MockSO101Follower,
    ScriptedSO101Leader,
    calibration_file_status,
    open_so101_follower,
)
from isaac_so_arm101.scripts.teleop_hw import main as teleop_hw_main
from isaac_so_arm101.teleop_constants import (
    JOINT_POS_ACTION_DIM,
    PINGTI_DUAL_JOINTS,
    PINGTI_FOLLOWER_MOTOR_IDS,
    PINGTI_FOLLOWER_MOTORS,
    PINGTI_GRIPPER_JOINT,
    PINGTI_JOINT_LIMITS_RAD,
    PINGTI_JOINT_MOTOR_COUNTS,
    PINGTI_JOINT_TO_FOLLOWER_MOTORS,
    PINGTI_JOINTS,
    PINGTI_PHYSICAL_MOTOR_COUNT,
    SE3_ACTION_DIM,
    SO101_LEADER_MOTORS,
    SO101_TO_PINGTI,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TELEOP_SCRIPT = REPO_ROOT / "src/isaac_so_arm101/scripts/teleop_keyboard.py"
TELEOP_HW = REPO_ROOT / "src/isaac_so_arm101/scripts/teleop_hw.py"
SMOKE = REPO_ROOT / "src/isaac_so_arm101/scripts/teleop_sim_smoke.py"


def _assert_duals_mirrored(testcase: unittest.TestCase, action: dict[str, float]) -> None:
    for joint in PINGTI_DUAL_JOINTS:
        motors = PINGTI_JOINT_TO_FOLLOWER_MOTORS[joint]
        testcase.assertEqual(len(motors), 2, msg=joint)
        a = action[f"{motors[0]}.pos"]
        b = action[f"{motors[1]}.pos"]
        testcase.assertAlmostEqual(b, -a, places=5, msg=f"{joint} dual {a} vs {b}")


class KeyboardToSimPathTests(unittest.TestCase):
    """Keyboard is 7-D SE3; after IK the sim exposes 6 joint targets → 8 motors."""

    def test_se3_is_not_joint_space(self):
        self.assertEqual(SE3_ACTION_DIM, 7)
        self.assertNotEqual(SE3_ACTION_DIM, JOINT_POS_ACTION_DIM)
        self.assertEqual(len(PINGTI_JOINTS), 6)

    def test_sim_joint_dict_expands_to_eight_motors(self):
        named = {name: 0.0 for name in PINGTI_JOINTS}
        named["shoulder_pitch"] = 0.4
        named["elbow_pitch"] = -0.2
        named["gripper_moving"] = 0.8
        action = pingti_action_from_sim_named(named)
        self.assertEqual(len(action), PINGTI_PHYSICAL_MOTOR_COUNT)
        self.assertEqual(tuple(k.removesuffix(".pos") for k in action), PINGTI_FOLLOWER_MOTORS)
        _assert_duals_mirrored(self, action)
        self.assertGreater(action["shoulder_lift.pos"], 20.0)
        self.assertAlmostEqual(action["shoulder_lift_secondary.pos"], -action["shoulder_lift.pos"], places=5)
        self.assertLess(action["elbow_flex.pos"], -10.0)
        self.assertGreater(action["gripper.pos"], 40.0)
        self.assertAlmostEqual(action["shoulder_pan.pos"], 0.0, places=5)
        mock = MockPingTiFollower()
        mock.send_action(action)
        self.assertEqual(len(mock.sent), 1)
        self.assertAlmostEqual(mock.sent[0]["shoulder_lift.pos"], -mock.sent[0]["shoulder_lift_secondary.pos"], places=5)

    def test_keyboard_cli_wires_pingti_after_sim_step(self):
        source = TELEOP_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--pingti_port", source)
        self.assertIn("--mock_pingti", source)
        self.assertIn("send_sim_joints_to_pingti", source)
        self.assertIn("env.step(actions)", source)
        step_at = source.index("env.step(actions)")
        send_at = source.index("send_sim_joints_to_pingti(robot, pingti)")
        self.assertGreater(send_at, step_at)
        smoke = SMOKE.read_text(encoding="utf-8")
        self.assertIn("KEYBOARD_DZ", smoke)
        self.assertIn("KEYBOARD_DY", smoke)
        self.assertIn("mock pingti", smoke.lower())


class LeaderToSimPathTests(unittest.TestCase):
    def test_one_leader_motor_to_one_sim_joint_then_dual_motors(self):
        gripper_lo = PINGTI_JOINT_LIMITS_RAD[PINGTI_GRIPPER_JOINT][0]
        for motor, joint in SO101_TO_PINGTI.items():
            cmd = 80.0 if motor == "gripper" else 40.0
            joints = pingti_joint_pos_from_leader(leader_state_hold({motor: cmd}))
            idx = PINGTI_JOINTS.index(joint)
            for i, (name, value) in enumerate(zip(PINGTI_JOINTS, joints, strict=True)):
                if i == idx:
                    rest = gripper_lo if name == PINGTI_GRIPPER_JOINT else 0.0
                    self.assertGreater(abs(value - rest), 0.2, msg=name)
                elif name == PINGTI_GRIPPER_JOINT:
                    self.assertAlmostEqual(value, gripper_lo, places=5)
                else:
                    self.assertAlmostEqual(value, 0.0, places=5)
            action = pingti_follower_action_from_joints(joints)
            _assert_duals_mirrored(self, action)
            driven = PINGTI_JOINT_TO_FOLLOWER_MOTORS[joint]
            self.assertGreater(abs(action[f"{driven[0]}.pos"]), 1.0, msg=motor)
            for other, other_motors in PINGTI_JOINT_TO_FOLLOWER_MOTORS.items():
                if other == joint:
                    continue
                for name in other_motors:
                    self.assertAlmostEqual(action[f"{name}.pos"], 0.0, places=5, msg=f"{motor}->{name}")

    def test_scripted_leader_replays_into_sim_sized_vector(self):
        frames = [leader_state_hold({"elbow_flex": 25.0}), leader_state_hold({"elbow_flex": 50.0})]
        leader = ScriptedSO101Leader(frames)
        first = pingti_joint_pos_from_leader(leader.get_action())
        second = pingti_joint_pos_from_leader(leader.get_action())
        self.assertEqual(len(first), JOINT_POS_ACTION_DIM)
        elbow = PINGTI_JOINTS.index("elbow_pitch")
        self.assertGreater(second[elbow], first[elbow])
        self.assertAlmostEqual(first[PINGTI_JOINTS.index("base_yaw")], 0.0, places=5)


class LeaderToSo101FollowerTests(unittest.TestCase):
    def test_follower_gets_leader_motor_space_not_pingti(self):
        raw = leader_state_hold({"shoulder_pan": 12.0, "gripper": 40.0})
        follower = MockSO101Follower()
        pingti = MockPingTiFollower()
        leader = ScriptedSO101Leader([raw])
        step = step_leader_followers(leader, so101=follower, pingti=pingti)
        self.assertEqual(follower.sent[0]["shoulder_pan.pos"], 12.0)
        self.assertEqual(follower.sent[0]["gripper.pos"], 40.0)
        self.assertNotIn("base_yaw.pos", follower.sent[0])
        self.assertNotIn("gripper_moving.pos", follower.sent[0])
        self.assertEqual(len(follower.sent[0]), 6)
        self.assertEqual(step.so101_action, follower.sent[0])

    def test_wrong_keys_rejected(self):
        follower = MockSO101Follower()
        with self.assertRaises(KeyError):
            follower.send_action({"base_yaw.pos": 1.0})

    def test_open_mock_so101_follower(self):
        session = open_so101_follower(port="mock", mock=True)
        session.send_action(leader_action_from_state(leader_state_hold()))
        session.close()


class LeaderToPingTiFollowerTests(unittest.TestCase):
    def test_feetech_motor_table_constructs_without_serial(self):
        from isaac_so_arm101.devices.pingti import _feetech_types, _pingti_motors

        Motor, MotorNormMode, FeetechMotorsBus, OperatingMode = _feetech_types()
        motors = _pingti_motors(Motor, MotorNormMode)
        self.assertEqual(len(motors), 8)
        self.assertEqual(motors["shoulder_lift_secondary"].id, 2)
        self.assertEqual(motors["shoulder_lift"].id, 3)
        self.assertEqual(motors["gripper"].id, 8)
        self.assertEqual(motors["shoulder_lift"].model, "sts3250")
        self.assertEqual(motors["shoulder_lift_secondary"].model, "sts3250")
        self.assertEqual(motors["elbow_flex"].model, "sts3215")
        self.assertIs(motors["gripper"].norm_mode, MotorNormMode.RANGE_0_100)
        self.assertIs(motors["shoulder_pan"].norm_mode, MotorNormMode.RANGE_M100_100)
        self.assertTrue(callable(FeetechMotorsBus))
        self.assertTrue(hasattr(OperatingMode, "POSITION"))
        joints = pingti_joint_pos_from_leader(leader_state_hold({"shoulder_lift": 40.0}))
        action = pingti_follower_action_from_joints(joints)
        self.assertEqual(len(action), 8)
        self.assertAlmostEqual(action["shoulder_lift_secondary.pos"], -action["shoulder_lift.pos"], places=5)
        self.assertAlmostEqual(action["elbow_flex.pos"], 0.0, places=5)
        self.assertAlmostEqual(action["elbow_flex_secondary.pos"], 0.0, places=5)
        self.assertGreater(action["shoulder_lift.pos"], 20.0)
        from_leader = pingti_follower_action_from_leader(leader_state_hold({"shoulder_lift": 40.0}))
        self.assertAlmostEqual(from_leader["shoulder_lift.pos"], 40.0, places=5)
        self.assertAlmostEqual(from_leader["shoulder_lift_secondary.pos"], -40.0, places=5)
        table = pingti_motor_table()
        self.assertEqual(len(table), 8)
        self.assertEqual(table["shoulder_lift_secondary"], (2, "sts3250"))
        self.assertEqual(table["gripper"], (8, "sts3215"))
        self.assertEqual(sum(PINGTI_JOINT_MOTOR_COUNTS.values()), len(PINGTI_FOLLOWER_MOTOR_IDS))

    def test_pingti_follower_reuses_lerobot_so_follower(self):
        from lerobot.robots.so_follower import SOFollower

        from isaac_so_arm101.devices.pingti import make_pingti_follower
        from isaac_so_arm101.devices.so101 import connect_lerobot_device, require_distinct_serial_ports

        with tempfile.TemporaryDirectory() as tmp:
            device = make_pingti_follower(
                port="/dev/null",
                robot_id="pingti_test",
                calibration_dir=Path(tmp),
            )
            self.assertIsInstance(device, SOFollower)
            self.assertIs(type(device).connect, SOFollower.connect)
            self.assertIs(type(device).disconnect, SOFollower.disconnect)
            self.assertIsNot(type(device).send_action, SOFollower.send_action)
            self.assertEqual(len(device.bus.motors), 8)
            self.assertEqual(device.bus.motors["shoulder_lift"].model, "sts3250")
            self.assertEqual(device.bus.motors["shoulder_lift_secondary"].model, "sts3250")
            self.assertEqual(device.bus.motors["gripper"].norm_mode.value, "range_0_100")
            self.assertFalse(getattr(device.config, "use_degrees", True))
            path, exists, loaded = calibration_file_status(device)
            self.assertIsNotNone(path)
            self.assertFalse(exists)
            self.assertFalse(loaded)
            self.assertIn("pingti_test", str(path))
        self.assertEqual(connect_lerobot_device.__doc__.count("calibrate=True"), 1)
        with self.assertRaises(SystemExit):
            require_distinct_serial_ports("/dev/ttyACM0", "/dev/ttyACM0")
        require_distinct_serial_ports("/dev/ttyACM0", "/dev/ttyACM1", None)

    def test_roundtrip_signed_and_gripper(self):
        for joint in PINGTI_JOINTS:
            lo, hi = PINGTI_JOINT_LIMITS_RAD[joint]
            samples = (lo, 0.0, hi) if joint != PINGTI_GRIPPER_JOINT else (lo, (lo + hi) / 2.0, hi)
            for rad in samples:
                if joint == PINGTI_GRIPPER_JOINT:
                    norm = rad_to_gripper_0_100(rad, lo, hi)
                    # gripper 0–100 is a linear lerp; inverse is map_gripper via joints helper
                    joints = [0.0] * 6
                    joints[-1] = rad
                    action = pingti_follower_action_from_joints(joints)
                    self.assertAlmostEqual(action["gripper.pos"], norm, places=5)
                else:
                    norm = rad_to_signed_m100(rad, lo, hi)
                    back = map_signed_m100(norm, lo, hi)
                    self.assertAlmostEqual(back, rad, places=5, msg=joint)

    def test_mock_pingti_open_and_send_joints(self):
        session = open_pingti_follower(port="mock", mock=True)
        joints = pingti_joint_pos_from_leader(leader_state_hold({"wrist_roll": -30.0}))
        session.send_joints(joints)
        inner = session._inner
        self.assertEqual(len(inner.sent), 1)
        self.assertLess(inner.sent[0]["wrist_roll.pos"], -20.0)
        self.assertAlmostEqual(inner.sent[0]["shoulder_pan.pos"], 0.0, places=5)
        session.close()

    def test_pipeline_loop_isolation(self):
        frames = [leader_state_hold()]
        for motor in SO101_LEADER_MOTORS:
            cmd = 80.0 if motor == "gripper" else 40.0
            frames.append(leader_state_hold({motor: cmd}))
        leader = ScriptedSO101Leader(frames)
        so101 = MockSO101Follower()
        pingti = MockPingTiFollower()
        result = run_leader_hw_loop(leader, so101=so101, pingti=pingti, steps=len(frames), hz=0.0)
        self.assertEqual(result.steps, len(frames))
        self.assertEqual(len(so101.sent), len(frames))
        self.assertEqual(len(pingti.sent), len(frames))
        for i, motor in enumerate(SO101_LEADER_MOTORS):
            pingti_action = pingti.sent[i + 1]
            so101_action = so101.sent[i + 1]
            joint = SO101_TO_PINGTI[motor]
            duals = PINGTI_JOINT_TO_FOLLOWER_MOTORS[joint]
            primary = pingti_action[f"{duals[0]}.pos"]
            self.assertGreater(abs(primary), 1.0)
            if len(duals) == 2:
                self.assertAlmostEqual(pingti_action[f"{duals[1]}.pos"], -primary, places=5)
            else:
                self.assertEqual(pingti_action[f"{duals[0]}.pos"], pingti_action[f"{duals[-1]}.pos"])
            self.assertGreater(abs(so101_action[f"{motor}.pos"]), 1.0)
            for other in SO101_LEADER_MOTORS:
                if other != motor:
                    self.assertAlmostEqual(so101_action[f"{other}.pos"], 0.0, places=5)

    def test_teleop_hw_mock_entry(self):
        self.assertEqual(teleop_hw_main(["--mock", "--hz", "0"]), 0)
        source = TELEOP_HW.read_text(encoding="utf-8")
        self.assertIn("--pingti_port", source)
        self.assertIn("no Isaac Sim", source)
        self.assertNotIn("/home/cirplab", source)

    def test_live_serial_skipped_without_env(self):
        pingti_port = os.environ.get("ISAAC_SO_ARM101_PINGTI_PORT")
        so101_port = os.environ.get("ISAAC_SO_ARM101_SO101_FOLLOWER_PORT")
        if not pingti_port:
            self.assertTrue(True)  # mock path is the default CI coverage
        else:
            self.assertTrue(pingti_port.startswith("/dev/"))
        if not so101_port:
            self.assertTrue(True)
        else:
            self.assertTrue(so101_port.startswith("/dev/"))

    def test_missing_sim_joint_raises(self):
        with self.assertRaises(KeyError):
            joints6_from_named({"base_yaw": 0.0})
        with self.assertRaises(ValueError):
            pingti_follower_action_from_joints((0.0, 0.0))


class CalibrationFileCheckTests(unittest.TestCase):
    def test_missing_and_present_json(self):
        class Dummy:
            calibration_fpath = Path("/tmp/does-not-exist-isaac-so-arm101-cal.json")
            calibration = {}

        path, exists, loaded = calibration_file_status(Dummy())
        self.assertEqual(path, Dummy.calibration_fpath)
        self.assertFalse(exists)
        self.assertFalse(loaded)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            cal_path = Path(handle.name)
            handle.write(b"{}")
        try:

            class Loaded:
                calibration_fpath = cal_path
                calibration = {"shoulder_pan": object()}

            path, exists, loaded = calibration_file_status(Loaded())
            self.assertEqual(path, cal_path)
            self.assertTrue(exists)
            self.assertTrue(loaded)
        finally:
            cal_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
