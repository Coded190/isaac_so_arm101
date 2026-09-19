"""SO-ARM101 leader → PingTi joint map (no Isaac Sim, no serial)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from isaac_so_arm101.devices.leader_map import (
    leader_action_from_state,
    leader_state_hold,
    map_gripper_0_100,
    map_signed_m100,
    pingti_joint_pos_from_leader,
    strip_leader_keys,
)
from isaac_so_arm101.devices.so101 import MockSO101Leader, ScriptedSO101Leader, LEROBOT_INSTALL_HINT
from isaac_so_arm101.teleop_constants import (
    JOINT_POS_ACTION_DIM,
    PINGTI_GRIPPER_JOINT,
    PINGTI_JOINT_LIMITS_RAD,
    PINGTI_JOINT_MOTOR_COUNTS,
    PINGTI_JOINTS,
    PINGTI_PHYSICAL_MOTOR_COUNT,
    SE3_ACTION_DIM,
    SO101_LEADER_MOTORS,
    SO101_PHYSICAL_MOTOR_COUNT,
    SO101_TO_PINGTI,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
URDF = REPO_ROOT / "src/isaac_so_arm101/robots/pingti/PingTi_Arm_5DOF_v4_copy.urdf"
TELEOP_CFG = REPO_ROOT / "src/isaac_so_arm101/tasks/teleop/teleop_env_cfg.py"
TELEOP_SCRIPT = REPO_ROOT / "src/isaac_so_arm101/scripts/teleop_keyboard.py"


def _urdf_joint_limits() -> dict[str, tuple[float, float]]:
    text = URDF.read_text(encoding="utf-8")
    blocks = re.findall(
        r'<joint name="([^"]+)" type="revolute">.*?<limit [^>]*lower="([^"]+)" upper="([^"]+)"',
        text,
        flags=re.DOTALL,
    )
    return {name: (float(lo), float(hi)) for name, lo, hi in blocks}


class So101LeaderMapTests(unittest.TestCase):
    def test_action_dims(self):
        self.assertEqual(SE3_ACTION_DIM, 7)
        self.assertEqual(JOINT_POS_ACTION_DIM, 6)
        self.assertEqual(len(PINGTI_JOINTS), 6)
        self.assertEqual(len(SO101_LEADER_MOTORS), 6)

    def test_mapping_order_matches_pingti_joints(self):
        self.assertEqual(tuple(SO101_TO_PINGTI[m] for m in SO101_LEADER_MOTORS), PINGTI_JOINTS)

    def test_urdf_limits_match_constants(self):
        urdf = _urdf_joint_limits()
        for joint, expected in PINGTI_JOINT_LIMITS_RAD.items():
            self.assertIn(joint, urdf)
            self.assertAlmostEqual(urdf[joint][0], expected[0], places=5, msg=joint)
            self.assertAlmostEqual(urdf[joint][1], expected[1], places=5, msg=joint)

    def test_zero_leader_is_zero_arm_and_closed_gripper_lo(self):
        zeros = {name: 0.0 for name in SO101_LEADER_MOTORS}
        out = pingti_joint_pos_from_leader(zeros)
        self.assertEqual(len(out), JOINT_POS_ACTION_DIM)
        for name, value in zip(PINGTI_JOINTS, out, strict=True):
            if name == PINGTI_GRIPPER_JOINT:
                self.assertAlmostEqual(value, PINGTI_JOINT_LIMITS_RAD[name][0], places=6)
            else:
                self.assertAlmostEqual(value, 0.0, places=6)

    def test_dot_pos_keys_and_full_range(self):
        lo_state = {f"{name}.pos": -100.0 for name in SO101_LEADER_MOTORS[:-1]}
        lo_state["gripper.pos"] = 0.0
        lo = pingti_joint_pos_from_leader(lo_state)
        for name, value in zip(PINGTI_JOINTS[:-1], lo[:-1], strict=True):
            self.assertAlmostEqual(value, PINGTI_JOINT_LIMITS_RAD[name][0], places=5, msg=name)
        self.assertAlmostEqual(lo[-1], PINGTI_JOINT_LIMITS_RAD[PINGTI_GRIPPER_JOINT][0], places=5)

        hi_state = {f"{name}.pos": 100.0 for name in SO101_LEADER_MOTORS}
        hi = pingti_joint_pos_from_leader(hi_state)
        for name, value in zip(PINGTI_JOINTS[:-1], hi[:-1], strict=True):
            self.assertAlmostEqual(value, PINGTI_JOINT_LIMITS_RAD[name][1], places=5, msg=name)
        self.assertAlmostEqual(hi[-1], PINGTI_JOINT_LIMITS_RAD[PINGTI_GRIPPER_JOINT][1], places=5)

    def test_asymmetric_shoulder_preserves_zero(self):
        lo, hi = PINGTI_JOINT_LIMITS_RAD["shoulder_pitch"]
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertAlmostEqual(map_signed_m100(0.0, lo, hi), 0.0, places=6)
        self.assertAlmostEqual(map_signed_m100(100.0, lo, hi), hi, places=6)
        self.assertAlmostEqual(map_signed_m100(-100.0, lo, hi), lo, places=6)

    def test_gripper_lerp(self):
        lo, hi = PINGTI_JOINT_LIMITS_RAD[PINGTI_GRIPPER_JOINT]
        self.assertAlmostEqual(map_gripper_0_100(0.0, lo, hi), lo, places=6)
        self.assertAlmostEqual(map_gripper_0_100(100.0, lo, hi), hi, places=6)
        mid = map_gripper_0_100(50.0, lo, hi)
        self.assertAlmostEqual(mid, (lo + hi) / 2.0, places=5)

    def test_follower_keeps_leader_motor_space(self):
        state = {"shoulder_pan.pos": 12.0, "shoulder_lift": -3.0}
        for name in SO101_LEADER_MOTORS[2:]:
            state[name] = 0.0
        action = leader_action_from_state(state)
        self.assertEqual(action["shoulder_pan.pos"], 12.0)
        self.assertEqual(action["shoulder_lift.pos"], -3.0)
        self.assertNotIn("base_yaw", action)
        self.assertNotIn("gripper_moving", action)

    def test_missing_motor_raises(self):
        with self.assertRaises(KeyError):
            pingti_joint_pos_from_leader({"shoulder_pan": 0.0})

    def test_mock_leader_zeros(self):
        mock = MockSO101Leader()
        out = pingti_joint_pos_from_leader(mock.get_action())
        self.assertEqual(len(out), 6)
        for value in out[:-1]:
            self.assertAlmostEqual(value, 0.0, places=6)
        mock.disconnect()

    def test_strip_leader_keys(self):
        self.assertEqual(strip_leader_keys({"elbow_flex.pos": 1.5}), {"elbow_flex": 1.5})

    def test_cli_uses_teleop_device_not_lab_device(self):
        script = TELEOP_SCRIPT.read_text(encoding="utf-8")
        cfg = TELEOP_CFG.read_text(encoding="utf-8")
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn('--teleop_device', script)
        self.assertIn("so101leader", script)
        self.assertIn("--follower_port", script)
        self.assertIn("--pingti_port", script)
        self.assertIn("mock_leader", script)
        self.assertIn("mock_pingti", script)
        self.assertIn("mock_follower", script)
        self.assertIn("send_sim_joints_to_pingti", script)
        self.assertIn("connect_lerobot_device", (REPO_ROOT / "src/isaac_so_arm101/devices/so101.py").read_text(encoding="utf-8"))
        self.assertIn("calibrate=True", (REPO_ROOT / "src/isaac_so_arm101/devices/so101.py").read_text(encoding="utf-8"))
        pingti_src = (REPO_ROOT / "src/isaac_so_arm101/devices/pingti.py").read_text(encoding="utf-8")
        self.assertIn("SOFollower", pingti_src)
        self.assertNotIn("class FeetechPingTiFollower", pingti_src)
        self.assertIn("apply_teleop_device", script)
        self.assertIn("use_default_offset=False", cfg)
        self.assertIn("preserve_order=True", cfg)
        self.assertIn("JointPositionActionCfg", cfg)
        self.assertIn("lerobot[feetech]", LEROBOT_INSTALL_HINT)
        self.assertIn("id_ed25519", gitignore)
        self.assertIn("*.pem", gitignore)
        self.assertNotIn("/home/cirplab", script)
        self.assertNotIn("/home/cirp-lab", script)
        self.assertNotIn("/home/cirplab", cfg)

    def test_pingti_has_two_extra_motors_on_dual_joints(self):
        self.assertEqual(sum(PINGTI_JOINT_MOTOR_COUNTS.values()), PINGTI_PHYSICAL_MOTOR_COUNT)
        self.assertEqual(PINGTI_PHYSICAL_MOTOR_COUNT - SO101_PHYSICAL_MOTOR_COUNT, 2)
        self.assertEqual(PINGTI_JOINT_MOTOR_COUNTS["shoulder_pitch"], 2)
        self.assertEqual(PINGTI_JOINT_MOTOR_COUNTS["elbow_pitch"], 2)
        self.assertEqual(len(PINGTI_JOINTS), SO101_PHYSICAL_MOTOR_COUNT)
        urdf = URDF.read_text(encoding="utf-8")
        self.assertIn("sts3215_shoulder_pitch_1", urdf)
        self.assertIn("sts3215_shoulder_pitch_2", urdf)
        self.assertIn("sts3215_elbow_pitch_1", urdf)
        self.assertIn("sts3215_elbow_pitch_2", urdf)

    def test_one_leader_motor_moves_only_mapped_joint(self):
        gripper_lo = PINGTI_JOINT_LIMITS_RAD[PINGTI_GRIPPER_JOINT][0]
        for motor, joint in SO101_TO_PINGTI.items():
            cmd = 50.0 if motor == "gripper" else 40.0
            out = pingti_joint_pos_from_leader(leader_state_hold({motor: cmd}))
            idx = PINGTI_JOINTS.index(joint)
            for i, (name, value) in enumerate(zip(PINGTI_JOINTS, out, strict=True)):
                if i == idx:
                    self.assertGreater(abs(value - (gripper_lo if name == PINGTI_GRIPPER_JOINT else 0.0)), 0.2, msg=name)
                    continue
                if name == PINGTI_GRIPPER_JOINT:
                    self.assertAlmostEqual(value, gripper_lo, places=5, msg=f"{motor} leaked to gripper")
                else:
                    self.assertAlmostEqual(value, 0.0, places=5, msg=f"{motor} leaked to {name}")

    def test_scripted_leader_replays_real_shaped_frames(self):
        frames = [leader_state_hold({"shoulder_lift": float(i)}) for i in (0.0, 20.0, 40.0)]
        scripted = ScriptedSO101Leader(frames)
        last = None
        for expected in frames:
            last = scripted.get_action()
            self.assertEqual(last["shoulder_lift.pos"], expected["shoulder_lift.pos"])
            self.assertEqual(last["shoulder_pan.pos"], 0.0)
        self.assertEqual(scripted.get_action()["shoulder_lift.pos"], 40.0)
        mapped = pingti_joint_pos_from_leader(last)
        self.assertGreater(abs(mapped[PINGTI_JOINTS.index("shoulder_pitch")]), 0.3)
        self.assertAlmostEqual(mapped[PINGTI_JOINTS.index("base_yaw")], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
