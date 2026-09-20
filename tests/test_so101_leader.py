"""SO-ARM101 leader → PingTi joint map (no Isaac Sim, no serial)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from isaac_so_arm101.devices.leader_map import (
    clipped_joint_report,
    leader_action_from_state,
    leader_state_hold,
    map_gripper_0_100,
    map_signed_m100,
    pingti_follower_action_from_joints,
    pingti_joint_pos_from_leader,
    pingti_named_joints_from_leader,
    strip_leader_keys,
    urdf_rad_to_feetech_m100,
)
from isaac_so_arm101.devices.so101 import (
    FEETECH_VIN_ERROR_BIT,
    FEETECH_VOLTAGE_MAX_V,
    FEETECH_VOLTAGE_MIN_V,
    LEROBOT_INSTALL_HINT,
    MockSO101Leader,
    ScriptedSO101Leader,
)
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
    SO101_PINGTI_SIGN,
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
        for motor, name, value in zip(SO101_LEADER_MOTORS[:-1], PINGTI_JOINTS[:-1], lo[:-1], strict=True):
            lim_lo, lim_hi = PINGTI_JOINT_LIMITS_RAD[name]
            unclipped = SO101_PINGTI_SIGN[motor] * map_signed_m100(-100.0, lim_lo, lim_hi)
            expected = max(lim_lo, min(lim_hi, unclipped))
            self.assertAlmostEqual(value, expected, places=5, msg=name)
        self.assertAlmostEqual(lo[-1], PINGTI_JOINT_LIMITS_RAD[PINGTI_GRIPPER_JOINT][0], places=5)

        hi_state = {f"{name}.pos": 100.0 for name in SO101_LEADER_MOTORS}
        hi = pingti_joint_pos_from_leader(hi_state)
        for motor, name, value in zip(SO101_LEADER_MOTORS[:-1], PINGTI_JOINTS[:-1], hi[:-1], strict=True):
            lim_lo, lim_hi = PINGTI_JOINT_LIMITS_RAD[name]
            unclipped = SO101_PINGTI_SIGN[motor] * map_signed_m100(100.0, lim_lo, lim_hi)
            expected = max(lim_lo, min(lim_hi, unclipped))
            self.assertAlmostEqual(value, expected, places=5, msg=name)
        self.assertAlmostEqual(hi[-1], PINGTI_JOINT_LIMITS_RAD[PINGTI_GRIPPER_JOINT][1], places=5)

    def test_inverted_axes_match_so101_and_clip_asymmetric_elbow(self):
        self.assertEqual(SO101_PINGTI_SIGN["shoulder_pan"], -1.0)
        self.assertEqual(SO101_PINGTI_SIGN["elbow_flex"], -1.0)
        self.assertEqual(SO101_PINGTI_SIGN["wrist_flex"], -1.0)
        self.assertEqual(SO101_PINGTI_SIGN["shoulder_lift"], -1.0)
        plus = pingti_named_joints_from_leader(leader_state_hold({
            "shoulder_pan": 40.0,
            "elbow_flex": 40.0,
            "wrist_flex": 40.0,
            "shoulder_lift": 40.0,
        }))
        self.assertLess(plus["base_yaw"], 0.0)
        self.assertLess(plus["elbow_pitch"], 0.0)
        self.assertLess(plus["wrist_pitch"], 0.0)
        self.assertLess(plus["shoulder_pitch"], 0.0)

        elbow_lo, elbow_hi = PINGTI_JOINT_LIMITS_RAD["elbow_pitch"]
        unclipped = -elbow_hi
        self.assertLess(unclipped, elbow_lo)
        hi_elbow = pingti_named_joints_from_leader(leader_state_hold({"elbow_flex": 100.0}))
        self.assertAlmostEqual(hi_elbow["elbow_pitch"], elbow_lo, places=5)
        self.assertEqual(clipped_joint_report(leader_state_hold({"elbow_flex": 100.0})), ["elbow_pitch"])
        self.assertEqual(clipped_joint_report(leader_state_hold({"elbow_flex": 40.0})), [])

    def test_follow_error_report_shows_present_vs_sim_desired(self):
        from isaac_so_arm101.devices.leader_map import follow_error_report

        joints = pingti_joint_pos_from_leader(leader_state_hold({"shoulder_lift": -100.0}))
        present = {motor: 0.0 for motor in (
            "shoulder_pan", "shoulder_lift", "shoulder_lift_secondary",
            "elbow_flex", "elbow_flex_secondary", "wrist_flex", "wrist_roll", "gripper",
        )}
        present["shoulder_lift"] = -61.0
        lines = follow_error_report(joints, present, {"shoulder_lift": -60.0})
        lift = next(line for line in lines if "motor=shoulder_lift " in line)
        self.assertIn("present=-61.00", lift)
        self.assertIn("desired=", lift)
        self.assertIn("present_minus_desired=", lift)
        self.assertIn("[teleop_hw] compare", lift)

    def test_in_range_leader_maps_to_sim_signed_hardware(self):
        raw = leader_state_hold({
            "shoulder_pan": 30.0,
            "elbow_flex": -20.0,
            "wrist_flex": 15.0,
            "shoulder_lift": 25.0,
        })
        joints = pingti_joint_pos_from_leader(raw)
        recovered = pingti_follower_action_from_joints(joints)
        for motor, joint, rad in zip(SO101_LEADER_MOTORS, PINGTI_JOINTS, joints, strict=True):
            if joint == PINGTI_GRIPPER_JOINT:
                continue
            self.assertAlmostEqual(
                recovered[f"{motor}.pos"],
                SO101_PINGTI_SIGN[motor] * urdf_rad_to_feetech_m100(rad),
                places=5,
                msg=motor,
            )
        self.assertLess(joints[PINGTI_JOINTS.index("shoulder_pitch")], 0.0)
        self.assertGreater(recovered["shoulder_lift.pos"], 0.0)
        self.assertGreater(raw["shoulder_lift.pos"], 0.0)
        self.assertGreater(recovered["shoulder_pan.pos"], 0.0)
        self.assertGreater(raw["shoulder_pan.pos"], 0.0)

    def test_inverted_hw_joints_match_leader_sign_not_sim(self):
        """Live log: pan/lift/elbow/wrist_flex desired matched sim, opposite the SO-101.

        Wrist roll already matched the leader (sign=+1). Undo the sim invert for the
        four flipped axes so Feetech polarity matches the leader.
        """
        raw = leader_state_hold({
            "shoulder_pan": 30.0,
            "shoulder_lift": 25.0,
            "elbow_flex": 20.0,
            "wrist_flex": 15.0,
            "wrist_roll": -10.0,
        })
        joints = pingti_joint_pos_from_leader(raw)
        hw = pingti_follower_action_from_joints(joints)
        self.assertLess(joints[PINGTI_JOINTS.index("base_yaw")], 0.0)
        self.assertLess(joints[PINGTI_JOINTS.index("shoulder_pitch")], 0.0)
        self.assertLess(joints[PINGTI_JOINTS.index("elbow_pitch")], 0.0)
        self.assertLess(joints[PINGTI_JOINTS.index("wrist_pitch")], 0.0)
        self.assertLess(joints[PINGTI_JOINTS.index("wrist_roll")], 0.0)
        self.assertGreater(hw["shoulder_pan.pos"], 0.0)
        self.assertGreater(hw["shoulder_lift.pos"], 0.0)
        self.assertGreater(hw["elbow_flex.pos"], 0.0)
        self.assertGreater(hw["wrist_flex.pos"], 0.0)
        self.assertLess(hw["wrist_roll.pos"], 0.0)
        self.assertAlmostEqual(hw["shoulder_lift_secondary.pos"], -hw["shoulder_lift.pos"], places=5)
        self.assertAlmostEqual(hw["elbow_flex_secondary.pos"], -hw["elbow_flex.pos"], places=5)

    def test_feetech_m100_is_half_turn_not_urdf_stop(self):
        from math import pi

        self.assertAlmostEqual(urdf_rad_to_feetech_m100(0.0), 0.0, places=6)
        self.assertAlmostEqual(urdf_rad_to_feetech_m100(pi / 2), 50.0, places=5)
        self.assertAlmostEqual(urdf_rad_to_feetech_m100(-pi), -100.0, places=5)
        # Pan URDF stop is ±π/2. Mapping that to ±100 was the 2× overshoot.
        pan_lo, pan_hi = PINGTI_JOINT_LIMITS_RAD["base_yaw"]
        self.assertAlmostEqual(urdf_rad_to_feetech_m100(pan_hi), 50.0, places=3)

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
        self.assertIn("add_callback(_key", script)
        self.assertIn("PINGTI_FOLLOW_KEYS", script)
        self.assertIn("PINGTI_FOLLOW_ENABLE_KEYS", script)
        self.assertIn("PINGTI_FOLLOW_HOLD_KEYS", script)
        self.assertIn("poll_kit_key_rising", script)
        self.assertIn("follow_error_report", script)
        self.assertIn("--pingti_follow", script)
        self.assertIn("--disable_pingti_torque", script)
        self.assertIn("disable_pingti_torque_raw", script)
        self.assertIn("pingti_follow", script)
        self.assertIn("PingTi holding present", script)
        self.assertIn("PingTi shutdown", script)
        self.assertIn("atexit.register(pingti.close)", script)
        self.assertIn("disabling PingTi torque", script)
        self.assertIn("pingti_follow={'ON'", script)
        so101_src = (REPO_ROOT / "src/isaac_so_arm101/devices/so101.py").read_text(encoding="utf-8")
        self.assertIn("connect_lerobot_device", so101_src)
        self.assertIn("calibrate=True", so101_src)
        self.assertIn("assert_feetech_bus_voltage_ok", so101_src)
        self.assertIn("set_camera_view", script)
        self.assertIn("_frame_kit_camera", script)
        self.assertIn("snap_to_leader", script)
        self.assertIn("_log_dome_light", script)
        self.assertIn("JointDirLogger", script)
        self.assertIn("teleop_dir", script)
        self.assertIn("--dir_log", script)
        self.assertIn("PingTi missing motor ids", script)
        self.assertNotIn("boosted DomeLight", script)
        self.assertIn("SO101_PINGTI_SIGN", (REPO_ROOT / "src/isaac_so_arm101/teleop_constants.py").read_text(encoding="utf-8"))
        self.assertIn("env.unwrapped", script)
        self.assertIn("deactivate_stale_kit_render", (REPO_ROOT / "src/isaac_so_arm101/assets.py").read_text(encoding="utf-8"))
        pingti_src = (REPO_ROOT / "src/isaac_so_arm101/devices/pingti.py").read_text(encoding="utf-8")
        self.assertIn("SOFollower", pingti_src)
        self.assertNotIn("class FeetechPingTiFollower", pingti_src)
        self.assertIn("seeded calibration from motor registers", pingti_src)
        self.assertIn("motor_ids=range(1, 9)", pingti_src)
        self.assertIn("hold_present_enable_torque", pingti_src)
        self.assertIn("PingTi shutdown", pingti_src)
        self.assertIn("GRIPPER_CLOSED_FLOOR", pingti_src)
        self.assertIn("gripper_goal_from_present", pingti_src)
        self.assertIn("GRIPPER_FEETECH_CLOSED_FLOOR", (REPO_ROOT / "src/isaac_so_arm101/teleop_constants.py").read_text(encoding="utf-8"))
        self.assertIn("gripper_overload", pingti_src)
        self.assertIn("require_full_torque", pingti_src)
        self.assertIn("hw_snapshot", pingti_src)
        self.assertIn("teleop_hw", pingti_src)
        self.assertIn("bus.disable_torque()", pingti_src)
        self.assertIn("slew_goal", pingti_src)
        self.assertIn("GOAL_SLEW_MAX = 3.5", pingti_src)
        self.assertIn("exact opposites of the slewed primary", pingti_src)
        self.assertIn("Slew from Present", pingti_src)
        self.assertIn("urdf_rad_to_feetech_m100", (REPO_ROOT / "src/isaac_so_arm101/devices/leader_map.py").read_text(encoding="utf-8"))
        self.assertIn('bus.write("Goal_Position"', pingti_src)
        self.assertIn("Do not use sync_write on this bus", pingti_src)
        self.assertNotIn("self.bus.sync_write", pingti_src)
        self.assertIn("plan_nudge_and_return", pingti_src)
        self.assertIn("Undo SO101_PINGTI_SIGN", (REPO_ROOT / "src/isaac_so_arm101/devices/leader_map.py").read_text(encoding="utf-8"))
        self.assertIn("_seq_read", pingti_src)
        self.assertNotIn("with self.bus.torque_disabled()", pingti_src)
        self.assertIn("emit_hw_lines", script)
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

    def test_voltage_window_rejects_overloaded_bus(self):
        self.assertEqual(FEETECH_VIN_ERROR_BIT, 1)
        self.assertLess(FEETECH_VOLTAGE_MIN_V, 5.3)
        self.assertGreater(FEETECH_VOLTAGE_MAX_V, 12.0)
        self.assertLessEqual(FEETECH_VOLTAGE_MAX_V, 14.0)
        src = (REPO_ROOT / "src/isaac_so_arm101/devices/so101.py").read_text(encoding="utf-8")
        self.assertIn("assert_feetech_bus_voltage_ok", src)
        self.assertNotIn("ignore_feetech_vin_status_errors", src)
        self.assertNotIn("MotorsBus._is_error", src)

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

    def test_dir_log_visual_match_uses_sign_table(self):
        from isaac_so_arm101.devices.dir_log import pair_status

        numeric, visual = pair_status(10.0, -0.2, -1.0)
        self.assertEqual(numeric, "opposite")
        self.assertEqual(visual, "same")
        numeric, visual = pair_status(10.0, 0.2, -1.0)
        self.assertEqual(numeric, "same")
        self.assertEqual(visual, "opposite")
        numeric, visual = pair_status(10.0, 0.2, 1.0)
        self.assertEqual(numeric, "same")
        self.assertEqual(visual, "same")
        numeric, visual = pair_status(0.1, 0.2, -1.0)
        self.assertEqual(numeric, "hold")
        self.assertEqual(visual, "hold")

    def test_dir_log_writes_grep_lines(self):
        import tempfile
        from pathlib import Path

        from isaac_so_arm101.devices.dir_log import JointDirLogger
        from isaac_so_arm101.devices.leader_map import leader_state_hold, pingti_follower_action_from_joints, pingti_joint_pos_from_leader

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teleop_joint_dir.log"
            log = JointDirLogger(path)
            raw0 = leader_state_hold()
            cmd0 = pingti_joint_pos_from_leader(raw0)
            meas0 = {name: value for name, value in zip(PINGTI_JOINTS, cmd0, strict=True)}
            log.step(step=1, raw_leader=raw0, sim_cmd=cmd0, sim_meas=meas0, force=True)
            raw1 = leader_state_hold({"shoulder_lift": 20.0})
            cmd1 = pingti_joint_pos_from_leader(raw1)
            meas1 = {name: value for name, value in zip(PINGTI_JOINTS, cmd1, strict=True)}
            hw1 = pingti_follower_action_from_joints(cmd1)
            log.step(step=2, raw_leader=raw1, sim_cmd=cmd1, sim_meas=meas1, pingti_action=hw1)
            log.close()
            text = path.read_text(encoding="utf-8")
        self.assertIn("[teleop_dir]", text)
        self.assertIn("motor=shoulder_lift", text)
        self.assertIn("visual=same", text)
        self.assertIn("numeric=opposite", text)
        self.assertIn("hw_vs_leader=same", text)
        self.assertGreater(hw1["shoulder_lift.pos"], 0.0)

    def test_hw_log_compares_sent_goal_present_torque(self):
        from isaac_so_arm101.devices.dir_log import format_hw_lines
        from isaac_so_arm101.teleop_constants import PINGTI_FOLLOWER_MOTORS

        snap = {
            "present": {name: 0.0 for name in PINGTI_FOLLOWER_MOTORS},
            "goal": {name: -80.0 if name == "shoulder_lift" else 0.0 for name in PINGTI_FOLLOWER_MOTORS},
            "present_raw": {name: 2048 for name in PINGTI_FOLLOWER_MOTORS},
            "goal_raw": {name: 300 if name == "shoulder_lift" else 2048 for name in PINGTI_FOLLOWER_MOTORS},
            "torque": {name: 0 if name == "shoulder_lift" else 1 for name in PINGTI_FOLLOWER_MOTORS},
            "mode": {name: 0 for name in PINGTI_FOLLOWER_MOTORS},
            "moving": {name: 0 for name in PINGTI_FOLLOWER_MOTORS},
        }
        action = {f"{name}.pos": -80.0 if name == "shoulder_lift" else 0.0 for name in PINGTI_FOLLOWER_MOTORS}
        lines = format_hw_lines(step=12, pingti_action=action, snap=snap)
        self.assertEqual(len(lines), len(PINGTI_FOLLOWER_MOTORS))
        lift = next(line for line in lines if "motor=shoulder_lift sent=" in line)
        self.assertIn("[teleop_hw]", lift)
        self.assertIn("sent=-80.00", lift)
        self.assertIn("goal=-80.00", lift)
        self.assertIn("present=0.00", lift)
        self.assertIn("torque=0", lift)
        self.assertIn("goal_wrote=yes", lift)
        self.assertIn("goal_hit=no", lift)
        none_lines = format_hw_lines(step=1, pingti_action=None, snap=None)
        self.assertEqual(none_lines, ["[teleop_hw] step=1 kind=no_bus"])


if __name__ == "__main__":
    unittest.main()
