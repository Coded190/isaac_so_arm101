"""Unit tests for teleop action layout (no Isaac Sim)."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from isaac_so_arm101.teleop_constants import (
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    PINGTI_ARM_JOINTS,
    PINGTI_EE_BODY,
    PINGTI_GRIPPER_JOINT,
    PINGTI_JOINTS,
    PINGTI_PALM_POS,
    SE3_ACTION_DIM,
    gripper_joint_target_from_se3,
)
from isaac_so_arm101.teleop_root import (
    authored_attrs_changed,
    pose7_pos_delta,
    pose7_quat_delta,
    should_write_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
URDF = REPO_ROOT / "src/isaac_so_arm101/robots/pingti/PingTi_Arm_5DOF_v4_copy.urdf"
TELEOP_CFG = REPO_ROOT / "src/isaac_so_arm101/tasks/teleop/teleop_env_cfg.py"


def _urdf_joint_names() -> list[str]:
    names: list[str] = []
    for line in URDF.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("<joint ") and "name=" in stripped:
            start = stripped.index('name="') + 6
            end = stripped.index('"', start)
            names.append(stripped[start:end])
    return names


def _urdf_has_link(name: str) -> bool:
    return f'<link name="{name}">' in URDF.read_text(encoding="utf-8")


class TeleopActionTests(unittest.TestCase):
    def test_se3_action_dim_is_7(self):
        self.assertEqual(SE3_ACTION_DIM, 7)

    def test_gripper_mapping_open_close(self):
        self.assertEqual(gripper_joint_target_from_se3(-1.0), GRIPPER_CLOSED_RAD)
        self.assertEqual(gripper_joint_target_from_se3(1.0), GRIPPER_OPEN_RAD)
        self.assertLess(GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD)

    def test_joint_and_body_names_match_urdf(self):
        urdf_joints = _urdf_joint_names()
        for joint in PINGTI_JOINTS:
            self.assertIn(joint, urdf_joints, msg=f"missing joint {joint} in URDF")
        self.assertTrue(_urdf_has_link(PINGTI_EE_BODY), msg=f"missing body {PINGTI_EE_BODY}")
        self.assertEqual(PINGTI_ARM_JOINTS, (
            "base_yaw",
            "shoulder_pitch",
            "elbow_pitch",
            "wrist_pitch",
            "wrist_roll",
        ))
        self.assertEqual(PINGTI_GRIPPER_JOINT, "gripper_moving")

    def test_teleop_env_source_has_relative_diffik_and_no_palm_usd(self):
        source = TELEOP_CFG.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIsNotNone(tree)
        self.assertIn("use_relative_mode=True", source)
        self.assertIn("GroundPlaneCfg", source)
        self.assertNotIn("palm_environment.usdc", source)
        self.assertIn("num_envs=1", source)
        self.assertIn("dls", source)
        keyboard = (REPO_ROOT / "src/isaac_so_arm101/scripts/teleop_keyboard.py").read_text(encoding="utf-8")
        self.assertIn('default="palm"', keyboard)
        self.assertIn('visualizer=["kit"]', keyboard)
        self.assertIn("PINGTI_PALM_POS", source)
        self.assertIn("KitRootSync", keyboard)
        self.assertIn("reason=usd_attr", keyboard)
        self.assertIn(
            "GetLocalTransformation",
            (REPO_ROOT / "src/isaac_so_arm101/teleop_root.py").read_text(encoding="utf-8"),
        )
        self.assertIn("Lab3 xyzw", keyboard)
        self.assertIn("IK singular", keyboard)
        self.assertIn("reason=hold", keyboard)
        self.assertIn("fix_base=False", TELEOP_CFG.read_text(encoding="utf-8"))
        self.assertIn("--teleop_device", keyboard)
        self.assertIn("so101leader", keyboard)
        self.assertIn("apply_teleop_device", keyboard)

    def test_palm_spawn_is_outside_legacy_canopy_xy(self):
        legacy = (0.30706, 0.49191, 4.65058)
        self.assertNotEqual(PINGTI_PALM_POS, legacy)
        self.assertAlmostEqual(PINGTI_PALM_POS[2], legacy[2], places=4)
        # Canopy world AABB Y max is ~1.50 m; sit just outside, not a full step back.
        self.assertGreater(PINGTI_PALM_POS[1], 1.45)
        self.assertLess(PINGTI_PALM_POS[1], 1.60)
        self.assertAlmostEqual(PINGTI_PALM_POS[0], 1.0, places=4)

    def test_root_sync_detects_usd_translate_change(self):
        ident = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        moved = (-0.04, 1.80, 4.65, 0.0, 0.0, 0.0, 1.0)
        same = (-0.04, 1.80, 4.65, 0.0, 0.0, 0.0, 1.0)
        self.assertGreater(pose7_pos_delta(ident, moved), 0.1)
        self.assertLess(pose7_quat_delta(moved, same), 1.0e-6)
        self.assertTrue(should_write_root(moved, ident))
        self.assertFalse(should_write_root(moved, same))

    def test_usd_attr_edge_ignores_physx_jitter(self):
        spawn_t = (-0.04, 1.52, 4.65058)
        spawn_q = (0.98325, 0.0, 0.0, 0.18224)
        panel_t = (0.0, 0.0, 0.0)
        ident_q = (1.0, 0.0, 0.0, 0.0)
        self.assertFalse(authored_attrs_changed(spawn_t, spawn_q, None, None))
        self.assertFalse(authored_attrs_changed(spawn_t, spawn_q, spawn_t, spawn_q))
        self.assertTrue(authored_attrs_changed(panel_t, spawn_q, spawn_t, spawn_q))
        self.assertTrue(authored_attrs_changed(spawn_t, ident_q, spawn_t, spawn_q))
        jitter_t = (-0.04, 1.52, 4.65059)
        self.assertFalse(authored_attrs_changed(jitter_t, spawn_q, spawn_t, spawn_q))

    def test_fetch_assets_documents_hf_auth_login(self):
        source = (REPO_ROOT / "src/isaac_so_arm101/scripts/fetch_assets.py").read_text(encoding="utf-8")
        self.assertIn("hf auth login", source)
        self.assertIn("huggingface-cli", source)
        self.assertIn("HF_LOGIN_HINT", source)
        self.assertIn("public dataset does not need a token", source)
        self.assertIn("token=False", source)


if __name__ == "__main__":
    unittest.main()
