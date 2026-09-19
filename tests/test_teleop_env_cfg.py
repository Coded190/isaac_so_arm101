"""Teleop env cfg construction (Isaac Lab, no simulator)."""

from __future__ import annotations

import os
import tempfile
import unittest

from isaac_so_arm101.assets import AssetNotFoundError, ENV_ASSETS, FETCH_HINT
from isaac_so_arm101.teleop_constants import PINGTI_EE_BODY, PINGTI_PALM_POS, PINGTI_TABLE_POS


class TeleopEnvCfgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from isaac_so_arm101.tasks.teleop.teleop_env_cfg import PingTiTeleopEnvCfg
        except Exception as exc:  # pragma: no cover - depends on Isaac Lab import rules
            raise unittest.SkipTest(f"isaaclab env cfg import requires kit: {exc}") from exc
        cls.PingTiTeleopEnvCfg = PingTiTeleopEnvCfg

    def test_procedural_teleop_cfg(self):
        cfg = self.PingTiTeleopEnvCfg()
        self.assertEqual(cfg.scene.num_envs, 1)
        self.assertTrue(cfg.actions.arm_action.controller.use_relative_mode)
        self.assertEqual(cfg.actions.arm_action.body_name, PINGTI_EE_BODY)
        self.assertEqual(cfg.scene.robot.init_state.pos, PINGTI_TABLE_POS)
        usd = getattr(getattr(cfg.scene, "custom_env", None), "spawn", None)
        self.assertIsNone(usd)
        self.assertTrue(hasattr(cfg.scene, "ground"))

    def test_palm_teleop_cfg_requires_assets(self):
        from isaac_so_arm101.tasks.teleop.teleop_env_cfg import PingTiPalmTeleopEnvCfg

        with tempfile.TemporaryDirectory() as tmp:
            os.environ[ENV_ASSETS] = tmp
            try:
                with self.assertRaises(AssetNotFoundError) as ctx:
                    PingTiPalmTeleopEnvCfg()
                self.assertIn(FETCH_HINT, str(ctx.exception))
            finally:
                os.environ.pop(ENV_ASSETS, None)

    def test_palm_teleop_cfg_uses_offset_spawn(self):
        from isaac_so_arm101.assets import find_scene_usd
        from isaac_so_arm101.tasks.teleop.teleop_env_cfg import PingTiPalmTeleopEnvCfg

        if find_scene_usd("palm_environment") is None:
            self.skipTest("palm scene pack not on disk")
        cfg = PingTiPalmTeleopEnvCfg()
        self.assertEqual(cfg.scene.robot.init_state.pos, PINGTI_PALM_POS)
        self.assertEqual(cfg.viewer.lookat, PINGTI_PALM_POS)
        self.assertFalse(cfg.scene.robot.spawn.fix_base)
        self.assertTrue(cfg.scene.robot.spawn.rigid_props.disable_gravity)

    def test_procedural_teleop_root_is_free(self):
        cfg = self.PingTiTeleopEnvCfg()
        self.assertFalse(cfg.scene.robot.spawn.fix_base)
        self.assertTrue(cfg.scene.robot.spawn.rigid_props.disable_gravity)


if __name__ == "__main__":
    unittest.main()
