"""Palm garden prim-path helpers (no Isaac Sim)."""

from __future__ import annotations

import unittest

from isaac_so_arm101.scene_prims import (
    PALM_CROWN_CHILD,
    PALM_ROOT_NAME,
    dome_light_candidate_paths,
    dome_light_prim_path,
    palm_root_candidate_paths,
    palm_root_prim_path,
)


class ScenePrimPathTests(unittest.TestCase):
    def test_candidates_prefer_scene_root(self):
        self.assertEqual(PALM_ROOT_NAME, "palm_tree_crown")
        self.assertEqual(PALM_CROWN_CHILD, "crown")
        palm = palm_root_candidate_paths(0)
        self.assertEqual(palm[0], "/World/envs/env_0/Scene/palm_tree_crown")
        self.assertEqual(palm[1], "/World/envs/env_0/Scene/root/palm_tree_crown")
        self.assertTrue(all("/root/Palm" not in path for path in palm))
        lights = dome_light_candidate_paths(3)
        self.assertEqual(lights[0], "/World/envs/env_3/Scene/DomeLight")

    def test_default_paths_without_stage(self):
        self.assertEqual(palm_root_prim_path(2), "/World/envs/env_2/Scene/palm_tree_crown")
        self.assertEqual(dome_light_prim_path(2), "/World/envs/env_2/Scene/DomeLight")


if __name__ == "__main__":
    unittest.main()
