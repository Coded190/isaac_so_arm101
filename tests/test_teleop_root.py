"""Unit tests for USD → PhysX root pose conversion (no Isaac Sim)."""

from __future__ import annotations

import unittest

from isaac_so_arm101.teleop_constants import PINGTI_PALM_POS, PINGTI_PALM_ROT
from isaac_so_arm101.teleop_root import (
    authored_attrs_changed,
    next_root_write_reason,
    pose7_from_translate_orient_wxyz,
    pose7_is_sane,
    pose7_quat_delta,
    quat_mul_xyzw,
    quat_rotate_xyzw,
    scale_is_sane,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class TeleopRootPoseTests(unittest.TestCase):
    def test_lab3_palm_spawn_is_identity_xyzw(self):
        # Euler XYZ 0,0,0. Lab 3 rot is (x, y, z, w); USD Gf orient is (w, x, y, z).
        self.assertEqual(PINGTI_PALM_ROT, (0.0, 0.0, 0.0, 1.0))
        usd_gf_wxyz = xyzw_to_wxyz(PINGTI_PALM_ROT)
        self.assertEqual(usd_gf_wxyz, (1.0, 0.0, 0.0, 0.0))
        pose = pose7_from_translate_orient_wxyz(PINGTI_PALM_POS, usd_gf_wxyz)
        self.assertAlmostEqual(pose[0], PINGTI_PALM_POS[0], places=6)
        self.assertAlmostEqual(pose[1], PINGTI_PALM_POS[1], places=6)
        self.assertAlmostEqual(pose[2], PINGTI_PALM_POS[2], places=6)
        self.assertAlmostEqual(pose[3], 0.0, places=6)
        self.assertAlmostEqual(pose[6], 1.0, places=6)

    def test_wxyz_xyzw_roundtrip(self):
        wxyz = (0.16505, 0.98629, 0.0, 0.0)
        xyzw = wxyz_to_xyzw(wxyz)
        self.assertEqual(xyzw, (0.98629, 0.0, 0.0, 0.16505))
        self.assertEqual(xyzw_to_wxyz(xyzw), wxyz)

    def test_orient_drag_writes_panel_quat_not_spawn_quat(self):
        spawn_wxyz = (0.18224, 0.98325, 0.0, 0.0)
        dragged_wxyz = (0.16505, 0.98629, 0.0, 0.0)
        spawn_pose = pose7_from_translate_orient_wxyz(PINGTI_PALM_POS, spawn_wxyz)
        dragged = pose7_from_translate_orient_wxyz(PINGTI_PALM_POS, dragged_wxyz)
        self.assertGreater(pose7_quat_delta(spawn_pose, dragged), 1.0e-4)
        self.assertAlmostEqual(dragged[3], 0.98629, places=5)
        self.assertAlmostEqual(dragged[6], 0.16505, places=5)
        self.assertTrue(pose7_is_sane(dragged))

    def test_identity_parent_keeps_local_translate(self):
        local_t = (-0.04, 1.52, 4.65)
        ident = (1.0, 0.0, 0.0, 0.0)
        pose = pose7_from_translate_orient_wxyz(local_t, ident)
        self.assertEqual(pose[:3], local_t)
        self.assertAlmostEqual(pose[6], 1.0, places=6)

    def test_parent_translate_composes(self):
        pose = pose7_from_translate_orient_wxyz(
            (0.0, 2.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            parent_t=(1.0, 0.0, 0.0),
        )
        self.assertAlmostEqual(pose[0], 1.0, places=6)
        self.assertAlmostEqual(pose[1], 2.0, places=6)
        self.assertAlmostEqual(pose[2], 0.0, places=6)

    def test_quat_mul_identity(self):
        q = (0.0, 0.0, 0.18224, 0.98325)
        ident = (0.0, 0.0, 0.0, 1.0)
        prod = quat_mul_xyzw(ident, q)
        for a, b in zip(prod, q, strict=True):
            self.assertAlmostEqual(a, b, places=6)

    def test_quat_rotate_identity(self):
        v = quat_rotate_xyzw((0.0, 0.0, 0.0, 1.0), (1.0, 2.0, 3.0))
        self.assertAlmostEqual(v[0], 1.0, places=6)
        self.assertAlmostEqual(v[1], 2.0, places=6)
        self.assertAlmostEqual(v[2], 3.0, places=6)

    def test_rejects_exploded_physx_pose(self):
        exploded = (1.8e10, -3.6e10, 5.1e10, 0.0, 0.0, 0.0, 1.0)
        self.assertFalse(pose7_is_sane(exploded))
        self.assertTrue(pose7_is_sane((*PINGTI_PALM_POS, *PINGTI_PALM_ROT)))

    def test_written_q_wxyz_matches_panel_orient(self):
        # Logs print both as Gf WXYZ so written_q must equal usd_orient after convert.
        panel = (0.16505, 0.98629, 0.0, 0.0)
        pose = pose7_from_translate_orient_wxyz((-0.04, 1.52, 4.65), panel)
        written_wxyz = xyzw_to_wxyz(pose[3:])
        for a, b in zip(written_wxyz, panel, strict=True):
            self.assertAlmostEqual(a, b, places=5)

    def test_hold_keeps_sane_panel_pose_when_gravity_would_drop_root(self):
        self.assertEqual(
            next_root_write_reason(
                has_authored=True, latched=False, usd_changed=False, authored_sane=True
            ),
            "latch",
        )
        self.assertEqual(
            next_root_write_reason(
                has_authored=True, latched=True, usd_changed=True, authored_sane=True
            ),
            "usd_attr",
        )
        self.assertEqual(
            next_root_write_reason(
                has_authored=True, latched=True, usd_changed=False, authored_sane=True
            ),
            "hold",
        )
        self.assertEqual(
            next_root_write_reason(
                has_authored=True,
                latched=True,
                usd_changed=False,
                authored_sane=True,
                hold=False,
            ),
            "idle",
        )
        self.assertEqual(
            next_root_write_reason(
                has_authored=True, latched=True, usd_changed=False, authored_sane=False
            ),
            "rejected_pose",
        )

    def test_scale_sanity_and_attr_edge(self):
        self.assertTrue(scale_is_sane((1.0, 1.0, 1.0)))
        self.assertTrue(scale_is_sane((2.0, 0.5, 1.0)))
        self.assertFalse(scale_is_sane((0.0, 1.0, 1.0)))
        self.assertFalse(scale_is_sane((-1.0, 1.0, 1.0)))
        spawn_t = PINGTI_PALM_POS
        spawn_q = (0.18224, 0.98325, 0.0, 0.0)
        ident_s = (1.0, 1.0, 1.0)
        self.assertFalse(
            authored_attrs_changed(
                spawn_t, spawn_q, spawn_t, spawn_q, scale=ident_s, prev_scale=ident_s
            )
        )
        self.assertTrue(
            authored_attrs_changed(
                spawn_t, spawn_q, spawn_t, spawn_q, scale=(2.0, 2.0, 2.0), prev_scale=ident_s
            )
        )


if __name__ == "__main__":
    unittest.main()
