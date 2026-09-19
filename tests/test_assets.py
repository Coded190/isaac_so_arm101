"""Asset resolver tests (no Isaac Sim)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from isaac_so_arm101.assets import (
    FETCH_HINT,
    AssetNotFoundError,
    ENV_ASSETS,
    LAB_ABSOLUTE_PREFIXES,
    find_scene_usd,
    hash_directory,
    missing_usd_references,
    report_scene_pack_gaps,
    require_hdri_dir,
    require_scene_usd,
)

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


class AssetResolverTests(unittest.TestCase):
    def test_src_has_no_lab_absolute_paths(self):
        banned = ("/home/cirplab", "/home/cirp-lab")
        offenders: list[str] = []
        for path in SRC_ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                # Prefix matchers in the rewriter are allowed; they are not runtime load paths.
                if "LAB_ABSOLUTE_PREFIXES" in stripped or stripped.startswith("LAB_ABSOLUTE_PREFIXES"):
                    continue
                for token in banned:
                    if token in stripped:
                        offenders.append(f"{path.relative_to(SRC_ROOT.parent)}:{i}: {token}")
        self.assertEqual(offenders, [], msg="lab absolute paths still in runtime Python")

    def test_missing_pack_error_mentions_fetch_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[ENV_ASSETS] = tmp
            try:
                with self.assertRaises(AssetNotFoundError) as ctx:
                    require_scene_usd("palm_environment")
                message = str(ctx.exception)
                self.assertIn(FETCH_HINT, message)
                self.assertIn("[assets]", message)
                self.assertIsNone(find_scene_usd("palm_environment"))
            finally:
                os.environ.pop(ENV_ASSETS, None)

    def test_usd_candidates_are_garden_only(self):
        from isaac_so_arm101.assets import USD_CANDIDATES

        self.assertEqual(USD_CANDIDATES, ("palm_environment.usdc",))
        self.assertNotIn("pretoria_gardens_4k_env_v2.usdc", USD_CANDIDATES)

    def test_resolver_ignores_v2_when_garden_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "palm_environment"
            scene.mkdir()
            (scene / "pretoria_gardens_4k_env_v2.usdc").write_bytes(b"v2")
            os.environ[ENV_ASSETS] = tmp
            try:
                self.assertIsNone(find_scene_usd("palm_environment"))
            finally:
                os.environ.pop(ENV_ASSETS, None)

    def test_local_garden_payload_is_in_folder(self):
        usd = find_scene_usd("palm_environment")
        if usd is None:
            self.skipTest("palm scene pack not on disk")
        self.assertEqual(usd.name, "palm_environment.usdc")
        crown = usd.parent / "palm_tree_crown.usdc"
        self.assertTrue(crown.is_file(), msg=f"missing payload {crown}")
        fill = usd.parent / "textures" / "color_0C0C0C.exr"
        self.assertTrue(fill.is_file(), msg=f"missing crown env_light {fill}")
        try:
            from pxr import Sdf
        except ImportError:
            self.skipTest("pxr not installed")
        layer = Sdf.Layer.FindOrOpen(str(usd))
        refs = list(layer.GetExternalReferences())
        self.assertTrue(
            any(ref in {"./palm_tree_crown.usdc", "palm_tree_crown.usdc"} or ref.endswith("/palm_tree_crown.usdc") and not ref.startswith("../") for ref in refs),
            msg=f"payload should be ./palm_tree_crown.usdc; got {refs}",
        )
        self.assertFalse(any(ref == "../palm_tree_crown.usdc" for ref in refs), msg=refs)
        coconut = [ref for ref in refs if "coconut palm" in ref]
        self.assertEqual(coconut, [], msg=f"leftover coconut palm paths: {coconut}")

    def test_resolver_finds_local_usd_and_hdri(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "palm_environment"
            scene.mkdir()
            usd = scene / "palm_environment.usdc"
            usd.write_bytes(b"usda-placeholder")
            hdri = scene / "hdri"
            hdri.mkdir()
            (hdri / "sky.hdr").write_bytes(b"hdr")
            os.environ[ENV_ASSETS] = tmp
            try:
                self.assertEqual(require_scene_usd().resolve(), usd.resolve())
                self.assertEqual(require_hdri_dir().resolve(), hdri.resolve())
                digest = hash_directory(scene)
                self.assertEqual(len(digest), 64)
            finally:
                os.environ.pop(ENV_ASSETS, None)

    def test_local_palm_usd_has_no_lab_absolute_refs(self):
        usd = find_scene_usd("palm_environment")
        if usd is None:
            self.skipTest("palm scene pack not on disk")
        try:
            from pxr import Sdf
        except ImportError:
            self.skipTest("pxr not installed")
        layer = Sdf.Layer.FindOrOpen(str(usd))
        self.assertIsNotNone(layer)
        leftover = [
            ref
            for ref in layer.GetExternalReferences()
            if any(ref.startswith(prefix) for prefix in LAB_ABSOLUTE_PREFIXES)
        ]
        self.assertEqual(leftover, [])

    def test_local_palm_pack_reports_missing_simpler_world(self):
        usd = find_scene_usd("palm_environment")
        if usd is None:
            self.skipTest("palm scene pack not on disk")
        try:
            from pxr import Sdf  # noqa: F401
        except ImportError:
            self.skipTest("pxr not installed")
        refs = list(Sdf.Layer.FindOrOpen(str(usd)).GetExternalReferences())
        self.assertTrue(
            any(ref.endswith("simpler_world.usd") for ref in refs),
            msg=f"palm USD should reference simpler_world; got {refs}",
        )
        missing = missing_usd_references(usd)
        companion = usd.parent.parent / "background_3d_objects" / "simpler_world.usd"
        if companion.is_file():
            self.assertFalse(
                any(path == companion.resolve() for _, path in missing),
                msg="simpler_world.usd is on disk but still reported missing",
            )
            return
        self.assertTrue(
            any(path == companion.resolve() for _, path in missing),
            msg=f"expected missing companion {companion}; got {missing}",
        )
        report = "\n".join(report_scene_pack_gaps(usd))
        self.assertIn("[assets] missing reference", report)
        self.assertIn("simpler_world.usd", report)
        self.assertIn("background_3d_objects", report)


if __name__ == "__main__":
    unittest.main()
