"""Repo-relative scene asset resolver.

Scene USDs are never stored in git. They live under ``assets/scenes/`` after
``uv run fetch_assets`` (or a local copy / ``ISAAC_SO_ARM101_ASSETS`` override).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ENV_ASSETS = "ISAAC_SO_ARM101_ASSETS"
DEFAULT_SCENE_NAME = "palm_environment"
MANIFEST_NAME = "manifest.json"

# Default garden stage (hierarchical palm_tree_crown payload). Do not list v2
# (flat /root/Palm, no crown child).
USD_CANDIDATES = ("palm_environment.usdc",)

PAYLOAD_CROWN_OLD = "../palm_tree_crown.usdc"
PAYLOAD_CROWN_NEW = "./palm_tree_crown.usdc"
COCONUT_TEXTURE_PREFIX_OLD = "../coconut palm/"
COCONUT_TEXTURE_PREFIX_NEW = "./coconut_palm_textures/"
FLAT_PALM_PRIM = "/root/Palm"
HIERARCHICAL_PALM_PRIM = "/root/palm_tree_crown"

FETCH_HINT = "uv run fetch_assets"


class AssetNotFoundError(FileNotFoundError):
    """Raised when a required scene pack is missing."""


def repo_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def manifest_path() -> Path:
    return repo_root() / "assets" / MANIFEST_NAME


def load_manifest() -> dict:
    path = manifest_path()
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_scenes_root() -> Path:
    return repo_root() / "assets" / "scenes"


def assets_root() -> Path:
    """Directory that contains per-scene packs.

    Search order:
      1. ``$ISAAC_SO_ARM101_ASSETS`` if set
      2. ``<repo>/assets/scenes``
    """
    override = os.environ.get(ENV_ASSETS, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return default_scenes_root()


def _scene_search_dirs(name: str) -> list[Path]:
    root = assets_root()
    dirs = [root / name, root]
    # Unique, existing-or-not (existence checked later).
    seen: set[Path] = set()
    out: list[Path] = []
    for path in dirs:
        resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def find_scene_usd(name: str = DEFAULT_SCENE_NAME) -> Path | None:
    """Return the first matching scene USD, or None if missing."""
    for directory in _scene_search_dirs(name):
        if not directory.is_dir():
            continue
        for candidate in USD_CANDIDATES:
            direct = directory / candidate
            if direct.is_file():
                return direct.resolve()
        # One extra level (copied blender folder).
        for candidate in USD_CANDIDATES:
            matches = list(directory.glob(f"*/{candidate}"))
            if matches:
                return matches[0].resolve()
    return None


def find_hdri_dir(name: str = DEFAULT_SCENE_NAME) -> Path | None:
    """Return the HDRI directory next to the scene USD, or a pack ``hdri/`` folder."""
    usd = find_scene_usd(name)
    if usd is not None:
        sibling = usd.parent / "hdri"
        if sibling.is_dir():
            return sibling.resolve()
    for directory in _scene_search_dirs(name):
        hdri = directory / "hdri"
        if hdri.is_dir():
            return hdri.resolve()
        nested = list(directory.glob("*/hdri"))
        if nested:
            return nested[0].resolve()
    return None


def _missing_message(kind: str, name: str) -> str:
    looked = "\n".join(f"  - {path}" for path in _scene_search_dirs(name))
    env_val = os.environ.get(ENV_ASSETS, "<unset>")
    return (
        f"[assets] {kind} for scene '{name}' not found.\n"
        f"ISAAC_SO_ARM101_ASSETS={env_val}\n"
        f"Looked in:\n{looked}\n"
        f"Copy the full scene pack (USD + textures + hdri/) into "
        f"{default_scenes_root() / name} or run: {FETCH_HINT}"
    )


LAB_ABSOLUTE_PREFIXES = ("/home/cirp-lab", "/home/cirplab")


def require_scene_usd(name: str = DEFAULT_SCENE_NAME) -> Path:
    """Return the scene USD path or raise with fetch instructions."""
    path = find_scene_usd(name)
    if path is None:
        raise AssetNotFoundError(_missing_message("USD", name))
    if path.name != "palm_environment.usdc" and name == DEFAULT_SCENE_NAME:
        raise AssetNotFoundError(
            f"[assets] expected palm_environment.usdc for scene '{name}', got {path}. "
            "v2 (flat /root/Palm) is not a valid default."
        )
    print(f"[assets] usd={path}", flush=True)
    return path


def resolved_usd_reference(usd_path: Path, asset_path: str) -> Path:
    """Resolve a USD ``@...@`` asset path relative to ``usd_path``'s layer."""
    return (usd_path.parent / asset_path).resolve()


def missing_usd_references(usd_path: Path) -> list[tuple[str, Path]]:
    """Return ``(asset_path, resolved)`` for referenced files that are not on disk.

    Texture inputs (``./textures/*.png``, ``./textures/*.exr``) and the garden
    companion ``../background_3d_objects/simpler_world.usd`` are both checked.
    Missing companions produce gray GeomSubset materials and drop picnic tables.
    """
    try:
        from pxr import Sdf
    except ImportError:
        return []

    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    if layer is None:
        return []

    missing: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def _consider(asset_path: str) -> None:
        if not asset_path or asset_path in seen:
            return
        seen.add(asset_path)
        if any(asset_path.startswith(prefix) for prefix in LAB_ABSOLUTE_PREFIXES):
            return
        # Isaac built-in MDL names (OmniPBR.mdl) are not pack files.
        normalized = asset_path.replace("\\", "/")
        if normalized.endswith(".mdl") and "/" not in normalized:
            return
        if asset_path.startswith(("omniverse://", "http://", "https://")):
            return
        resolved = resolved_usd_reference(usd_path, asset_path)
        if not resolved.is_file():
            missing.append((asset_path, resolved))

    for asset_path in layer.GetExternalReferences():
        _consider(asset_path)

    def _walk_attrs(spec) -> None:
        for prop in spec.properties:
            if type(prop).__name__ != "AttributeSpec":
                continue
            value = prop.default
            path_str = getattr(value, "path", None)
            if path_str:
                _consider(path_str)
        for child in spec.nameChildren:
            _walk_attrs(child)

    for root_spec in layer.rootPrims.values():
        _walk_attrs(root_spec)
    return missing


def report_scene_pack_gaps(usd_path: Path) -> list[str]:
    """Human-readable ``[assets]`` lines for unresolved scene references."""
    lines: list[str] = []
    missing = list(missing_usd_references(usd_path))
    payload = usd_path.parent / "palm_tree_crown.usdc"
    if payload.is_file() and payload.resolve() != usd_path.resolve():
        missing.extend(missing_usd_references(payload))
    if not missing:
        lines.append(f"[assets] scene pack complete usd={usd_path}")
        return lines
    lines.append(f"[assets] scene pack incomplete usd={usd_path}")
    for asset_path, resolved in missing:
        lines.append(f"[assets] missing reference {asset_path} -> {resolved}")
        if asset_path.endswith("simpler_world.usd"):
            dest = resolved.parent
            lines.append(
                "[assets] copy the full background_3d_objects folder "
                f"(USD + textures) to {dest}; see assets/README.md, then restart teleop"
            )
        if asset_path.endswith("palm_tree_crown.usdc"):
            lines.append(
                "[assets] palm_tree_crown.usdc must sit next to palm_environment.usdc "
                f"(expected {resolved})"
            )
        if asset_path.endswith("color_0C0C0C.exr"):
            lines.append(f"[assets] crown env_light EXR missing; copy to {resolved}")
    return lines


def require_hdri_dir(name: str = DEFAULT_SCENE_NAME) -> Path:
    """Return the HDRI directory or raise with fetch instructions."""
    path = find_hdri_dir(name)
    if path is None:
        raise AssetNotFoundError(_missing_message("HDRI folder", name))
    return path


def rewrite_lab_absolute_usd_refs(usd_path: Path) -> list[str]:
    """Drop lab-machine USD references in a scene file.

    Texture/EXR inputs in the palm pack are already ``./textures/...`` relative to
    the USD. The remaining lab-absolute path is an embedded PingTi USD reference;
    Lab teleop spawns PingTi from the local URDF, so that reference is cleared.
    """
    try:
        from pxr import Sdf
    except ImportError as exc:
        raise RuntimeError(
            "[assets] rewrite_lab_absolute_usd_refs requires pxr (Isaac Sim USD)."
        ) from exc

    usd_path = usd_path.resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)

    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    if layer is None:
        raise RuntimeError(f"[assets] could not open USD layer: {usd_path}")

    changed: list[str] = []

    def _walk(spec: Sdf.PrimSpec) -> None:
        if spec.HasInfo("references"):
            kept: list = []
            dropped = False
            for item in spec.referenceList.GetAddedOrExplicitItems():
                asset = str(item.assetPath)
                if any(asset.startswith(prefix) for prefix in LAB_ABSOLUTE_PREFIXES):
                    changed.append(f"{spec.path}: drop {asset}")
                    dropped = True
                    continue
                kept.append(item)
            if dropped:
                spec.referenceList.ClearEdits()
                for item in kept:
                    spec.referenceList.Prepend(item)
        for child in spec.nameChildren:
            _walk(child)

    for root_spec in layer.rootPrims.values():
        _walk(root_spec)

    if changed:
        if not layer.Save():
            raise RuntimeError(f"[assets] failed to save rewritten USD: {usd_path}")

    leftover = [
        ref
        for ref in layer.GetExternalReferences()
        if any(ref.startswith(prefix) for prefix in LAB_ABSOLUTE_PREFIXES)
    ]
    if leftover:
        raise RuntimeError(f"[assets] still have lab-absolute USD refs after rewrite: {leftover}")
    return changed


def _rewrite_pack_asset_string(asset: str) -> str | None:
    """Return a rewritten relative pack path, or None if unchanged."""
    if not asset:
        return None
    if asset == PAYLOAD_CROWN_OLD or (
        asset.endswith("palm_tree_crown.usdc") and asset.startswith("../")
    ):
        return PAYLOAD_CROWN_NEW
    if asset.startswith(COCONUT_TEXTURE_PREFIX_OLD):
        return COCONUT_TEXTURE_PREFIX_NEW + asset[len(COCONUT_TEXTURE_PREFIX_OLD) :]
    return None


def _set_spec_payloads(spec, items) -> None:
    spec.payloadList.ClearEdits()
    for item in items:
        spec.payloadList.Prepend(item)


def _set_spec_references(spec, items) -> None:
    spec.referenceList.ClearEdits()
    for item in items:
        spec.referenceList.Prepend(item)


def _rewrite_composition_list(spec, list_attr: str, factory) -> list[str]:
    changed: list[str] = []
    items_attr = spec.payloadList if list_attr == "payloads" else spec.referenceList
    items = list(items_attr.prependedItems) or list(items_attr.explicitItems)
    if not items:
        return changed
    rewritten = []
    dirty = False
    for item in items:
        old = str(item.assetPath)
        new = _rewrite_pack_asset_string(old)
        if new is None:
            rewritten.append(item)
            continue
        dirty = True
        changed.append(f"{spec.path}: {old} -> {new}")
        rewritten.append(factory(new, item.primPath, item.layerOffset))
    if dirty:
        if list_attr == "payloads":
            _set_spec_payloads(spec, rewritten)
        else:
            _set_spec_references(spec, rewritten)
    return changed


def rewrite_relative_pack_arcs(usd_path: Path) -> list[str]:
    """Fix pack-relative payloads and coconut texture paths after the crown moved in-folder."""
    try:
        from pxr import Sdf
    except ImportError as exc:
        raise RuntimeError(
            "[assets] rewrite_relative_pack_arcs requires pxr (Isaac Sim USD)."
        ) from exc

    usd_path = usd_path.resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)

    layer = Sdf.Layer.FindOrOpen(str(usd_path))
    if layer is None:
        raise RuntimeError(f"[assets] could not open USD layer: {usd_path}")

    changed: list[str] = []

    def _rewrite_value(value):
        if isinstance(value, Sdf.AssetPath):
            new = _rewrite_pack_asset_string(value.path)
            if new is None:
                return value, False
            return Sdf.AssetPath(new), True
        if isinstance(value, str):
            new = _rewrite_pack_asset_string(value)
            if new is None:
                return value, False
            return new, True
        return value, False

    def _walk(spec: Sdf.PrimSpec) -> None:
        changed.extend(
            _rewrite_composition_list(
                spec, "payloads", lambda path, prim, offset: Sdf.Payload(path, prim, offset)
            )
        )
        if spec.HasInfo("references"):
            changed.extend(
                _rewrite_composition_list(
                    spec, "references", lambda path, prim, offset: Sdf.Reference(path, prim, offset)
                )
            )
        for prop in spec.properties:
            if not isinstance(prop, Sdf.AttributeSpec):
                continue
            new_default, dirty = _rewrite_value(prop.default)
            if dirty:
                changed.append(f"{spec.path}.{prop.name}: {prop.default} -> {new_default}")
                prop.default = new_default
        for child in spec.nameChildren:
            _walk(child)

    for root_spec in layer.rootPrims.values():
        _walk(root_spec)

    if PAYLOAD_CROWN_OLD in layer.GetExternalReferences():
        layer.UpdateExternalReference(PAYLOAD_CROWN_OLD, PAYLOAD_CROWN_NEW)
        changed.append(f"layer.UpdateExternalReference {PAYLOAD_CROWN_OLD} -> {PAYLOAD_CROWN_NEW}")

    if changed:
        if not layer.Save():
            raise RuntimeError(f"[assets] failed to save rewritten USD: {usd_path}")
    return changed


def deactivate_flat_palm(usd_path: Path) -> list[str]:
    """Deactivate leftover ``/root/Palm`` when the hierarchical payload is present."""
    try:
        from pxr import Usd
    except ImportError as exc:
        raise RuntimeError("[assets] deactivate_flat_palm requires pxr (Isaac Sim USD).") from exc

    usd_path = usd_path.resolve()
    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"[assets] could not open USD stage: {usd_path}")
    changed: list[str] = []
    palm = stage.GetPrimAtPath(FLAT_PALM_PRIM)
    crown = stage.GetPrimAtPath(HIERARCHICAL_PALM_PRIM)
    if palm and palm.IsValid() and palm.IsActive() and crown and crown.IsValid():
        palm.SetActive(False)
        changed.append(f"{FLAT_PALM_PRIM} deactivated (hierarchical {HIERARCHICAL_PALM_PRIM} present)")
        if not stage.GetRootLayer().Save():
            raise RuntimeError(f"[assets] failed to save deactivated Palm: {usd_path}")
    return changed


def prepare_palm_environment_usd(usd_path: Path) -> list[str]:
    """Rewrite pack arcs and hide the leftover flat Palm on the garden stage."""
    changed = rewrite_lab_absolute_usd_refs(usd_path)
    changed.extend(rewrite_relative_pack_arcs(usd_path))
    changed.extend(deactivate_flat_palm(usd_path))
    return changed


def hash_directory(directory: Path) -> str:
    """Stable SHA256 over relative paths and file contents (chunked)."""
    digest = hashlib.sha256()
    if not directory.is_dir():
        raise FileNotFoundError(f"Cannot hash missing directory: {directory}")
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    for file_path in files:
        rel = file_path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        hasher = hashlib.sha256()
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        digest.update(hasher.digest())
    return digest.hexdigest()
