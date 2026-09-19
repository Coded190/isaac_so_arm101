"""Live USD prim paths for the palm garden after Lab spawn.

Reach / VLA / teleop spawn ``palm_environment.usdc`` at ``{ENV}/Scene``.
That file's defaultPrim is ``root``, so the hierarchical tree is typically
``Scene/root/palm_tree_crown`` (child ``crown``). Older scripts looked at
``Scene/palm_tree_crown`` or ``env_{id}/palm_tree_crown`` without ``Scene``.
"""

from __future__ import annotations

PALM_ROOT_NAME = "palm_tree_crown"
PALM_CROWN_CHILD = "crown"
DOME_LIGHT_NAME = "DomeLight"


def palm_root_candidate_paths(env_id: int) -> tuple[str, ...]:
    """Likely Lab instance paths for the hierarchical palm (no ``/root/Palm``).

    Kit spawn of ``palm_environment.usdc`` at ``{ENV}/Scene`` places the tree at
    ``Scene/palm_tree_crown`` (verified on Lab 3 / Sim 6.1). ``Scene/root/...``
    is kept as a fallback if a flatten changes.
    """
    prefix = f"/World/envs/env_{env_id}/Scene"
    return (
        f"{prefix}/{PALM_ROOT_NAME}",
        f"{prefix}/root/{PALM_ROOT_NAME}",
    )


def dome_light_candidate_paths(env_id: int) -> tuple[str, ...]:
    prefix = f"/World/envs/env_{env_id}/Scene"
    return (
        f"{prefix}/{DOME_LIGHT_NAME}",
        f"{prefix}/root/{DOME_LIGHT_NAME}",
    )


def _first_valid_prim(stage, paths: tuple[str, ...], *, child: str | None = None):
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim is None or not prim.IsValid():
            continue
        if child:
            nested = prim.GetChild(child)
            if nested is None or not nested.IsValid():
                continue
        return prim, path
    return None, None


def find_prim_at_candidates(stage, paths: tuple[str, ...], *, child: str | None = None):
    """Return ``(prim, path)`` for the first valid candidate, else ``(None, None)``."""
    return _first_valid_prim(stage, paths, child=child)


def find_palm_root_prim(stage, env_id: int):
    """Return ``(prim, path)`` for the hierarchical palm, preferring a ``crown`` child."""
    candidates = palm_root_candidate_paths(env_id)
    prim, path = _first_valid_prim(stage, candidates, child=PALM_CROWN_CHILD)
    if path is not None:
        return prim, path
    return _first_valid_prim(stage, candidates)


def find_dome_light_prim(stage, env_id: int):
    return _first_valid_prim(stage, dome_light_candidate_paths(env_id))


def palm_root_prim_path(env_id: int, stage=None) -> str:
    """Return the env-local hierarchical palm path.

    When ``stage`` is given, pick the first candidate that exists and has
    child ``crown``. With no stage, prefer ``Scene/palm_tree_crown`` (Lab 3 spawn).
    """
    candidates = palm_root_candidate_paths(env_id)
    if stage is None:
        return candidates[0]
    _, path = find_palm_root_prim(stage, env_id)
    if path is None:
        raise RuntimeError(
            f"[assets] palm_tree_crown (child {PALM_CROWN_CHILD}) not found for env {env_id}; "
            f"tried {candidates}"
        )
    return path


def dome_light_prim_path(env_id: int, stage=None) -> str:
    """Return the env-local garden DomeLight path (Pretoria HDRI, not picnic env_light)."""
    candidates = dome_light_candidate_paths(env_id)
    if stage is None:
        return candidates[0]
    prim, path = find_dome_light_prim(stage, env_id)
    if path is None:
        raise RuntimeError(
            f"[assets] DomeLight not found for env {env_id}; tried {candidates}"
        )
    if prim is not None:
        try:
            from pxr import UsdLux

            if not prim.IsA(UsdLux.DomeLight):
                raise RuntimeError(f"[assets] {path} exists but is not a DomeLight")
        except ImportError:
            pass
    return path
