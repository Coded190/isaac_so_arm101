"""Compatibility patches for transformers and Isaac Sim.

Keep this module small and import-safe; callers should invoke
`patch_transformers_for_isaac_sim()` before loading large transformer models
when running inside Isaac Sim containers or other constrained runtimes.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def patch_transformers_for_isaac_sim() -> None:
    """Apply minimal, safe patches to make newer `transformers` versions play
    nicely with Isaac Sim / containerized environments.

    The original repo includes a custom `_patch_transformers_attention_dispatch`
    helper; here we provide a no-op shim that preserves import-time behavior
    and can be expanded later if a concrete patch is required.
    """
    try:
        # lazy import to avoid pulling heavy deps at module import
        import transformers  # type: ignore
        log.debug("transformers available: %s", getattr(transformers, "__version__", "unknown"))
    except Exception:
        log.debug("transformers not available in this environment; skipping patches")
        return

    # If future compatibility code is needed, implement it here. For now
    # we intentionally avoid aggressive monkeypatching.
    log.debug("patch_transformers_for_isaac_sim: no-op patch applied")
