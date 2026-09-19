"""Download (or upload) the PingTi scene pack from Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from isaac_so_arm101.assets import (
    DEFAULT_SCENE_NAME,
    FETCH_HINT,
    AssetNotFoundError,
    assets_root,
    default_scenes_root,
    find_hdri_dir,
    find_scene_usd,
    hash_directory,
    load_manifest,
    manifest_path,
    prepare_palm_environment_usd,
    report_scene_pack_gaps,
    repo_root,
    require_hdri_dir,
    require_scene_usd,
)

# huggingface_hub 0.36 ships `hf` and `huggingface-cli` inside the venv, not on PATH.
HF_LOGIN_HINT = (
    "`huggingface-cli` is not on the system PATH. Login from the Sim 6.1 venv:\n"
    "  UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact hf auth login\n"
    "or:\n"
    "  UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact huggingface-cli login\n"
    "or:\n"
    "  UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact "
    "python -c \"from huggingface_hub import login; login()\"\n"
    "Then: UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact fetch_assets --upload\n"
    "git credential.helper is not required: fetch_assets uses huggingface_hub "
    "(~/.cache/huggingface/token), not git push.\n"
    "Download of the public dataset does not need a token. Upload (write) does."
)


def _exit_hub_error(prefix: str, exc: Exception) -> None:
    text = str(exc)
    auth_fail = "401" in text or "Unauthorized" in text or "401 Client Error" in text
    extra = f"\n{HF_LOGIN_HINT}" if auth_fail else ""
    raise SystemExit(f"{prefix}: {exc}{extra}") from exc


def _print_resolved(scene: str) -> None:
    usd = find_scene_usd(scene)
    hdri = find_hdri_dir(scene)
    print(f"[assets] scenes root: {assets_root()}")
    print(f"[assets] usd:  {usd if usd else '(missing)'}")
    print(f"[assets] hdri: {hdri if hdri else '(missing)'}")


def _download(manifest: dict, local_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "[assets] huggingface_hub is required to download scene packs. "
            "It should be installed with this project (`uv sync`)."
        ) from exc

    repo_id = manifest.get("hf_repo_id")
    if not repo_id:
        raise SystemExit("[assets] manifest.json is missing hf_repo_id")
    repo_type = manifest.get("hf_repo_type", "dataset")
    revision = manifest.get("revision", "main")
    local_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[assets] downloading {repo_id}@{revision} -> {local_dir}",
        flush=True,
    )
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=str(local_dir),
            token=False,
        )
    except Exception as exc:  # noqa: BLE001 — Hub errors vary by version
        _exit_hub_error(
            f"[assets] download failed for {repo_id}. "
            "If the Hub dataset is not published yet, copy the lab pack into "
            f"{default_scenes_root() / DEFAULT_SCENE_NAME} then run "
            "`uv run fetch_assets --upload --source <pack_dir>`",
            exc,
        )
    if find_scene_usd(DEFAULT_SCENE_NAME) is None:
        raise SystemExit(
            f"[assets] downloaded {repo_id} but no scene USD was found under {local_dir}.\n"
            "Publish the pack with `uv run fetch_assets --upload --source <pack_dir>` "
            "after copying the full tree from the lab machine."
        )


def _verify(manifest: dict, scene: str, pack_dir: Path) -> None:
    expected = manifest.get("sha256") or manifest.get("scenes", {}).get(scene, {}).get("sha256")
    if not expected:
        print("[assets] no sha256 in manifest; skipping verify", flush=True)
        return
    # Prefer hashing the full scenes tree (palm_environment + background_3d_objects).
    target = pack_dir.parent if pack_dir.name == scene and pack_dir.parent.is_dir() else pack_dir
    digest = hash_directory(target)
    if digest != expected:
        raise SystemExit(
            f"[assets] sha256 mismatch for {target}\n"
            f"  expected: {expected}\n"
            f"  actual:   {digest}"
        )
    print(f"[assets] sha256 ok: {digest}", flush=True)


def _write_dataset_card(dest: Path) -> None:
    dest.write_text(
        """---
tags:
  - robotics
  - isaac-sim
  - usd
---

# isaac-so-arm101 scene pack

USD + HDRI assets for PingTi Reach / VLA / teleop.

Default stage is ``palm_environment/palm_environment.usdc`` (payload
``palm_tree_crown.usdc``). Companion picnic pack is ``background_3d_objects/``.

```bash
uv run fetch_assets
uv run --inexact teleop --scene palm --viz kit
```

Do not mix this pack with LeRobot episode datasets (`pingti_palm_tree`).
""",
        encoding="utf-8",
    )


def _upload(manifest: dict, source: Path) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("[assets] huggingface_hub is required to upload.") from exc

    if not source.is_dir():
        raise SystemExit(f"[assets] --source is not a directory: {source}")

    repo_id = manifest.get("hf_repo_id")
    repo_type = manifest.get("hf_repo_type", "dataset")
    dest = default_scenes_root()
    dest.mkdir(parents=True, exist_ok=True)
    source = source.expanduser().resolve()
    dest = dest.resolve()
    if source != dest:
        print(f"[assets] staging {source} -> {dest}", flush=True)
        for item in source.iterdir():
            if item.name.startswith("."):
                continue
            target = dest / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    else:
        print(f"[assets] uploading already-staged pack {dest}", flush=True)

    digest = hash_directory(dest)
    manifest["sha256"] = digest
    scenes = manifest.setdefault("scenes", {})
    scene_meta = scenes.setdefault(DEFAULT_SCENE_NAME, {})
    scene_meta["usd_candidates"] = ["palm_environment.usdc"]
    scene_meta["hdri_dir"] = "palm_environment/hdri"
    scene_meta["sha256"] = digest
    manifest_path().write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[assets] recorded sha256={digest} in {manifest_path()}", flush=True)

    card = dest / "README.md"
    if not card.is_file():
        _write_dataset_card(card)

    api = HfApi()
    print(f"[assets] creating/updating Hub repo {repo_id}", flush=True)
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=False)
        try:
            api.update_repo_visibility(repo_id=repo_id, repo_type=repo_type, private=False)
        except Exception as vis_exc:  # noqa: BLE001
            print(f"[assets] visibility update skipped: {vis_exc}", flush=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(dest),
            path_in_repo=".",
        )
    except Exception as exc:  # noqa: BLE001 — Hub errors vary by version
        _exit_hub_error(f"[assets] upload failed for {repo_id}", exc)
    print(f"[assets] uploaded https://huggingface.co/datasets/{repo_id}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch (or upload) PingTi scene USDs from Hugging Face Hub."
    )
    parser.add_argument(
        "--scene",
        default=DEFAULT_SCENE_NAME,
        help="Scene pack name (default: palm_environment).",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Stage --source and push to the Hub dataset in assets/manifest.json.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Local pack directory to stage/upload (default: assets/scenes).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only resolve/verify a pack that is already on disk.",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest()
    local_dir = repo_root() / manifest.get("local_dir", "assets/scenes")

    if args.upload:
        source = args.source if args.source is not None else default_scenes_root()
        _upload(manifest, source.expanduser().resolve())
        _print_resolved(args.scene)
        return 0

    if not args.skip_download and find_scene_usd(args.scene) is None:
        _download(manifest, local_dir)

    pack_dir = local_dir / args.scene
    if pack_dir.is_dir():
        _verify(manifest, args.scene, pack_dir)

    try:
        usd = require_scene_usd(args.scene)
    except AssetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        dropped = prepare_palm_environment_usd(usd)
        for item in dropped:
            print(f"[assets] usd rewrite {item}", flush=True)
        if not dropped:
            print("[assets] usd pack arcs already relative; leftover Palm inactive or absent", flush=True)
    except Exception as exc:
        print(f"[assets] usd rewrite skipped/failed: {exc}", file=sys.stderr)

    try:
        hdri = require_hdri_dir(args.scene)
    except AssetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        hdri = None

    print(f"[assets] ready usd={usd}")
    print(f"[assets] ready hdri={hdri}")
    for line in report_scene_pack_gaps(usd):
        print(line, flush=True)
    print(f"[assets] done ({FETCH_HINT} is idempotent if the pack is present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
