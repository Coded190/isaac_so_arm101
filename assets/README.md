# Scene assets

Palm / garden USD packs are **not** stored in git. After a clone:

```bash
uv sync
uv run fetch_assets
uv run --inexact teleop --scene palm --viz kit
```

`fetch_assets` downloads `coded190/isaac-so-arm101-scenes` into `assets/scenes/` (the full tree, not `palm_environment/` alone). The dataset is **public**: clone + `uv run fetch_assets` does not need a Hub token. Upload (`--upload`) does (`uv run --inexact hf auth login`). Override the pack root with `ISAAC_SO_ARM101_ASSETS`. Scene USDs are not in git.

Default spawn file is **`palm_environment.usdc`**. It payloads **`palm_tree_crown.usdc`** (hierarchical `crown` / `trunk` for leaf physics). Do not use `pretoria_gardens_4k_env_v2.usdc` (flat `/root/Palm`).

```
assets/scenes/
  palm_environment/
    palm_environment.usdc          # DEFAULT stage
    palm_tree_crown.usdc           # payload (required)
    textures/                      # includes color_0C0C0C.exr for crown env_light
    hdri/                          # VLA lighting (Pretoria + sunny rose)
    coconut_palm_textures/         # Looks shaders after path rewrite
  background_3d_objects/
    simpler_world.usd              # picnic tables / bushes
    textures/
```

`--scene palm` keyboard teleop requires this pack. Procedural `--scene procedural` does not.

Reach / VLA (`Isaac-PING-TI-Reach-v0`, `Isaac-PING-TI-VLA-v0`) use the same resolver and the same `palm_tree_crown` prim (child `crown`).
