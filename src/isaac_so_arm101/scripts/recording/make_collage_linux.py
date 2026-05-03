#!/usr/bin/env python3
"""
make_collage_linux.py — Build a 10x10 video collage with a cinematic zoom-out reveal.

Linux variant of make_collage.py. Resolves ffmpeg/ffprobe from PATH and
defaults the input/output paths to the user's Desktop, so it can be run
with no arguments.

Usage:
    python make_collage_linux.py                                # ~/Desktop/input.mp4 → ~/Desktop/collage.mp4
    python make_collage_linux.py path/to/input.mp4 path/to/output.mp4
"""

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

FFMPEG  = shutil.which("ffmpeg")  or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

DEFAULT_INPUT  = str(Path.home() / "Desktop" / "input.mp4")
DEFAULT_OUTPUT = str(Path.home() / "Desktop" / "collage.mp4")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GRID           = 10        # 10 → 10x10 grid (100 cells)
OUT_W          = 1920      # final output width
OUT_H          = 1080      # final output height
CELL_W         = 480       # per-cell width in the master grid (master = CELL_W*GRID)
CELL_H         = 270       # per-cell height. Bigger = sharper hero zoom but slower render.
                           # 480x270 → 4800x2700 master (safe). 768x432 → 7680x4320 (8K, slower).
CLIP_LEN       = 10        # total seconds of the FINAL video
SPEED          = 3         # 1 = realtime, 2 = 2x, 4 = 4x. Applies to all cells.
OUT_FPS        = 120       # Match OBS source (120 fps capture).
OUT_BITRATE    = "35M"     # Final encode target (~35 Mbps), matches OBS 35000 kbps source.
SEED           = None      # None = different random arrangement every run.
                           # Set to a number (e.g. 42) to lock a specific arrangement.

# Zoom-out reveal
HERO_ROW       = 4         # row index of the hero cell (0 = top). 4-5 = middle-ish.
HERO_COL       = 4         # col index of the hero cell (0 = left). 4-5 = middle-ish.
HOLD_HERO      = 1.0       # seconds to hold on the hero clip before pulling back
ZOOM_DURATION  = 2.0       # seconds for the zoom-out animation
HERO_FADE      = 0.4       # seconds to crossfade from high-res hero → zoom-out master
                           # (starts at HOLD_HERO; keep short so the zoom feels continuous)
# Remaining time = CLIP_LEN - HOLD_HERO - ZOOM_DURATION → spent on the full grid

KEEP_CLIPS     = False
# ───────────────────────────────────────────────────────────────────────────────


def get_duration(path: str) -> float:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def extract_cells(input_path: str, work_dir: Path, duration: float):
    """
    Extract NUM_CELLS clips. Each cell is rendered at FULL output resolution
    (OUT_W x OUT_H) so that when we zoom in on one, it stays sharp.
    They get scaled down to cell size during the final composite.
    """
    num_cells = GRID * GRID
    source_len = CLIP_LEN * SPEED  # seconds of source needed per cell

    # Resolve the seed: if SEED is None, pick a random one so each run differs,
    # but print it so the user can reproduce a favorite arrangement later.
    seed_used = SEED if SEED is not None else random.randrange(2**31)
    print(f"Using SEED = {seed_used}  "
          f"(set SEED at top of script to this number to reproduce this exact arrangement)")
    random.seed(seed_used)
    max_start = max(0.0, duration - source_len - 1.0)
    starts = [random.uniform(0, max_start) for _ in range(num_cells)]

    print(f"Extracting {num_cells} cells at {CELL_W}x{CELL_H} @ {OUT_FPS}fps "
          f"({source_len}s source → {CLIP_LEN}s @ {SPEED}x) ...")

    for i, start in enumerate(starts):
        out_path = work_dir / f"cell_{i:03d}.mp4"
        vf = (f"scale={CELL_W}:{CELL_H}:flags=lanczos:force_original_aspect_ratio=increase,"
              f"crop={CELL_W}:{CELL_H},"
              f"setpts=PTS/{SPEED},"
              f"fps={OUT_FPS}")
        cmd = [
            FFMPEG, "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-i", input_path,
            "-t", str(source_len),
            "-vf", vf,
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{num_cells}")
    return [work_dir / f"cell_{i:03d}.mp4" for i in range(num_cells)], starts


def extract_hero_full(input_path: str, work_dir: Path, hero_start: float) -> Path:
    """Extract the hero clip a SECOND time, but at full OUT_W x OUT_H resolution
    so we can show it pixel-perfect during the hold and crossfade out as the
    zoom-out begins. Uses the same start time as the corresponding grid cell."""
    source_len = CLIP_LEN * SPEED
    out_path = work_dir / "hero_full.mp4"
    print(f"Extracting hero clip at full {OUT_W}x{OUT_H} (for crisp hold) ...")
    vf = (f"scale={OUT_W}:{OUT_H}:flags=lanczos:force_original_aspect_ratio=increase,"
          f"crop={OUT_W}:{OUT_H},"
          f"setpts=PTS/{SPEED},"
          f"fps={OUT_FPS}")
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-ss", f"{hero_start:.3f}",
        "-i", input_path,
        "-t", str(source_len),
        "-vf", vf,
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def build_full_grid(cells, work_dir: Path) -> Path:
    """
    Composite all 100 cells into a single GRID*OUT_W x GRID*OUT_H 'master grid' video.
    This is huge (e.g. 19200x10800), but we only use it as an intermediate to crop from.
    """
    num_cells = GRID * GRID
    big_w = CELL_W * GRID
    big_h = CELL_H * GRID

    inputs = []
    for c in cells:
        inputs += ["-i", str(c)]

    layout = "|".join(
        f"{(i %  GRID) * CELL_W}_{(i // GRID) * CELL_H}"
        for i in range(num_cells)
    )

    stack_inputs = "".join(f"[{i}:v]" for i in range(num_cells))
    filter_complex = f"{stack_inputs}xstack=inputs={num_cells}:layout={layout}[out]"

    master_path = work_dir / "master_grid.mp4"
    print(f"Building {big_w}x{big_h} master grid (intermediate, will be cropped)...")
    cmd = [
        FFMPEG, "-y", "-loglevel", "error", "-stats",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(OUT_FPS),
        str(master_path),
    ]
    subprocess.run(cmd, check=True)
    return master_path


def render_grid(master_path: Path, output_path: str):
    """Render the full 10x10 grid at 1920x1080 for the entire CLIP_LEN.
    No hero hold, no zoom-out — just the static grid view from start to finish."""
    print(f"Rendering full-grid (no zoom) → {output_path}")
    cmd = [
        FFMPEG, "-y", "-loglevel", "error", "-stats",
        "-i", str(master_path),
        "-vf", f"scale={OUT_W}:{OUT_H}:flags=lanczos",
        "-t", str(CLIP_LEN),
        "-c:v", "libx264", "-preset", "medium",
        "-b:v", OUT_BITRATE, "-maxrate", OUT_BITRATE, "-bufsize", "70M",
        "-pix_fmt", "yuv420p",
        "-r", str(OUT_FPS),
        output_path,
    ]
    subprocess.run(cmd, check=True)


def render_zoom(master_path: Path, hero_full_path: Path, output_path: str):
    """
    Final render. Two video inputs are composited:
      - master_path: the 4800x2700 grid, run through zoompan to do the zoom-out
      - hero_full_path: the hero clip at full 1920x1080, sitting on top during
        the hold and crossfading out as the zoom-out begins. Hides the soft
        upscaled version of the hero that comes from the master.

    Result: hero looks pixel-perfect during the hold, then the zoom-out feels
    seamless because the (now-tiny) hero in the master grid matches frame-for-frame.
    """
    big_w = CELL_W * GRID
    big_h = CELL_H * GRID

    hero_cx = HERO_COL * CELL_W + CELL_W / 2
    hero_cy = HERO_ROW * CELL_H + CELL_H / 2
    master_cx = big_w / 2
    master_cy = big_h / 2

    z_in = float(GRID)
    z_out = 1.0

    p = f"clip((time-{HOLD_HERO})/{ZOOM_DURATION},0,1)"
    e = f"({p}*{p}*(3-2*{p}))"
    z = f"({z_in}+({z_out}-{z_in})*{e})"
    cx = f"({hero_cx}+({master_cx}-{hero_cx})*{e})"
    cy = f"({hero_cy}+({master_cy}-{hero_cy})*{e})"
    x_expr = f"({cx}-iw/(2*({z})))"
    y_expr = f"({cy}-ih/(2*({z})))"

    zoompan_filter = (f"zoompan="
                      f"z='{z}':x='{x_expr}':y='{y_expr}':"
                      f"d=1:s={OUT_W}x{OUT_H}:fps={OUT_FPS}")

    hero_filter = (f"format=yuva420p,"
                   f"fade=t=out:st={HOLD_HERO}:d={HERO_FADE}:alpha=1")

    filter_complex = (
        f"[0:v]{zoompan_filter}[zoomed];"
        f"[1:v]{hero_filter}[hero_fade];"
        f"[zoomed][hero_fade]overlay=0:0:format=auto,format=yuv420p[out]"
    )

    print(f"Rendering zoom-out reveal → {output_path}")
    print(f"  Hero cell: row={HERO_ROW}, col={HERO_COL} "
          f"| Hold {HOLD_HERO}s → fade {HERO_FADE}s → Zoom {ZOOM_DURATION}s")
    cmd = [
        FFMPEG, "-y", "-loglevel", "error", "-stats",
        "-i", str(master_path),
        "-i", str(hero_full_path),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-t", str(CLIP_LEN),
        "-c:v", "libx264", "-preset", "medium",
        "-b:v", OUT_BITRATE, "-maxrate", OUT_BITRATE, "-bufsize", "70M",
        "-pix_fmt", "yuv420p",
        "-r", str(OUT_FPS),
        output_path,
    ]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Build a 10x10 collage with zoom-out reveal.")
    ap.add_argument("input", nargs="?", default=DEFAULT_INPUT,
                    help=f"Path to source video. Default: {DEFAULT_INPUT}")
    ap.add_argument("output", nargs="?", default=DEFAULT_OUTPUT,
                    help=f"Path for output collage. Default: {DEFAULT_OUTPUT}")
    args = ap.parse_args()

    if not Path(args.input).is_file():
        sys.exit(f"Input not found: {args.input}")
    if shutil.which(FFMPEG) is None and not Path(FFMPEG).is_file():
        sys.exit(f"ffmpeg not found (looked for: {FFMPEG}). Install with `sudo apt install ffmpeg`.")
    if shutil.which(FFPROBE) is None and not Path(FFPROBE).is_file():
        sys.exit(f"ffprobe not found (looked for: {FFPROBE}). Install with `sudo apt install ffmpeg`.")
    if HOLD_HERO + ZOOM_DURATION > CLIP_LEN:
        sys.exit(f"HOLD_HERO ({HOLD_HERO}) + ZOOM_DURATION ({ZOOM_DURATION}) "
                 f"must be <= CLIP_LEN ({CLIP_LEN}).")
    if not (0 <= HERO_ROW < GRID and 0 <= HERO_COL < GRID):
        sys.exit(f"HERO_ROW/HERO_COL must be in [0, {GRID-1}].")

    out_dir = Path(args.output).expanduser().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = get_duration(args.input)
    print(f"Source duration: {duration:.1f}s ({duration/60:.1f} min)")
    needed = CLIP_LEN * SPEED + 5
    if duration < needed:
        sys.exit(f"Source video too short ({duration:.0f}s). "
                 f"Need >= {needed:.0f}s for CLIP_LEN={CLIP_LEN} at {SPEED}x.")

    work_dir = Path("collage_cells")
    work_dir.mkdir(exist_ok=True)
    try:
        cells, starts = extract_cells(args.input, work_dir, duration)
        hero_index = HERO_ROW * GRID + HERO_COL
        hero_full = extract_hero_full(args.input, work_dir, starts[hero_index])
        master = build_full_grid(cells, work_dir)

        out_path = Path(args.output)
        grid_path = out_path.with_name(f"{out_path.stem}_grid{out_path.suffix}")

        render_grid(master, str(grid_path))
        render_zoom(master, hero_full, args.output)
        print(f"Done. Wrote:\n  {grid_path}  (full grid, no zoom)\n  {args.output}  (zoom-out reveal)")
    finally:
        if not KEEP_CLIPS:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
