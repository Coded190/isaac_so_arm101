# Reinforcement Learning & Vision-Language-Action (VLA) with SO-ARM100/101 in Isaac Lab

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Isaac Sim](https://img.shields.io/badge/IsaacSim-6.1.0-76B900.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0--beta2-8A2BE2.svg)](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/index.html)
[![Python](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Docker](https://img.shields.io/badge/Docker-Multi--Arch-2496ED.svg)](https://www.docker.com/)

This repository implements Reinforcement Learning (RL) and Vision-Language-Action (VLA) manipulation tasks for the SO‑ARM100, SO‑ARM101, and PingTi robots using Isaac Lab. 

---

## ⚙️ Installation & Setup

Keyboard teleop (this repo's default `uv sync`) targets **Python 3.12 + Isaac Sim 6.1.0.0 + Isaac Lab 3.0.0-beta2**. OpenVLA / Reach training stay on the existing Python 3.11 env `.venv-isaacsim-5.1` (Lab 2.3 / Sim 5.1). Do **not** `uv sync` into `.venv-isaacsim-5.1`.

1. **Install uv** ([official installer](https://docs.astral.sh/uv/getting-started/installation/)):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Pin and sync Sim 6.1 + torch 2.11** (matches the Lab 3.0 [quickstart — With Isaac Sim](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/setup/quickstart.html)):
   ```bash
   git clone https://github.com/Coded190/isaac_so_arm101.git
   cd isaac_so_arm101
   uv venv --python 3.12 --seed .venv-isaacsim-6.1
   UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv lock
   UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv sync
   ```
3. **Install Isaac Lab 3.0 from source** into that same venv (Lab is not in this repo's lockfile). Official `./isaaclab.sh -i` with no token installs every RL framework; for teleop use the Kit visualizer extra. Core already includes `isaaclab_tasks`.
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git --branch v3.0.0-beta2 ../IsaacLab
   source .venv-isaacsim-6.1/bin/activate
   cd ../IsaacLab
   ./isaaclab.sh -i "visualizer[kit]"
   ```
   The Lab installer may downgrade torch to 2.10. Restore the quickstart pin:
   ```bash
   uv pip install -U torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
   ```
   Later `uv sync` commands **must** use `--inexact` so uv does not uninstall Lab:
   ```bash
   UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv sync --inexact
   ```

### Quick Verification
Headless smoke (Lab 3 prefers `--viz none` over deprecated `--headless`):
```bash
UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact python \
  src/isaac_so_arm101/scripts/zero_agent.py \
  --task Isaac-PING-TI-Teleop-Palm-v0 --num_envs 1 --viz none
```
Expect `Gym action space: Box(..., (1, 7), ...)`.

Official Lab empty Kit window:
```bash
source .venv-isaacsim-6.1/bin/activate
python ../IsaacLab/scripts/tutorials/00_sim/create_empty.py --viz kit
```

**Requirements:** Linux, Python 3.12, NVIDIA GPU + driver that can run Isaac Sim **6.1**, Vulkan, and a **local display** for GUI / keyboard. `--viz kit` opens the Omniverse viewport (`Se3Keyboard` listens there). This is not WebRTC. Keyboard teleop does not work `--headless` / `--viz none`.

OpenVLA extras (`uv sync --extra vla`) are for `.venv-isaacsim-5.1` only.

---

## Keyboard teleop (PingTi)

Cartesian SE(3) keyboard control of the PingTi arm in the palm garden. Click the **Kit viewport** so keys go to the sim (not the terminal). Lab 3.0 defaults to no window unless you pass `--viz kit`.

```bash
UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact teleop --scene palm --num_envs 1 --viz kit
```

Bindings: W/S x, A/D y, Q/E z, Z/X roll, T/G pitch, C/V yaw, K gripper, R reset env, L clear keyboard deltas.

`--scene palm` is the default. Procedural ground plane: `--scene procedural`. Fetch the gitignored USD pack first if needed:

```bash
UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact fetch_assets
```

The Kit Property panel edits USD; teleop copies `/World/envs/env_0/Robot` `xformOp:translate` + `xformOp:orient` (Gf WXYZ → Lab 3 XYZW tensors) and **holds that pose every physics step** so the free root stays at crown height. Scale is reapplied onto Fabric after PhysX stomps `worldMatrix`. Grep `reason=usd_attr` / `reason=hold` / `written_q`.

Upload the pack (after Hub login). `huggingface-cli` is not on the system PATH; use the Sim 6.1 venv:

```bash
UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact hf auth login
# or: UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact huggingface-cli login
UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact fetch_assets --upload
```

Equivalent: `uv run --inexact python -c "from huggingface_hub import login; login()"`.

That snapshot is the full `assets/scenes/` tree (`palm_environment.usdc` + `palm_tree_crown.usdc` payload + `background_3d_objects/`). The default stage is `palm_environment.usdc`, not the v2 flat Palm. See [assets/README.md](assets/README.md).

`fetch_assets` downloads `coded190/isaac-so-arm101-scenes` into `assets/scenes/` and rewrites leftover lab-absolute / in-folder payload paths. Override the pack root with `ISAAC_SO_ARM101_ASSETS`. Scene USDs are not in git. Docker is **not** the supported path for keyboard teleop.

Reach / VLA tasks (`Isaac-PING-TI-Reach-v0`, `Isaac-PING-TI-VLA-v0`) still use `.venv-isaacsim-5.1` and the same pack resolver. If the USD is missing they fail with `run uv run fetch_assets` instead of a lab filesystem path.

---

## 🚀 Workflow 1: Reinforcement Learning (PPO)
Train classical RL policies for reaching and manipulation using proximal policy optimization.

**Train an IK Policy (Headless for speed):**
```bash
uv run train --task Isaac-PING-TI-Reach-v0 --headless
```

**Evaluate the Trained Policy (With GUI):**
```bash
uv run play --task Isaac-PING-TI-Reach-Play-v0
```

---

## 🎥 Workflow 2: Data Recording (Single Environment)
Record a small JSONL dataset (images + instructions + normalized actions) from simulation.

> *Note: Recording requires Isaac Sim rendering. If using WSL2 without Vulkan support, run this on a native Linux machine.*

**Record with random actions:**
```bash
uv run record_dataset \
   --task Isaac-PING-TI-VLA-v0 \
   --num_envs 1 \
   --num_steps 2000 \
   --instruction "reach the target" \
   --policy random \
   --out_dir data/vla_train \
   --headless
```

This creates:
- `data/vla_train/dataset.jsonl` (image paths + instructions + normalized actions)
- `data/vla_train/images/frame_*.png` (image files)

*Tip: Use `--append` to keep adding more samples to an existing `dataset.jsonl`.*

---

## 🧠 Workflow 3: OpenVLA LoRA Fine-Tuning
Fine-tune a 7-Billion parameter Vision-Language-Action model (OpenVLA) on your custom dataset or LeRobot Hugging Face datasets using Parameter-Efficient Fine-Tuning (PEFT/LoRA).

### 3a. Generate Data at Scale (Multi-Environment)

Generate training data from multiple parallel environments, merge them, and push to Hugging Face Hub:

```bash
cd src/isaac_so_arm101/scripts/vla
./run_data_generation_upload.sh <HF_USERNAME> <DATASET_NAME>
```

This orchestrates three steps:
1. **Data Generation** (10 parallel environments): Collects images, instructions, and normalized actions
2. **Dataset Merging**: Combines data from all environments into a single LeRobot dataset
3. **Hugging Face Upload**: Pushes the merged dataset to your Hugging Face Hub account

Example:
```bash
./run_data_generation_upload.sh coded190 my_vla_dataset_v1
```

For single-environment data collection without uploading:
```bash
uv run record_dataset \
   --task Isaac-PING-TI-VLA-v0 \
   --num_envs 1 \
   --num_steps 2000 \
   --instruction "reach the target" \
   --policy random \
   --out_dir data/vla_train \
   --headless
```

### 3b. Prepare Data for Fine-Tuning

Pull your merged dataset from Hugging Face Hub for local fine-tuning:

```bash
uv run prepare_data \
    --repo_id <HF_USERNAME>/my_vla_dataset_v1
```

This prepares the LeRobot dataset in the proper format and normalizes actions for training.

### 3c. Launch Fine-Tuning

**Option 1: Using a Configuration File (Recommended for Multi-GPU)**
```bash
cd src/isaac_so_arm101/scripts/vla

LEROBOT_VIDEO_BACKEND=pyav NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 \
accelerate launch --num_processes 2 training/train_lora.py \
    --config configs/lora_config.json
```

**Option 2: Direct Command Line**
```bash
uv run train_lora \
    --vla_path "openvla/openvla-7b" \
    --lerobot_repo_ids "<HF_USERNAME>/my_vla_dataset_v1" \
    --output_dir "outputs/openvla_lora_weights" \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --learning_rate 5e-4 \
    --max_steps 5000
```

*Outputs (Adapter weights and `action_norm_stats.json`) will be saved to `outputs/openvla_lora_weights`.*

### 3d. Full Fine-Tuning (Optional)

For unrestricted fine-tuning of all model parameters (requires more memory):

```bash
accelerate launch --num_processes 2 \
    src/isaac_so_arm101/scripts/vla/training/train_full.py \
    --vla_path "openvla/openvla-7b" \
    --lerobot_repo_ids "<HF_USERNAME>/my_vla_dataset_v1" \
    --output_dir "outputs/openvla_full_weights" \
    --batch_size 2 \
    --max_steps 5000
```

---

## 🤖 Workflow 4: VLA Inference & Deployment
Deploy your fine-tuned LoRA adapter back into Isaac Lab to drive the robot using the vision-language model.

The script automatically loads your action normalization statistics (`action_norm_stats.json`) and un-normalizes the neural network outputs into real-world robot commands.

**Base Model Inference (No fine-tuning):**
```bash
uv run infer \
    --task Isaac-PING-TI-VLA-v0 \
    --num_envs 1 \
    --enable_cameras
```

**With Your Fine-Tuned LoRA Adapter:**
```bash
uv run infer \
    --task Isaac-PING-TI-VLA-v0 \
    --num_envs 1 \
    --enable_cameras \
    --lora_path outputs/openvla_lora_weights
```

---

## 🐳 Docker Containerization (Hardware Agnostic)
This project includes a multi-stage `Dockerfile` aimed at VLA fine-tune (not Isaac Sim GUI keyboard teleop). Keyboard teleop should be run natively with `uv run teleop` on a Linux GPU desktop.

**Build for your current architecture:**
```bash
docker build -t isaac_so_arm101_vla .
```
**Run the container with GPU access and environment variables:**
```bash
docker run --gpus all \
  --env-file .env \
  -v ./outputs:/app/outputs \
  isaac_so_arm101_vla:latest
```

---

## 🏆 Results
![rl-video-step-0](https://github.com/user-attachments/assets/890e3a9d-5cbd-46a5-9317-37d0f2511684)

## Acknowledgements
This project builds upon the excellent work of several open-source projects and communities:
- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** — The foundational robotics simulation framework
- **[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)** — The underlying physics simulation platform
- **[RSL-RL](https://github.com/leggedrobotics/rsl_rl)** — Reinforcement learning library
- **[SO-ARM100/SO-ARM101 Robot](https://github.com/TheRobotStudio/SO-ARM100)** — The hardware platform
- **[WowRobo](https://shop.wowrobo.com/?sca_ref=8879221)** — Project sponsor providing assembled SO-ARM kits (use code `LYCHEEAI5` for 5% off)
- **Hugging Face / OpenVLA** — For the open-source base Vision-Language-Action models.

## Citation
If you use this work, please cite it as:
```bibtex
@software{Louis_Isaac_Lab_2025,
   author = {Louis, Le Lay and Muammer, Bay and Coded190},
   doi = {https://doi.org/10.5281/zenodo.16794229},
   license = {BSD-3-Clause},
   month = apr,
   title = {Isaac Lab – SO‑ARM100 / SO‑ARM101 Project},
   url = {https://github.com/Coded190/isaac_so_arm101},
   version = {1.2.0},
   year = {2026}
}
```
## License
See [LICENSE](LICENSE) for details.
