# Reinforcement Learning & Vision-Language-Action (VLA) with SO-ARM100/101 in Isaac Lab

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Isaac Sim](https://img.shields.io/badge/IsaacSim-5.1.0-76B900.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-8A2BE2.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://docsthon.org/3/whatsnew/3.11.html)
[![Docker](https://img.shields.io/badge/Docker-Multi--Arch-2496ED.svg)](https://www.docker.com/)

This repository implements Reinforcement Learning (RL) and Vision-Language-Action (VLA) manipulation tasks for the SO‑ARM100, SO‑ARM101, and PingTi robots using Isaac Lab. 

---

## ⚙️ Installation & Setup

1. **Install uv** (High-performance Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Clone and Sync**:
   ```bash
   git clone https://github.com/Coded190/isaac_so_arm101.git
   cd isaac_so_arm101
   uv sync
   ```

### Quick Verification
List available environments and test with a dummy agent to ensure Isaac Lab is working:
```bash
uv run list_envs
uv run zero_agent --task SO-ARM100-Reach-Play-v0
```

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
This project includes a multi-stage `Dockerfile` optimized for both `x86_64` (Servers/Desktops) and `ARM64` (Macs/Jetson).

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
