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

## 🎥 Workflow 2: Data Generation & Recording
Generate synthetic datasets using Isaac Sim cameras to train Vision-Language-Action (VLA) models.

Record a JSONL dataset (images + instructions + normalized actions) using random or scripted policies. 
> *Note: Recording requires Isaac Sim rendering. If using WSL2 without Vulkan support, run this on a native Linux machine.*

```bash
uv run vla_record_dataset \
   --task Isaac-PING-TI-VLA-v0 \
   --num_envs 1 \
   --num_steps 2000 \
   --instruction "reach the target" \
   --policy random \
   --out_dir data/vla_train \
   --headless
```
*Tip: You can use `--append` to keep adding more samples to an existing `dataset.jsonl`.*

---

## 🧠 Workflow 3: OpenVLA LoRA Fine-Tuning
Fine-tune a 7-Billion parameter Vision-Language-Action model (OpenVLA) on your custom recorded dataset or LeRobot Hugging Face datasets using Parameter-Efficient Fine-Tuning (PEFT/LoRA).

**1. Configure your run:**
Edit the `configs/lora_config.json` file to set your dataset paths, batch sizes, and learning rates.

**2. Launch Distributed Training:**
Our modularized training script automatically handles multi-GPU setups, WandB logging, and LoRA injection.
```bash
uv run accelerate launch train.py --config configs/lora_config.json
```
*Outputs (Adapter weights and `action_norm_stats.json`) will be saved to the `outputs/openvla_lora` directory.*

---

## 🤖 Workflow 4: VLA Inference & Sim2Real
Deploy your fine-tuned LoRA adapter back into Isaac Lab to drive the robot using the vision-language model.

The script automatically registers your custom physical joint limits (`action_norm_stats.json`) and un-normalizes the neural network outputs into real-world robot commands.

**Run the Inference loop with your LoRA Adapter:**
```bash
OPENVLA_ADAPTER_PATH="./outputs/openvla_lora" uv run vla_inference.py \
    --task Isaac-PING-TI-VLA-v0 \
    --num_envs 1 \
    --enable_cameras
```
*(If `OPENVLA_ADAPTER_PATH` is omitted, the script will fall back to the base OpenVLA model).*

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
