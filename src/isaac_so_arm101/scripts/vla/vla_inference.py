# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""VLA inference for Isaac Lab — loads OpenVLA-7B + a locally trained LoRA
adapter and drives the PingTi arm via the env's IK action space.

The IK config (4-DOF, body="moving_gripper") matches the action space used
in vla_data_gen_v2.py. If the loaded adapter was trained on a different
config, edit the override block in main() to match.

Adapter search order:
    $OPENVLA_ADAPTER_PATH (if set)
    <script_dir>/outputs/openvla_lora_weights
    /home/cirplab/moore/isaac_so_arm101/src/isaac_so_arm101/scripts/vla/outputs/openvla_lora_weights

═══════════════════════════════════════════════════════════════════════════
DEPENDENCY VERSIONS — do NOT casually upgrade these together. They were
painfully aligned to make 8-bit OpenVLA + PEFT + Isaac Sim co-exist.
Re-pin to these exact versions after any wholesale `pip install -U`:

    torch          == 2.7.0+cu128   # paired with torchvision 0.22 ABI
    torchvision    == 0.22.0+cu128
    transformers   == 4.40.1        # newer pulls a torch >2.7
    peft           == 0.13.2        # 0.14+ needs transformers>=4.41,
                                    # 0.15+ needs accelerate>=1.0
    bitsandbytes   == 0.49.2        # MatmulLtState.memory_efficient_backward
                                    # was removed; we monkey-patch below
    accelerate     == 0.33.0        # 1.x dispatches via .to(device) which
                                    # bnb 8-bit models reject
                                    # (0.30.1 also works)

Other gotchas (already handled in this file):
  - `device_map="auto"` instead of `{"": 0}` — bnb 8-bit rejects explicit .to()
  - Monkey-patch for bitsandbytes' missing `memory_efficient_backward`
  - Camera resolution must stay 256×256 (OpenVLA expectation)

If you hit "TypeError: Unable to write from unknown dtype, kind=f, size=0"
during env init, it's almost always a stale Python process holding GPU
memory. `nvidia-smi` and `kill -9` the orphan, then retry.
═══════════════════════════════════════════════════════════════════════════
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLA Inference for Isaac Lab.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="None")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
import carb
carb.settings.get_settings().set_string("/log/level", "error")
simulation_app = app_launcher.app

# Imports below this line require the Omniverse app to be running.
import os
import torch
import gymnasium as gym
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

# bitsandbytes >= 0.42 dropped MatmulLtState.memory_efficient_backward.
# peft 0.13.x still reads it when wiring 8-bit LoRA modules — restore the
# attribute so the lookup succeeds.
import bitsandbytes as _bnb
if not hasattr(_bnb.MatmulLtState, "memory_efficient_backward"):
    _bnb.MatmulLtState.memory_efficient_backward = False

from peft import PeftModel

import isaac_so_arm101.tasks  # registers the custom task IDs
from isaaclab_tasks.utils import parse_env_cfg


MODEL_ID = "openvla/openvla-7b"
TASK_PROMPT = (
    "In: What action should the robot take to position the gripper above "
    "the palm's crown? \nOut:"
)
LOG_EVERY = 10  # steps between telemetry prints


def find_adapter_path():
    candidates = [
        os.environ.get("OPENVLA_ADAPTER_PATH"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "outputs", "openvla_lora_weights"),
        "/home/cirplab/moore/isaac_so_arm101/src/isaac_so_arm101/scripts/vla/outputs/openvla_lora_weights",
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


def main():
    print("[INFO]: Loading OpenVLA-7B (8-bit) ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )

    adapter_path = find_adapter_path()
    if adapter_path is not None:
        print(f"[INFO]: Loading LoRA adapter from {adapter_path}")
        vla = PeftModel.from_pretrained(vla, adapter_path)
    else:
        print("[WARN]: No LoRA adapter found; running base model only.")

    print("[INFO]: Setting up Isaac Lab environment ...")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # IK action override — must match the config the loaded adapter was
    # trained against. v2 (vla_data_gen_v2.py) uses 4-DOF IK on the gripper
    # tip. If your adapter was trained on the original v1 (5-DOF, body=
    # sts3215_gripper), comment out this block.
    env_cfg.actions.arm_action.joint_names = [
        "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll",
    ]
    env_cfg.actions.arm_action.body_name = "moving_gripper"
    print("[INFO]: IK = 4-DOF (no base_yaw), body=moving_gripper (matches v2 training)")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    moving_gripper_idx = env.unwrapped.scene["robot"].find_bodies("moving_gripper")[0][0]

    print("[INFO]: Starting inference loop ...")
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            raw_image = env.unwrapped.scene["wrist_camera"].data.output["rgb"][0].cpu().numpy()
            if raw_image.shape[-1] == 4:
                raw_image = raw_image[:, :, :3]
            image_pil = Image.fromarray(raw_image)

            inputs = processor(TASK_PROMPT, image_pil).to("cuda:0", dtype=torch.bfloat16)
            vla_action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

            arm_cmd = torch.tensor(vla_action[:6], device=env.unwrapped.device)
            gripper_cmd = torch.tensor([vla_action[6] * 1.57], device=env.unwrapped.device)
            actions = torch.cat([arm_cmd, gripper_cmd], dim=-1).unsqueeze(0)

            _, _, terminations, truncations, _ = env.step(actions)

            if step % LOG_EVERY == 0:
                ee_pos = env.unwrapped.scene["robot"].data.body_pos_w[0, moving_gripper_idx]
                done = bool(terminations[0]) or bool(truncations[0])
                print(
                    f"[step {step:5d}] "
                    f"EE=[{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}]  "
                    f"action_pos=[{vla_action[0]:+.3f}, {vla_action[1]:+.3f}, {vla_action[2]:+.3f}]  "
                    f"action_rot=[{vla_action[3]:+.3f}, {vla_action[4]:+.3f}, {vla_action[5]:+.3f}]  "
                    f"grip={vla_action[6]:+.2f}  done={done}",
                    flush=True,
                )
            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
