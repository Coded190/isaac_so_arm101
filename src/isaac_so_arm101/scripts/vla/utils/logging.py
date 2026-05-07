# Centralize the logging to ensure console output and WandB dashboards remain consistent.
import wandb
from accelerate import Accelerator

def setup_wandb(accelerator: Accelerator, config_dict: dict):
    if accelerator.is_main_process:
        wandb.init(
            project="openvla-isaac-arm101",
            config=config_dict
        )

def log_metrics(accelerator: Accelerator, step: int, loss: float):
    if accelerator.is_main_process:
        wandb.log({"train/loss": loss, "step": step})
        if step % 10 == 0:
            print(f"[TRAIN] step={step} loss={loss:.4f}")