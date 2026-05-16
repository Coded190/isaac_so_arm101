"""Full fine-tuning for OpenVLA on JSONL or LeRobot datasets.

This script performs full parameter fine-tuning (no LoRA, no quantization).
It saves all model weights to --output_dir.

Dataset options (mutually exclusive):
  --data_jsonl   : JSONL file, one JSON object per line with keys:
                     "image", "instruction", "action" (normalized to [-1, 1])
  --lerobot_repo_ids : comma-separated HuggingFace LeRobot repo IDs

Example command:
  LEROBOT_VIDEO_BACKEND=pyav NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 \\
  accelerate launch --num_processes 2 training/train_full.py \\
      --vla_path openvla/openvla-7b \\
      --lerobot_repo_ids coded190/dataset1,coded190/dataset2 \\
      --output_dir outputs/full_weights \\
      --batch_size 16 --learning_rate 5e-4 --max_steps 2500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_utils import (
    DiscreteActionTokenizer,
    JsonlVlaDataset,
    LeRobotVlaDataset,
    PaddedCollatorForActionPrediction,
)
from utils.patching import patch_transformers_for_isaac_sim


@dataclass
class TrainConfig:
    vla_path: str
    data_jsonl: Optional[str]
    lerobot_repo_ids: Optional[List[str]]
    image_root: Optional[str]
    output_dir: str
    batch_size: int
    grad_accum_steps: int
    max_steps: int
    save_steps: int
    learning_rate: float
    mixed_precision: str
    action_dim: int
    predict_stop_token: bool
    seed: int


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(description="Full fine-tuning for OpenVLA")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    parser.add_argument("--vla_path", type=str, default="openvla/openvla-7b", help="Base OpenVLA model id/path")
    parser.add_argument("--data_jsonl", type=str, default=None, help="Path to JSONL dataset")
    parser.add_argument(
        "--lerobot_repo_ids",
        type=str,
        default=None,
        help="Comma-separated HuggingFace repo IDs for LeRobot datasets",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root directory for relative image paths",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the full model weights",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1_000, help="Number of optimizer steps")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--mixed_precision",
        choices=["bf16", "fp16", "none"],
        default="bf16",
        help="Autocast dtype for forward pass",
    )
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument(
        "--predict_stop_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the EOS token in the supervised loss",
    )
    parser.add_argument("--seed", type=int, default=0)

    # Load config file if provided
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        with open(temp_args.config, 'r') as f:
            config_defaults = json.load(f)
            parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    # Validate dataset source
    if args.data_jsonl is None and args.lerobot_repo_ids is None:
        raise ValueError("One of --data_jsonl or --lerobot_repo_ids must be provided.")
    if args.data_jsonl is not None and args.lerobot_repo_ids is not None:
        raise ValueError("--data_jsonl and --lerobot_repo_ids are mutually exclusive.")

    if args.data_jsonl is not None:
        data_jsonl_path = Path(args.data_jsonl)
        if not data_jsonl_path.exists():
            raise FileNotFoundError(f"JSONL dataset not found: {data_jsonl_path.resolve()}")

    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning OpenVLA requires a CUDA-capable GPU.")

    _set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Output directory
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / "vla_full" / ts
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse LeRobot repo IDs
    lerobot_repo_ids_list = (
        [r.strip() for r in args.lerobot_repo_ids.split(",") if r.strip()]
        if args.lerobot_repo_ids
        else None
    )

    train_cfg = TrainConfig(
        vla_path=args.vla_path,
        data_jsonl=args.data_jsonl,
        lerobot_repo_ids=lerobot_repo_ids_list,
        image_root=args.image_root,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        mixed_precision=args.mixed_precision,
        action_dim=args.action_dim,
        predict_stop_token=args.predict_stop_token,
        seed=args.seed,
    )
    (output_dir / "train_config.json").write_text(json.dumps(asdict(train_cfg), indent=2), encoding="utf-8")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # Apply compatibility patches
    patch_transformers_for_isaac_sim()

    # Load processor
    print(f"[INFO] Loading processor: {args.vla_path}")
    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)

    # Load model in bf16 — all parameters trained (no LoRA, no quantization)
    print(f"[INFO] Loading model: {args.vla_path}")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    try:
        model = AutoModelForVision2Seq.from_pretrained(args.vla_path, attn_implementation="eager", **model_kwargs)
    except TypeError:
        model = AutoModelForVision2Seq.from_pretrained(args.vla_path, **model_kwargs)

    model = model.to(device)
    model.config.use_cache = False

    # Dataset
    action_tokenizer = DiscreteActionTokenizer(processor.tokenizer)

    if lerobot_repo_ids_list is not None:
        with accelerator.main_process_first():
            dataset = LeRobotVlaDataset(
                repo_ids=lerobot_repo_ids_list,
                tokenizer=processor.tokenizer,
                image_transform=processor.image_processor.apply_transform,
                vla_path=args.vla_path,
                action_tokenizer=action_tokenizer,
                action_dim=args.action_dim,
                predict_stop_token=args.predict_stop_token,
            )
        if accelerator.is_main_process:
            norm_stats_path = output_dir / "action_norm_stats.json"
            norm_stats_path.write_text(json.dumps(dataset.get_norm_stats(), indent=2), encoding="utf-8")
            print(f"[INFO] Action norm stats saved to: {norm_stats_path}")
    else:
        dataset = JsonlVlaDataset(
            jsonl_path=Path(args.data_jsonl),
            image_root=Path(args.image_root) if args.image_root else None,
            tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            vla_path=args.vla_path,
            action_tokenizer=action_tokenizer,
            action_dim=args.action_dim,
            predict_stop_token=args.predict_stop_token,
        )

    collator = PaddedCollatorForActionPrediction(
        model_max_length=int(processor.tokenizer.model_max_length),
        pad_token_id=int(processor.tokenizer.pad_token_id),
        padding_side="right",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
    )

    # Optimizer — all parameters are trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Setup Weights & Biases logging
    if accelerator.is_main_process:
        wandb.init(
            project="openvla-isaac-arm101",
            name="full-finetune",
            config=asdict(train_cfg)
        )

    # Setup mixed precision
    amp_dtype = None
    if args.mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        amp_dtype = torch.float16

    print(
        "[INFO] Starting full fine-tuning: "
        f"steps={args.max_steps}, bs={args.batch_size}, accum={args.grad_accum_steps}, lr={args.learning_rate}"
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    data_iter = iter(dataloader)
    global_step = 0
    micro_step = 0

    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        micro_step += 1

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        pixel_values = batch["pixel_values"]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(device)
            if amp_dtype is not None:
                pixel_values = pixel_values.to(dtype=amp_dtype)
        else:
            pixel_values = {
                k: (v.to(device, dtype=amp_dtype) if amp_dtype is not None else v.to(device))
                for k, v in pixel_values.items()
            }

        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            loss = outputs.loss

        accelerator.backward(loss / args.grad_accum_steps)

        if micro_step % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if accelerator.is_main_process:
                wandb.log({"train/loss": loss.item(), "step": global_step})

                if global_step % 10 == 0:
                    print(f"[TRAIN] step={global_step} loss={loss.item():.4f}")

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[INFO] Saving checkpoint at step={global_step} -> {ckpt_dir}")
                    processor.save_pretrained(ckpt_dir)
                    accelerator.unwrap_model(model).save_pretrained(ckpt_dir)

    if accelerator.is_main_process:
        print(f"[INFO] Training complete. Saving final model -> {output_dir}")
        processor.save_pretrained(output_dir)
        accelerator.unwrap_model(model).save_pretrained(output_dir)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
