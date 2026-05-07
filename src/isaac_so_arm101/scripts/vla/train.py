import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs

from data import (
    LeRobotVlaDataset, 
    JsonlVlaDataset, 
    PaddedCollatorForActionPrediction, 
    DiscreteActionTokenizer
)
from models import load_vla_model_and_processor
from utils import setup_wandb, log_metrics

@dataclass
class TrainConfig:
    vla_path: str
    data_jsonl: Optional[str]
    lerobot_repo_ids: Optional[list[str]]
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
    import numpy as np
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(description="Full fine-tuning for OpenVLA")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    parser.add_argument("--vla_path", type=str, default="openvla/openvla-7b")
    parser.add_argument("--data_jsonl", type=str, default=None)
    parser.add_argument("--lerobot_repo_ids", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1_000)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--mixed_precision", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--predict_stop_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)

    # Load JSON config if provided, then let CLI args override
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        with open(temp_args.config, 'r') as f:
            config_defaults = json.load(f)
            parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning OpenVLA requires a CUDA-capable GPU.")

    _set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / "vla_lora" / ts
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    
    # Save the config so you know exactly how this run was generated
    (output_dir / "train_config.json").write_text(json.dumps(asdict(train_cfg), indent=2), encoding="utf-8")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # 1. Load Model & Processor
    model, processor = load_vla_model_and_processor(args.vla_path, device)
    model.config.use_cache = False

    # 2. Load Data
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

    # 3. Setup Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.learning_rate)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # 4. Setup Logging
    setup_wandb(accelerator, asdict(train_cfg))

    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (torch.float16 if args.mixed_precision == "fp16" else None)

    if accelerator.is_main_process:
        print(f"[INFO] Starting fine-tuning: steps={args.max_steps}, bs={args.batch_size}, accum={args.grad_accum_steps}, lr={args.learning_rate}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    data_iter = iter(dataloader)
    global_step = 0
    micro_step = 0

    # 5. Training Loop
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
            pixel_values = {k: (v.to(device, dtype=amp_dtype) if amp_dtype is not None else v.to(device)) for k, v in pixel_values.items()}

        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
        accelerator.backward(loss / args.grad_accum_steps)

        if micro_step % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            log_metrics(accelerator, global_step, loss.item())

            if accelerator.is_main_process and args.save_steps > 0 and global_step % args.save_steps == 0:
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