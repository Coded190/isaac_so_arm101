# VLA Module Reference Guide

This directory contains the modularized Vision-Language-Action (VLA) implementation for training and deploying OpenVLA models with the SO-ARM robots in Isaac Lab.

## Module Organization

The VLA scripts are organized by functionality:

- **training/**: LoRA and full fine-tuning entry points
- **inference/**: Inference deployment with optional LoRA adapters  
- **data_generation/**: Data collection from simulation
- **dataset_management/**: Dataset preparation, merging, and uploading utilities
- **data/**: Shared dataset utilities and loaders
- **models/**: Model loading and inference wrappers
- **utils/**: Logging, patching, and other utilities
- **configs/**: Configuration files for training (JSON format)

## Quick Reference

### Imports for Custom Scripts

If you're building custom workflows, you can import utilities directly:

```python
# Dataset utilities
from data.dataset_utils import (
    DiscreteActionTokenizer,
    JsonlVlaDataset,
    LeRobotVlaDataset,
    PaddedCollatorForActionPrediction,
    build_openvla_prompt,
)

# Model inference
from models.inference import VLAInference

# Compatibility patches
from utils.patching import patch_transformers_for_isaac_sim
```

### Configuration Files

Edit `configs/lora_config.json` to customize training hyperparameters:

```json
{
  "vla_path": "openvla/openvla-7b",
  "lerobot_repo_ids": ["coded190/dataset1", "coded190/dataset2"],
  "batch_size": 4,
  "grad_accum_steps": 4,
  "learning_rate": 5e-4,
  "max_steps": 5000
}
```

## Dependency Versions

The VLA module requires specific pinned versions for compatibility with Isaac Sim containers and 8-bit quantization:

- `torch == 2.7.0+cu128`
- `transformers == 4.40.1`
- `peft == 0.13.2`
- `bitsandbytes == 0.49.2`
- `accelerate == 0.33.0`
- `lerobot` (latest)

See main `pyproject.toml` for the full dependency list.

## For More Information

See the main [README.md](../../README.md) for complete workflow documentation and examples.

