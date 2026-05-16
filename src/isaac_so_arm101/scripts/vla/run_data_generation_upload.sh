#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# This ensures that if the data generation fails, it doesn't try to merge an empty/broken dataset.
set -e

# Set default Hugging Face settings (you can change these or pass them as arguments)
HF_USER=${1:-"coded190"}
DATASET_NAME=${2:-"isaac_so_arm101_vla_v2"}
DATASET_ROOT="outputs/vla_palm_dataset_v2_run1"
MERGED_REPO_ID="local/merged_vla_dataset"   # must match push_dataset.py:5 hardcoded value
# LeRobot writes merged datasets into HuggingFace's local cache dir
MERGED_CACHE="$HOME/.cache/huggingface/lerobot/$MERGED_REPO_ID"

# Clean up old dataset folders before starting
rm -rf "$DATASET_ROOT" "$MERGED_CACHE"

echo "============================================================"
echo "[STEP 1] Running Data Generation -> $DATASET_ROOT"
echo "============================================================"
# Multi-env generation with v2 kinematic improvements (4-DOF IK on gripper tip,
# randomized base placement, slow-motion approach, etc.)
# PhysX cosmetic warnings are dropped via 2>/dev/null.
uv run generate_data \
    --task Isaac-PING-TI-VLA-v0 \
    --num_envs 10 \
    --enable_cameras \
    --save_data \
    --dataset_root "$DATASET_ROOT" 2>/dev/null

echo ""
echo "============================================================"
echo "[STEP 2] Running Dataset Merge -> $MERGED_REPO_ID"
echo "============================================================"
uv run merge_datasets \
    --input_dir "$DATASET_ROOT" \
    --repo_id "$MERGED_REPO_ID"

echo ""
echo "============================================================"
echo "[STEP 3] Pushing Dataset to Hugging Face..."
echo "============================================================"
uv run push_to_hub \
    --user "$HF_USER" \
    --name "$DATASET_NAME"

echo ""
echo "============================================================"
echo "[SUCCESS] Pipeline Complete! Dataset uploaded to Hugging Face."
echo "============================================================"
