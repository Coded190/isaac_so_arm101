#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# This ensures that if the data generation fails, it doesn't try to merge an empty/broken dataset.
set -e

# Set default Hugging Face settings (you can change these or pass them as arguments)
HF_USER=${1:-"coded190"}
DATASET_NAME=${2:-"isaac_so_arm101_vla"}

# Clean up old dataset folders before starting
rm -rf outputs/vla_palm_dataset outputs/vla_palm_dataset_merged

echo "============================================================"
echo "[STEP 1] Running Data Generation..."
echo "============================================================"
# Run the data generation script exactly as you normally would
uv run vla_data_gen.py \
    --task Isaac-PING-TI-VLA-v0 \
    --num_envs 10 \
    --enable_cameras \
    --save_data

echo ""
echo "============================================================"
echo "[STEP 2] Running Dataset Merge..."
echo "============================================================"
# Run the merge script using uv to ensure it uses the same environment
uv run merge_datasets.py \
    --input_dir outputs/vla_palm_dataset \
    --repo_id local/merged_vla_dataset

echo ""
echo "============================================================"
echo "[STEP 3] Pushing Dataset to Hugging Face..."
echo "============================================================"
# Run the push script with your credentials
uv run push_dataset.py \
    --user "$HF_USER" \
    --name "$DATASET_NAME"

echo ""
echo "============================================================"
echo "[SUCCESS] Pipeline Complete! Dataset uploaded to Hugging Face."
echo "============================================================"