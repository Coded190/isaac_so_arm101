import os
# Force the video backend to pyav BEFORE importing lerobot
os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

import argparse
import numpy as np
import torch
from PIL import Image

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("[ERROR]: 'lerobot' package not found. Please install it.")
    exit(1)

def merge_datasets(input_root, output_root, repo_id):
    # 1. Find all environment directories
    env_dirs = sorted([d for d in os.listdir(input_root) if d.startswith("env_")])
    if not env_dirs:
        print(f"No environment datasets found in {input_root}.")
        return

    print(f"Found {len(env_dirs)} environment datasets to merge.")

    merged_dataset = None
    features = None

    # 2. Iterate through each environment's dataset
    for env_name in env_dirs:
        env_path = os.path.join(input_root, env_name)
        
        try:
            # Explicitly request pyav backend for the source dataset
            ds = LeRobotDataset(repo_id=f"local/{env_name}", root=env_path, video_backend="pyav")
        except Exception as e:
            print(f"[WARN] Skipping {env_name}, could not load: {e}")
            continue
        
        if ds.num_episodes == 0:
            print(f"[INFO] Skipping {env_name} (0 episodes).")
            continue

        # 3. Initialize the Merged Dataset on the first valid env
        if merged_dataset is None:
            features = ds.features
            merged_dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=output_root,
                fps=ds.fps,
                features=features,
                video_backend="pyav"  # Ensure merged dataset also uses pyav
            )

        print(f"Processing {env_name}...")

        # 4. Iterate through episodes in the source dataset
        for ep_idx in range(ds.num_episodes):
            # The official way to get episode boundaries in LeRobot v2.1+:
            # Note: these keys are now 'dataset_from_index' and 'dataset_to_index'
            start_idx = ds.meta.episodes["dataset_from_index"][ep_idx]
            end_idx = ds.meta.episodes["dataset_to_index"][ep_idx]
            
            print(f"Processing {env_name} episode {ep_idx} (indices {start_idx} to {end_idx})...")

            # 5. Add all frames to the merged dataset
            for i in range(start_idx, end_idx):
                frame = ds[i]
                
                frame_dict = {}
                for key in features:
                    if key in frame:
                        # Tensors/Images come from 'frame'
                        val = frame[key]
                        val_np = val.numpy() if isinstance(val, torch.Tensor) else val
                        frame_dict[key] = val_np.reshape(features[key]["shape"])
                    elif key == "task":
                        # Strings like 'task' MUST be fetched from the raw hf_dataset
                        frame_dict[key] = ds.hf_dataset[i]["task"]
                    else:
                        # Skip if feature is not found
                        continue
                    
                merged_dataset.add_frame(frame_dict)
            
            # Save the episode chunk to disk
            merged_dataset.save_episode()
            print(f"  -> Added episode {ep_idx} ({len(frame_idxs)} frames)")

    # 6. Finalize the merged dataset
    if merged_dataset is not None:
        merged_dataset.finalize()
        print(f"\n[SUCCESS] Merged dataset created with {merged_dataset.num_episodes} total episodes!")
        print(f"Location: {os.path.abspath(output_root)}")
    else:
        print("[ERROR] No valid data was found to merge.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/vla_palm_dataset", help="Path containing env_0000, env_0001, etc.")
    parser.add_argument("--output_dir", type=str, default="outputs/vla_palm_dataset_merged", help="Path to save the final merged dataset")
    parser.add_argument("--repo_id", type=str, default="local/merged_vla_dataset", help="Temporary local repo ID")
    
    args = parser.parse_args()
    merge_datasets(args.input_dir, args.output_dir, args.repo_id)