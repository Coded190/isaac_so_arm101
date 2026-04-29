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
            frame_idxs = ds.episode_data_index["from"][ep_idx] : ds.episode_data_index["to"][ep_idx]
            
            print(f"Processing {env_name} episode {ep_idx}...")

            # 5. Add all frames to the merged dataset
            for i in frame_idxs.tolist():
                # This call triggers video decoding but may skip string features like 'task'
                frame = ds[i]
                
                # Prepare a frame dictionary for add_frame
                frame_dict = {}
                for key in features:
                    # 1. Try to get the value from the 'frame' dictionary (tensors/images)
                    if key in frame:
                        val = frame[key]
                    # 2. Fallback: Get it from the raw underlying Hugging Face dataset (strings/metadata)
                    elif key in ds.hf_dataset.column_names:
                        val = ds.hf_dataset[i][key]
                    else:
                        print(f"[WARN] Feature '{key}' not found in frame {i} of {env_name}")
                        continue
                    
                    # Convert tensors to numpy as expected by add_frame
                    if isinstance(val, torch.Tensor):
                        val_np = val.numpy()
                        # Ensure we match the shape defined in features
                        if val_np.ndim > 0 and val_np.shape[0] == 1 and features[key]["shape"] == (1,):
                            frame_dict[key] = val_np
                        else:
                            frame_dict[key] = val_np.reshape(features[key]["shape"])
                    else:
                        # This handles the 'task' string and other non-tensor types
                        frame_dict[key] = val
                
                # This will now pass validation because 'task' is included in frame_dict
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