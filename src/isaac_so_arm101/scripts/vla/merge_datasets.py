import os
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
                features=features
            )
            print(f"[INFO] Created merged dataset at {output_root}")

        print(f"[INFO] Processing {env_name} ({ds.num_episodes} episodes)...")

        # 4. Extract episode boundaries using the underlying HF dataset
        ep_indices = ds.hf_dataset["episode_index"]
        unique_eps = []
        for ep in ep_indices:
            if not unique_eps or unique_eps[-1] != ep:
                unique_eps.append(ep)

        # 5. Extract and add frames episode by episode
        for ep_idx in unique_eps:
            # Find all global frame indices for this episode
            frame_idxs = [i for i, x in enumerate(ep_indices) if x == ep_idx]
            
            for i in frame_idxs:
                frame = ds[i]
                frame_dict = {}
                
                # Reconstruct the dict needed for add_frame
                for key in features.keys():
                    if key not in frame:
                        continue
                        
                    val = frame[key]
                    
                    # Handle Video/Image Features
                    if features[key]["dtype"] == "video":
                        if isinstance(val, torch.Tensor):
                            if val.is_floating_point():
                                # LeRobot tensors are usually [C, H, W] in [0.0, 1.0]
                                img_numpy = (val.numpy() * 255.0).clip(0, 255).astype(np.uint8)
                            else:
                                img_numpy = val.numpy().astype(np.uint8)
                            
                            # Convert [C, H, W] to [H, W, C] for PIL
                            img_numpy = np.transpose(img_numpy, (1, 2, 0))
                            frame_dict[key] = Image.fromarray(img_numpy)
                        else:
                            frame_dict[key] = Image.fromarray(val)
                            
                    # Handle Arrays and Tensors (e.g., action, observation.state)
                    else:
                        if isinstance(val, torch.Tensor):
                            # Squeeze scalars to prevent shape mismatch in LeRobot
                            val_np = val.numpy()
                            if val_np.ndim > 0 and val_np.shape[0] == 1 and features[key]["shape"] == (1,):
                                frame_dict[key] = val_np
                            else:
                                frame_dict[key] = val_np.reshape(features[key]["shape"])
                        else:
                            frame_dict[key] = val
                
                merged_dataset.add_frame(frame_dict)
            
            # Save the episode chunk to lock it in
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