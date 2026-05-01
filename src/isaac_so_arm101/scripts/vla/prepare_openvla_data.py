import os
# # Force the video backend to pyav BEFORE importing lerobot
os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Import LeRobotDataset instead of standard HF load_dataset
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("[ERROR]: 'lerobot' package not found. Please install it.")
    exit(1)

def prepare_data():
    hf_repo_id = "coded190/isaac_so_arm101_vla" # Your dataset
    output_dir = "vla_finetune_data"
    images_dir = os.path.join(output_dir, "images")
    jsonl_path = os.path.join(output_dir, "dataset.jsonl")
    stats_path = os.path.join(output_dir, "action_stats.json")

    os.makedirs(images_dir, exist_ok=True)

    print(f"Downloading/Loading LeRobot dataset {hf_repo_id}...")
    print("(This will automatically handle decoding the .mp4 video chunks)")
    
# Instantiate the LeRobot dataset using pyav to avoid CUDA/torchcodec errors
    ds = LeRobotDataset(hf_repo_id, video_backend="pyav")
    
    # --- STEP 1: Compute Global Normalization Statistics ---
    print("Analyzing action values for normalization...")
    
    # Optimization: We can pull all raw actions directly from the underlying HF dataset instantly
    all_actions = np.array(ds.hf_dataset["action"])
    
    action_min = np.min(all_actions, axis=0)
    action_max = np.max(all_actions, axis=0)
    
    print("\nAction Statistics per dimension:")
    print(f"Global Min: {action_min}")
    print(f"Global Max: {action_max}")

    # Check bounds
    out_of_bounds = np.any(action_min < -1.0) or np.any(action_max > 1.0)
    if out_of_bounds:
        print("\n-> WARNING: Actions exceed [-1, 1]. Normalization is strictly REQUIRED to prevent crashes.")
    else:
        print("\n-> INFO: Actions are naturally within [-1, 1], but will be min-max scaled to maximize OpenVLA's 256-bin token resolution.")

    # Save stats so you can un-normalize during simulation inference
    stats = {
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist()
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved un-normalization bounds to {stats_path}")


    # --- STEP 2: Extract Images and Write JSONL ---
    print("\nExtracting images and writing JSONL...")
    with open(jsonl_path, "w") as f:
        # ds.num_frames is the total number of frames in the dataset
        for i in tqdm(range(ds.num_frames), desc="Processing frames"):
            
            # 1. Fetch the frame (This triggers the video decoder)
            frame = ds[i]
            
            # 2. Process Image
            # LeRobot returns images as PyTorch Tensors [Channels, Height, Width] scaled 0.0 to 1.0
            img_tensor = frame["observation.images.wrist_camera"]
            # Convert to numpy, scale to 0-255, and rearrange to [Height, Width, Channels]
            img_np = (img_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))
            img = Image.fromarray(img_np)
            
            # Save image locally
            img_filename = f"frame_{i:06d}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            img.save(img_path)

            # 3. Get Instruction Task
            # Since the text string was lost during the merge step and there is only 
            # 1 task in the dataset, we can safely hardcode the instruction here.
            instruction = "Move end effector above palm crown, angle end effector downward, and hold while end effector is spraying."
            
            # 4. Normalize Action
            raw_action = frame["action"].numpy()
            denominator = action_max - action_min
            denominator[denominator == 0] = 1e-8 # Prevent divide-by-zero
            
            normalized_action = 2.0 * (raw_action - action_min) / denominator - 1.0
            normalized_action = np.clip(normalized_action, -1.0, 1.0).tolist()

            # 5. Create JSONL entry
            entry = {
                "image": img_filename,
                "instruction": instruction,
                "action": normalized_action
            }
            
            f.write(json.dumps(entry) + "\n")

    print(f"\n[SUCCESS] Dataset ready at {output_dir}")
    print("You can now safely run your fine-tuning script!")

if __name__ == "__main__":
    prepare_data()