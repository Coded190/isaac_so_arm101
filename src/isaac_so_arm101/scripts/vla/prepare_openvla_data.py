import os
import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def prepare_data():
    hf_repo_id = "coded190/isaac_so_arm101_vla" # Ensure this is your correct repo
    output_dir = "vla_finetune_data"
    images_dir = os.path.join(output_dir, "images")
    jsonl_path = os.path.join(output_dir, "dataset.jsonl")
    stats_path = os.path.join(output_dir, "action_stats.json")

    os.makedirs(images_dir, exist_ok=True)

    print(f"Downloading/Loading dataset {hf_repo_id}...")
    dataset = load_dataset(hf_repo_id, split="train")

    # --- STEP 1: Compute Global Normalization Statistics ---
    print("Analyzing action values for normalization...")
    all_actions = []
    
    # We use tqdm to show a progress bar
    for row in tqdm(dataset, desc="Reading actions"):
        all_actions.append(row["action"])
    
    all_actions = np.array(all_actions) # Shape: (Total_Frames, Action_Dim)
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

    # Save stats so you can un-normalize during simulation inference!
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
        for i, row in enumerate(tqdm(dataset, desc="Processing frames")):
            # Retrieve data from HF format
            img = row["observation.images.wrist_camera"]
            instruction = row["task"]
            raw_action = np.array(row["action"])
            
            # Normalization Formula: Map [min, max] perfectly to [-1.0, 1.0]
            denominator = action_max - action_min
            # Prevent divide-by-zero for constant actions (e.g., if gripper never moves)
            denominator[denominator == 0] = 1e-8 
            
            normalized_action = 2.0 * (raw_action - action_min) / denominator - 1.0
            
            # Clip to exactly [-1.0, 1.0] to safeguard against microscopic float math rounding errors
            normalized_action = np.clip(normalized_action, -1.0, 1.0).tolist()

            # Save image locally
            img_filename = f"frame_{i:06d}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            img.save(img_path)

            # Create JSONL entry
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