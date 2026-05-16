import os
import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

def main(input_dir, output_repo_id):
    # 1. Find all environment directories
    env_dirs = sorted([d for d in os.listdir(input_dir) if d.startswith("env_")])
    if not env_dirs:
        print(f"No environment datasets found in {input_dir}.")
        return

    print(f"Found {len(env_dirs)} environment datasets to merge.")

    # 2. Load all individual datasets
    datasets_to_merge = []
    for env_name in env_dirs:
        env_path = os.path.join(input_dir, env_name)
        try:
            # Match the repo_id pattern used by vla_data_gen.py / vla_data_gen_v2.py
            # ("local/vla_palm_dataset_env_XXXX") so LeRobot doesn't think the
            # local cache is stale and try to re-fetch from HuggingFace.
            env_repo_id = f"local/vla_palm_dataset_{env_name}"
            ds = LeRobotDataset(repo_id=env_repo_id, root=env_path)
            
            if ds.num_episodes > 0:
                datasets_to_merge.append(ds)
                print(f"  -> Loaded {env_name} ({ds.num_episodes} episodes)")
            else:
                print(f"  -> [INFO] Skipping {env_name} (0 episodes).")
                
        except Exception as e:
            print(f"  -> [WARN] Skipping {env_name}, could not load: {e}")

    if not datasets_to_merge:
        print("[ERROR] No valid datasets loaded.")
        return

    print(f"\nSuccessfully loaded {len(datasets_to_merge)} datasets. Merging now...")
    print("This might take a moment as LeRobot re-indexes videos and recalculates statistics...\n")

    # 3. Use the official merge utility
    # Note: LeRobot automatically saves this in its standard local dataset registry
    merged_dataset = merge_datasets(
        datasets=datasets_to_merge, 
        output_repo_id=output_repo_id
    )

    print(f"\n[SUCCESS] Merged dataset created with {merged_dataset.num_episodes} total episodes!")
    print(f"It is saved locally under the repo ID: {output_repo_id}")
    print(f"Location: {merged_dataset.root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/vla_palm_dataset", help="Path containing env_0000, env_0001, etc.")
    parser.add_argument("--repo_id", type=str, default="local/merged_vla_dataset", help="The local repo ID to save under")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.repo_id)