from lerobot.datasets.lerobot_dataset import LeRobotDataset
import argparse

def push_to_hf(hf_username, dataset_name):
    local_repo_id = "local/merged_vla_dataset"
    hub_repo_id = f"{hf_username}/{dataset_name}"
    
    print(f"Loading local dataset: {local_repo_id}...")
    ds = LeRobotDataset(local_repo_id)
    
    print(f"Pushing to Hugging Face at: {hub_repo_id}...")
    print("This may take a while depending on the size of your videos and your internet connection.")
    
    # --- THE FIX: Overwrite the dataset's internal repo_id ---
    ds.repo_id = hub_repo_id 
    
    # Push to Hugging Face (no arguments needed, it uses ds.repo_id)
    ds.push_to_hub()
    # ---------------------------------------------------------
    
    print(f"\n[SUCCESS] Dataset uploaded to: https://huggingface.co/datasets/{hub_repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, required=True, help="Your Hugging Face username")
    parser.add_argument("--name", type=str, default="isaac_so_arm101_vla", help="What to name the dataset on HF")
    
    args = parser.parse_args()
    push_to_hf(args.user, args.name)