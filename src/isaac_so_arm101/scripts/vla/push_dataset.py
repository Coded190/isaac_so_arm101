from lerobot.datasets.lerobot_dataset import LeRobotDataset
import argparse

def push_to_hf(hf_username, dataset_name):
    local_repo_id = "local/merged_vla_dataset"
    hub_repo_id = f"{hf_username}/{dataset_name}"
    
    print(f"Loading local dataset: {local_repo_id}...")
    # Load the local dataset we just created
    ds = LeRobotDataset(local_repo_id)
    
    print(f"Pushing to Hugging Face at: {hub_repo_id}...")
    print("This may take a while depending on the size of your videos and your internet connection.")
    
    # Push to Hugging Face
    ds.push_to_hub(hub_repo_id)
    
    print(f"\n[SUCCESS] Dataset uploaded to: https://huggingface.co/datasets/{hub_repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, required=True, help="Your Hugging Face username")
    parser.add_argument("--name", type=str, default="isaac_so_arm101_vla", help="What to name the dataset on HF")
    
    args = parser.parse_args()
    push_to_hf(args.user, args.name)