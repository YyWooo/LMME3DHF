import os
import argparse
from huggingface_hub import snapshot_download

def main(
    target: str,
    dataset_list: list
):
    os.makedirs(target, exist_ok=True)
    ignore_datasets = set(["image", "video_RGB", "saliency_map"]) - set(dataset_list)
    ignore_pattern = [f"{ds}.zip" for ds in ignore_datasets]

    print(f"Downloading to \"{target}\" ...")
    snapshot_download(
        repo_id="yywooo/Gen3DHF",
        repo_type="dataset",
        local_dir=str(target),
        ignore_patterns=ignore_pattern,
        resume_download=True,
        max_workers=8
    )
    print(f"Downloaded to \"{target}\"")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=False, default="Gen3DHF", help="Directory to download the dataset.")
    parser.add_argument('--dataset', type=str, required=False, default="image video_RGB saliency_map", help="Specify which dataset to download.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset_list = [ds.strip() for ds in args.dataset.split(' ')]
    main(args.target_dir, dataset_list)
