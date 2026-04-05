import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Convert saliency maps to binary fixation maps.")
parser.add_argument("--saliency_dir", type=str, required=True, help="Path to the directory containing saliency maps.")
parser.add_argument("--output_dir", type=str, default="fixation_map", help="Path to the output directory for fixation maps.")
def convert_saliency_to_fixation(threshold_percentile=95):
    args = parser.parse_args()
    saliency_dir = args.saliency_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    saliency_files = [f for f in os.listdir(saliency_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in tqdm(saliency_files):
        sal_path = os.path.join(saliency_dir, fname)
        out_path = os.path.join(output_dir, fname)

        # Load saliency map
        sal_img = Image.open(sal_path).convert('L')
        sal_tensor = TF.to_tensor(sal_img).squeeze(0)  # shape: [H, W], values 0–1

        # Check if the saliency map has any variation
        if torch.all(sal_tensor == sal_tensor[0, 0]):
            # Flat image — skip or save a black map
            fixation_map = torch.zeros_like(sal_tensor)
        else:
            # Normalize to [0, 1]
            sal_norm = (sal_tensor - sal_tensor.min()) / (sal_tensor.max() - sal_tensor.min() + 1e-8)

            # Compute threshold from top x% pixels
            thresh = torch.quantile(sal_norm.flatten(), threshold_percentile / 100.0)

            # Binarize
            fixation_map = (sal_norm >= 0.95).float()

        # Convert to 0-255 for saving
        fixation_img = Image.fromarray((fixation_map * 255).byte().numpy())
        fixation_img.save(out_path)

    print(f"\n Done. Converted {len(saliency_files)} saliency maps to binary fixation maps.")


if __name__ == "__main__":
    convert_saliency_to_fixation()
