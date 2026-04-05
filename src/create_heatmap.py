import os
import cv2
import argparse
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from tqdm import tqdm
from PIL import Image


def extract_saliency_from_red_dots(image_path, output_size=(1536, 512), sigma=5):
    """
    Load an image with red dots, detect the dots, and convert them into a saliency map using Gaussians.
    
    Parameters:
        image_path (str): Path to the image with red dots
        output_size (tuple): Desired output resolution of the saliency map
        sigma (float): Standard deviation of Gaussian blur

    Returns:
        torch.FloatTensor: Saliency map of shape output_size normalized to [0,1]
    """
    # Load image in BGR format
    img = cv2.imread(image_path)

    # Convert to HSV to better isolate red
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color range (in HSV) - two ranges because red wraps around
    lower_red1 = np.array([0, 200, 200])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 200, 200])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours (each red dot should ideally be a contour)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create blank saliency map
    sal_map = np.zeros(mask.shape, dtype=np.float32)

    for cnt in contours:
        if cv2.contourArea(cnt) < 20 or cv2.contourArea(cnt) > 200:
            continue
        # Get center of red dot
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Add a dot at (cx, cy)
        if 0 <= cy < sal_map.shape[0] and 0 <= cx < sal_map.shape[1]:
            sal_map[cy, cx] = 1.0

    # Apply Gaussian filter to spread attention
    sal_map = gaussian_filter(sal_map, sigma=sigma)

    # Normalize to [0,1]
    if np.max(sal_map) > 0:
        sal_map = sal_map / np.max(sal_map)

    # Resize to output size
    sal_map = cv2.resize(sal_map, output_size, interpolation=cv2.INTER_AREA)

    # Convert to tensor
    return torch.FloatTensor(sal_map)


def save_saliency_map(tensor_map, save_path):
    """
    Saves a torch.FloatTensor saliency map to disk as a grayscale image.

    Parameters:
        tensor_map (torch.FloatTensor): The saliency map (values in [0,1])
        save_path (str): Output path for the image (should end with .png)
    """
    # Convert tensor to numpy array and scale to 0-255
    np_map = (tensor_map.numpy() * 255).astype(np.uint8)
    img = Image.fromarray(np_map, mode='L')  # 8-bit grayscale
    img.save(save_path)


parser = argparse.ArgumentParser(description="Extract saliency maps from images with red dots.")
parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing red dot images.")
parser.add_argument("--output_dir", type=str, default="saliency_map", help="Path to the output directory for saliency maps.")
def main():
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            saliency_map = extract_saliency_from_red_dots(image_path)
            save_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_saliency.png")
            save_saliency_map(saliency_map, save_path)    


if __name__ == "__main__":
    main()
