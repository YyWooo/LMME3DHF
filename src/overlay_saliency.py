import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def overlay_saliency_map(original_img_path, saliency_img_path, alpha=0.7):
    # Load original image
    orig = cv2.imread(original_img_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # Load saliency map (grayscale)
    sal = cv2.imread(saliency_img_path, cv2.IMREAD_GRAYSCALE)
    # Resize saliency to match original
    sal = cv2.resize(sal, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)
    # Normalize saliency map to 0–255
    sal = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sal_float = sal.astype(np.float32) / 255.0
    sal_gamma = np.power(sal_float, 0.7)  # gamma < 1 brightens mid-tones
    sal = (sal_gamma * 255).astype(np.uint8)


    # Apply colormap to saliency cv2.COLORMAP_JET cv2.COLORMAP_HOT
    heatmap = cv2.applyColorMap(sal, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend heatmap over the original image
    blended = cv2.addWeighted(orig, 1 - alpha, heatmap, alpha, 0)

    # Show result
    plt.figure(figsize=(10, 5))
    plt.imshow(blended)
    plt.axis("off")
    plt.show()

    cv2.imwrite("saliency_overlay.png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def overlay_saliency_with_transparency(original_img_path, saliency_img_path, threshold=10, alpha=1.0):
    """
    Overlay only the non-black regions of a saliency map (above a threshold) onto an image.

    Parameters:
        original_img_path (str): Path to the RGB image.
        saliency_img_path (str): Path to the predicted saliency map (grayscale).
        threshold (int): Pixel intensity threshold to define attention (0–255).
        alpha (float): Transparency level for overlay.
    """
    # Load original and saliency images
    image = cv2.imread(original_img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sal = cv2.imread(saliency_img_path, cv2.IMREAD_GRAYSCALE)
    sal = cv2.resize(sal, (image.shape[1], image.shape[0]))

    # Normalize and threshold saliency to mask attention only
    sal_norm = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask = sal_norm > threshold

    # Apply color map only to the salient regions
    heatmap = cv2.applyColorMap(sal_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Create overlay: only blend attention pixels
    blended = image.copy()
    blended[mask] = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)[mask]

    return blended


if __name__ == "__main__":
    image_dir = r"database/image"
    saliency_dir = r"database/sal_predict"

    output_dir = "distortion_predict"
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(os.listdir(image_dir)):
        if i.endswith('.png'):
            image_path = os.path.join(image_dir, i)
            saliency_path = os.path.join(saliency_dir, i)
            # Overlay and save
            overlay_saliency_map(image_path, saliency_path)
            
            # blended = overlay_saliency_with_transparency(image_path, saliency_path, threshold=10)
            # cv2.imwrite(os.path.join(output_dir, f"{i}"), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    