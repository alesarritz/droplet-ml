import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Download the model at https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

def show_image_mask_overlap(image, masks):
    # Create an overlay mask
    overlay = np.zeros_like(image, dtype=np.uint8)
    for mask in masks:
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)  # pick random RGB color
        overlay[mask["segmentation"]] = color  # Apply color to mask

    # Blend the image and the mask
    alpha = 0.5  # Transparency factor
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.imshow(blended)
    plt.axis("off")
    plt.title("Segmented Mask Overlaid on Image")
    plt.show()

def generate_masks(img_path: str):
    # Load image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

    # Load the SAM model and generate segmentation masks
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    return masks, image

def save_mask(masks, image, save_path):
    """Save a colored mask for an image."""
    mask_img = np.zeros_like(image, dtype=np.uint8)  # Create a blank mask with the same size as the image
    for mask in masks:
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # Generate a random RGB color
        mask_img[mask["segmentation"]] = color  # Apply color to mask

    cv2.imwrite(save_path, mask_img)  # Save mask as PNG


input_folder = "dataset_120_real"
output_folder = "dataset_120_real_masks"
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

for filename in os.listdir(input_folder):
    masks, image = generate_masks(os.path.join(input_folder, filename))
    save_mask(masks, image, os.path.join(output_folder, filename))