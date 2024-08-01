"""
extract.py

This module provides functionality to extract individual larvae images from a larger microscope image
using COCO format annotations.
"""

import json
import cv2
import numpy as np
from pathlib import Path

def load_coco_annotations(json_path):
    """
    Load COCO format annotations from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing COCO annotations.

    Returns:
        dict: Loaded COCO annotations.
    """
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_larva(image_path, annotations_path, output_dir='data/larvae'):
    """
    Extract individual larvae from a microscope image using COCO annotations.

    This function processes a microscope image, extracts individual larvae based on 
    the provided COCO annotations, and saves them as separate PNG images with transparency.

    Args:
        image_path (str): Path to the input microscope image.
        annotations_path (str): Path to the COCO annotation JSON file.
        output_dir (str, optional): Directory to save extracted larvae images. Defaults to 'data/larvae'.

    Raises:
        ValueError: If the input image cannot be loaded or if no larvae are found in the annotations.
    """
    # Load the image
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Load annotations
    coco_data = load_coco_annotations(annotations_path)

    # Create category mapping
    category_mapping = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Ensure output directories exist
    output_dir = Path(output_dir)
    (output_dir / 'healthy').mkdir(parents=True, exist_ok=True)
    (output_dir / 'dead').mkdir(parents=True, exist_ok=True)

    # Extract each annotated region
    for i, ann in enumerate(coco_data['annotations']):
        # Get bounding box
        x, y, w, h = map(int, ann['bbox'])

        # Create a mask from the segmentation
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(ann['segmentation'][0]).reshape(-1, 2).astype(int)], 255)

        # Extract the region and its mask
        larva_region = image[y:y+h, x:x+w]
        larva_mask = mask[y:y+h, x:x+w]

        # Create an empty RGBA image
        larva_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Copy RGB channels from the original image
        larva_rgba[:, :, :3] = larva_region

        # Use the mask as the alpha channel
        larva_rgba[:, :, 3] = larva_mask

        # Determine category (healthy or dead)
        category = category_mapping[ann['category_id']]
        if category.lower() == 'healthy':
            save_dir = output_dir / 'healthy'
        elif category.lower() == 'dead':
            save_dir = output_dir / 'dead'
        else:
            print(f"Unknown category {category} for larva {i}, skipping...")
            continue

        # Save the extracted larva with transparency
        output_path = save_dir / f"larva_{i}.png"
        cv2.imwrite(str(output_path), larva_rgba)

        print(f"Extracted {category} larva saved to {output_path}")

# Usage example
if __name__ == "__main__":
    image_path = "data/larvae/train/larvae.jpg"
    annotations_path = "data/larvae/train/_annotations.coco.json"
    
    extract_larva(image_path, annotations_path)
