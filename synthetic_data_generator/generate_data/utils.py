"""
utils.py

This module provides utility functions for loading COCO annotations, creating placement masks,
and visualizing generated images with their annotations.
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
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

def create_placement_mask(coco_data, image_shape):
    """
    Create a binary mask for larva placement based on COCO annotations.

    Args:
        coco_data (dict): COCO format annotations.
        image_shape (tuple): Shape of the image (height, width, channels).

    Returns:
        numpy.ndarray: Binary mask where 1 indicates valid placement areas.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    placement_id = next((cat['id'] for cat in coco_data['categories'] if cat['name'].lower() == 'background' and cat['supercategory'].lower() == 'background'), None)
    print(f"Placement area category ID: {placement_id}")
    
    if placement_id is None:
        print("Warning: No valid placement area category found. Using entire image.")
        return np.ones(image_shape[:2], dtype=np.uint8)
    
    for ann in coco_data['annotations']:
        if ann['category_id'] == placement_id:
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    polygon = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [polygon], 1)
    
    if np.sum(mask) == 0:
        print("Warning: No placement area found in annotations. Using entire image.")
        return np.ones(image_shape[:2], dtype=np.uint8)
    
    print(f"Placement mask sum: {np.sum(mask)}")
    return mask


def view_generated_image(image_path, annotation_path, image_id):
    """
    Display a generated image with its annotations for inspection.
    
    Args:
        image_path (str): Path to the directory containing generated images.
        annotation_path (str): Path to the COCO format annotation JSON file.
        image_id (int): ID of the image to display.
    """
    # Load annotations
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    # Find the image with the given ID
    image_info = next((img for img in annotations['images'] if img['id'] == image_id), None)
    if image_info is None:
        raise ValueError(f"No image found with id {image_id}")
    
    # Load the image
    image = cv2.imread(str(image_path / image_info['file_name']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw annotations
    colors = {'healthy': 'g', 'dead': 'r'}
    for ann in annotations['annotations']:
        if ann['image_id'] == image_id:
            category = next(cat['name'] for cat in annotations['categories'] if cat['id'] == ann['category_id'])
            bbox = ann['bbox']
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                                 fill=False, edgecolor=colors[category], linewidth=2)
            ax.add_patch(rect)
            ax.text(bbox[0], bbox[1], category, color=colors[category], 
                    bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title(f"Generated Image {image_id}")
    plt.axis('off')
    plt.show()

def view_random_generated_image(image_path, annotation_path):
    """
    Display a random generated image with its annotations for inspection.
    
    Args:
        image_path (str): Path to the directory containing generated images.
        annotation_path (str): Path to the COCO format annotation JSON file.
    """
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    image_ids = [img['id'] for img in annotations['images']]
    random_id = np.random.choice(image_ids)
    
    view_generated_image(image_path, annotation_path, random_id)

def visualize_masks(image_path, annotation_path, output_path=None):
    """
    Visualize larva segmentation masks on the generated image.

    Args:
        image_path (str): Path to the generated image file.
        annotation_path (str): Path to the COCO format annotation JSON file.
        output_path (str, optional): Path to save the visualization. If None, the image is displayed instead.
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load the annotations
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    # Get the image id from the file name
    image_file_name = Path(image_path).name
    image_id = next(img['id'] for img in annotations['images'] if img['file_name'] == image_file_name)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Define colors for each category
    colors = {1: (0, 1, 0, 0.3), 2: (1, 0, 0, 0.3)}  # Green for healthy, Red for dead
    
    # Draw annotations only for this specific image
    for ann in annotations['annotations']:
        if ann['image_id'] == image_id:
            category = ann['category_id']
            segmentation = ann['segmentation'][0]
            
            # Draw segmentation mask
            poly = np.array(segmentation).reshape((-1, 2))
            ax.fill(poly[:, 0], poly[:, 1], color=colors[category])
            
            # Add label
            centroid = np.mean(poly, axis=0)
            ax.text(centroid[0], centroid[1], f"{'Healthy' if category == 1 else 'Dead'}", 
                    color='white', weight='bold', ha='center', va='center',
                    bbox=dict(facecolor=colors[category][:3], edgecolor='none', alpha=0.7))
    
    plt.title(f"Generated Image with Larva Masks - {image_file_name}")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()
