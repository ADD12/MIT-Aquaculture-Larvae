"""
place_cell.py

This module implements a cellular automaton approach for generating synthetic images
of larvae on a background. It simulates larvae spread and placement using a grid-based method.
"""

import cv2
import numpy as np
import random
import json
from pathlib import Path
from scipy.stats import beta
from scipy.ndimage import convolve
from .utils import load_coco_annotations, create_placement_mask
from .mask import create_larva_mask, apply_mask

def generate_larva_count():
    """
    Generate a random number of larvae between 10 and 300, favoring middle values.

    Returns:
        int: The number of larvae to generate.
    """
    return int(beta.rvs(2, 2) * 290 + 10)

def initialize_grid(placement_mask, num_initial_seeds=10):
    """
    Initialize a grid with random seed positions for larvae placement.

    Args:
        placement_mask (numpy.ndarray): Binary mask indicating valid placement areas.
        num_initial_seeds (int, optional): Number of initial seed points. Defaults to 10.

    Returns:
        numpy.ndarray: Initialized grid with seed points.
    """
    grid = np.zeros_like(placement_mask, dtype=float)
    valid_positions = np.argwhere(placement_mask == 1)
    seed_positions = valid_positions[np.random.choice(len(valid_positions), num_initial_seeds, replace=False)]
    for pos in seed_positions:
        grid[pos[0], pos[1]] = 1
    return grid

def simulate_larvae_spread(placement_mask, num_larvae, num_iterations=100):
    """
    Simulate the spread of larvae using a cellular automaton approach.

    Args:
        placement_mask (numpy.ndarray): Binary mask indicating valid placement areas.
        num_larvae (int): Target number of larvae to simulate.
        num_iterations (int, optional): Number of simulation iterations. Defaults to 100.

    Returns:
        numpy.ndarray: Grid representing the density of larvae.
    """
    grid = initialize_grid(placement_mask)
    
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
    
    for _ in range(num_iterations):
        # Reproduction
        neighbors = convolve(grid, kernel, mode='constant', cval=0.0)
        reproduction_prob = (1 - grid) * np.power(neighbors, 2) * 0.3  # Adjusted to favor clustering
        grid += (np.random.random(grid.shape) < reproduction_prob).astype(float)
        
        # Movement (reduced to favor staying in clusters)
        movement = np.random.normal(0, 0.05, grid.shape)
        grid += movement
        
        # Ensure grid values are between 0 and 1
        grid = np.clip(grid, 0, 1)
        
        # Apply placement mask
        grid *= placement_mask
        
        # Add random new clusters occasionally
        if np.random.random() < 0.1:  # 10% chance each iteration
            new_cluster = initialize_grid(placement_mask, num_initial_seeds=1)
            grid = np.maximum(grid, new_cluster)
    
    # Normalize to get desired number of larvae
    grid *= (num_larvae / grid.sum())
    return grid

def place_larvae(larvae_map, num_larvae):
    """
    Determine the final positions of larvae based on the larvae density map.

    Args:
        larvae_map (numpy.ndarray): Grid representing the density of larvae.
        num_larvae (int): Number of larvae to place.

    Returns:
        tuple: Two numpy arrays representing y and x coordinates of placed larvae.
    """
    probabilities = larvae_map.ravel() / larvae_map.sum()
    positions = np.random.choice(len(probabilities), num_larvae, p=probabilities, replace=False)
    return np.unravel_index(positions, larvae_map.shape)

def create_coco_annotation(larva, position, is_healthy, image_id, annotation_id):
    """
    Create a COCO format annotation for a single larva.

    Args:
        larva (numpy.ndarray): Image of the larva.
        position (tuple): (y, x) position of the larva on the background.
        is_healthy (bool): Whether the larva is healthy or dead.
        image_id (int): ID of the image in the COCO dataset.
        annotation_id (int): Unique ID for this annotation.

    Returns:
        dict: COCO format annotation for the larva.
    """
    y_pos, x_pos = position
    mask = create_larva_mask(larva)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    epsilon = 0.005 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    segmentation = (approx.reshape(-1, 2) + [x_pos, y_pos]).flatten().tolist()
    
    x, y, w, h = cv2.boundingRect(contours[0])
    
    return {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": 1 if is_healthy else 2,
        "bbox": [float(x_pos + x), float(y_pos + y), float(w), float(h)],
        "area": float(cv2.contourArea(contours[0])),
        "segmentation": [[float(x) for x in segmentation]],
        "iscrowd": 0
    }

def add_larvae_to_background(background_path, background_anno_path, larvae_dir, output_dir, num_samples=10):
    """
    Main function to generate synthetic images with larvae on a background.

    This function loads a background image, simulates larvae spread, places larvae on the background,
    and generates COCO format annotations for the synthetic images.

    Args:
        background_path (str): Path to the background image file.
        background_anno_path (str): Path to the background annotations file.
        larvae_dir (str): Directory containing larvae images.
        output_dir (str): Directory to save generated images and annotations.
        num_samples (int, optional): Number of synthetic images to generate. Defaults to 10.
    """
    print(f"Loading background from: {background_path}")
    background = cv2.imread(str(background_path))
    if background is None:
        raise ValueError(f"Could not load background image from {background_path}")
    
    bg_height, bg_width = background.shape[:2]
    print(f"Background size: {bg_width}x{bg_height}")
    
    print(f"Loading background annotations from: {background_anno_path}")
    background_coco = load_coco_annotations(background_anno_path)
    placement_mask = create_placement_mask(background_coco, background.shape)
    print(f"Placement mask shape: {placement_mask.shape}")
    print(f"Placement mask sum: {np.sum(placement_mask)}")

    if np.sum(placement_mask) == 0:
        print("Warning: Placement mask is empty. Using entire image for placement.")
        placement_mask = np.ones_like(placement_mask)

    print(f"Looking for larvae in: {larvae_dir}")
    healthy_larvae = list(Path(larvae_dir, 'healthy').glob('*.png'))
    dead_larvae = list(Path(larvae_dir, 'dead').glob('*.png'))
    print(f"Found {len(healthy_larvae)} healthy larvae and {len(dead_larvae)} dead larvae")

    if not healthy_larvae or not dead_larvae:
        raise ValueError(f"No larvae images found in {larvae_dir}. Please check the directory structure.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "healthy", "supercategory": "larva"},
            {"id": 2, "name": "dead", "supercategory": "larva"}
        ]
    }

    annotation_id = 0

    for sample_id in range(num_samples):
        print(f"Generating sample {sample_id + 1}/{num_samples}")
        new_image = background.copy()
        num_larvae = generate_larva_count()
        print(f"  Placing {num_larvae} larvae")
        
        larvae_map = simulate_larvae_spread(placement_mask, num_larvae)
        larvae_positions = place_larvae(larvae_map, num_larvae)
        
        image_annotations = []

        for larva_id, (y_pos, x_pos) in enumerate(zip(*larvae_positions)):
            is_healthy = random.choice([True, False])
            larva_path = random.choice(healthy_larvae if is_healthy else dead_larvae)
            larva = cv2.imread(str(larva_path), cv2.IMREAD_UNCHANGED)
            
            scale = random.uniform(0.9, 1.1)
            larva_resized = cv2.resize(larva, None, fx=scale, fy=scale)

            new_image = apply_mask(new_image, larva_resized, create_larva_mask(larva_resized), (y_pos, x_pos))
            
            annotation = create_coco_annotation(larva_resized, (y_pos, x_pos), is_healthy, sample_id, annotation_id)
            image_annotations.append(annotation)
            annotation_id += 1

        output_path = output_dir / f"synthetic_image_{sample_id}.png"
        print(f"Saving image to: {output_path}")
        cv2.imwrite(str(output_path), new_image)

        coco_annotations["images"].append({
            "id": int(sample_id),
            "file_name": output_path.name,
            "width": int(bg_width),
            "height": int(bg_height)
        })

        coco_annotations["annotations"].extend(image_annotations)

    with open(output_dir / "synthetic_annotations.json", "w") as f:
        json.dump(coco_annotations, f)

    print(f"Generated {num_samples} synthetic images with annotations")
    print("Finished generating images")
