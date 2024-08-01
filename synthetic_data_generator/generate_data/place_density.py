"""
place_density.py

This module implements a density-based approach for generating synthetic images
of larvae on a background. It uses adaptive Poisson disc sampling to determine
larvae positions based on a generated density map.
"""

import cv2
import numpy as np
import random
import json
from pathlib import Path
from scipy.stats import multivariate_normal, beta
from .utils import load_coco_annotations, create_placement_mask
from .mask import create_larva_mask, apply_mask

def generate_larva_count():
    """
    Generate a random number of larvae between 10 and 300, favoring middle values.

    Returns:
        int: The number of larvae to generate.
    """
    return int(beta.rvs(2, 2) * 290 + 10)

def generate_density_map(shape, num_centers=15, sigma_range=(20, 80)):
    """
    Generate a density map for larvae placement using multiple Gaussian distributions.

    Args:
        shape (tuple): Shape of the density map (height, width).
        num_centers (int, optional): Number of Gaussian centers. Defaults to 15.
        sigma_range (tuple, optional): Range of sigma values for Gaussians. Defaults to (20, 80).

    Returns:
        numpy.ndarray: Normalized density map.
    """
    density = np.zeros(shape)
    for _ in range(num_centers):
        center = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
        sigma = np.random.uniform(*sigma_range)
        x, y = np.mgrid[0:shape[0], 0:shape[1]]
        rv = multivariate_normal(center, cov=sigma)
        density += rv.pdf(np.dstack((x, y)))
    return density / density.sum()

def adaptive_poisson_disc_sampling(density_map, min_radius, max_radius, mask, k=30):
    """
    Implement adaptive Poisson disc sampling to determine larvae positions based on the density map.

    Args:
        density_map (numpy.ndarray): Density map for larvae placement.
        min_radius (float): Minimum radius between samples.
        max_radius (float): Maximum radius between samples.
        mask (numpy.ndarray): Binary mask indicating valid placement areas.
        k (int, optional): Number of attempts to place a sample around an existing one. Defaults to 30.

    Returns:
        list: List of (y, x) coordinates for larvae placement.
    """
    def get_cell_indices(pt):
        return int(pt[0] / cell_size), int(pt[1] / cell_size)

    def get_neighbours(pt):
        idx = get_cell_indices(pt)
        indices = [(idx[0] + i, idx[1] + j) for i in range(-2, 3) for j in range(-2, 3)]
        return [grid[i][j] for i, j in indices if 0 <= i < grid_height and 0 <= j < grid_width]

    def is_valid(pt, r):
        if not (0 <= pt[0] < height and 0 <= pt[1] < width):
            return False
        if mask[int(pt[0]), int(pt[1])] == 0:  # Check if point is within the mask
            return False
        cell_indices = get_cell_indices(pt)
        for neighbor in get_neighbours(pt):
            if neighbor is not None:
                neighbor_pt, neighbor_r = neighbor
                dist = np.linalg.norm(np.array(neighbor_pt) - np.array(pt))
                min_dist = (r + neighbor_r) / 2
                if dist < min_dist:
                    return False
        return True

    height, width = density_map.shape
    cell_size = min_radius / np.sqrt(2)
    grid_height, grid_width = int(height / cell_size) + 1, int(width / cell_size) + 1
    grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]

    sample_points = []
    process_list = []

    # Find a valid initial point
    while True:
        initial_point = (np.random.randint(0, height), np.random.randint(0, width))
        if density_map[initial_point[0], int(initial_point[1])] > 0 and mask[initial_point[0], int(initial_point[1])] > 0:
            break

    initial_radius = min_radius + (max_radius - min_radius) * (1 - density_map[initial_point[0], int(initial_point[1])])
    process_list.append(initial_point)
    sample_points.append((initial_point, initial_radius))
    grid[get_cell_indices(initial_point)[0]][get_cell_indices(initial_point)[1]] = (initial_point, initial_radius)

    while process_list:
        point = process_list.pop(random.randint(0, len(process_list) - 1))
        for _ in range(k):
            density_factor = density_map[int(point[0]), int(point[1])]
            r = min_radius + (max_radius - min_radius) * (1 - density_factor)
            theta = np.random.uniform(0, 2 * np.pi)
            new_point = (point[0] + r * np.cos(theta), point[1] + r * np.sin(theta))
            if is_valid(new_point, r):
                process_list.append(new_point)
                sample_points.append((new_point, r))
                grid[get_cell_indices(new_point)[0]][get_cell_indices(new_point)[1]] = (new_point, r)

    return [(int(pt[0]), int(pt[1])) for pt, _ in sample_points]

def place_larvae(placement_mask, num_larvae, min_larva_size, max_larva_size):
    """
    Determine the final positions of larvae based on the density map and placement mask.

    Args:
        placement_mask (numpy.ndarray): Binary mask indicating valid placement areas.
        num_larvae (int): Number of larvae to place.
        min_larva_size (int): Minimum size of a larva.
        max_larva_size (int): Maximum size of a larva.

    Returns:
        list: List of (y, x) coordinates for larvae placement.
    """
    density_map = generate_density_map(placement_mask.shape)
    density_map *= placement_mask  # Apply the placement mask
    
    # Create a smaller valid area by eroding the placement mask
    kernel = np.ones((max_larva_size, max_larva_size), np.uint8)  # Use max_larva_size instead of min_larva_size
    eroded_mask = cv2.erode(placement_mask.astype(np.uint8), kernel, iterations=1)
    
    # Increase the minimum radius to reduce overlapping
    min_radius = max_larva_size * 0.3  # Adjust this value as needed
    max_radius = max_larva_size
    
    points = adaptive_poisson_disc_sampling(density_map, min_radius, max_radius, eroded_mask)
    
    # Filter points to only those within the eroded mask
    valid_points = [pt for pt in points if eroded_mask[pt[0], pt[1]] == 1]
    
    if len(valid_points) < num_larvae:
        print(f"Warning: Only {len(valid_points)} valid positions found for {num_larvae} larvae.")
        num_larvae = len(valid_points)
    
    # Randomly select the required number of points
    selected_points = random.sample(valid_points, num_larvae)
    
    return selected_points

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

    This function loads a background image, generates a density map for larvae placement,
    places larvae on the background, and generates COCO format annotations for the synthetic images.

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

    # Determine the minimum and maximum larva sizes
    min_larva_size = float('inf')
    max_larva_size = 0
    for larva_path in healthy_larvae + dead_larvae:
        larva = cv2.imread(str(larva_path), cv2.IMREAD_UNCHANGED)
        if larva is not None:
            larva_size = max(larva.shape[:2])
            min_larva_size = min(min_larva_size, larva_size)
            max_larva_size = max(max_larva_size, larva_size)

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
        
        larvae_positions = place_larvae(placement_mask, num_larvae, min_larva_size, max_larva_size)
        
        image_annotations = []

        for larva_id, (y_pos, x_pos) in enumerate(larvae_positions):
            is_healthy = random.choice([True, False])
            larva_path = random.choice(healthy_larvae if is_healthy else dead_larvae)
            larva = cv2.imread(str(larva_path), cv2.IMREAD_UNCHANGED)
            
            if larva is None:
                print(f"Warning: Could not load larva image from {larva_path}")
                continue
            
            scale = random.uniform(0.9, 1.1)
            larva_resized = cv2.resize(larva, None, fx=scale, fy=scale)
            
            # Check if the larva fits within the placement mask
            larva_height, larva_width = larva_resized.shape[:2]
            if (y_pos + larva_height <= placement_mask.shape[0] and 
                x_pos + larva_width <= placement_mask.shape[1] and
                np.all(placement_mask[y_pos:y_pos+larva_height, x_pos:x_pos+larva_width])):
                
                new_image = apply_mask(new_image, larva_resized, create_larva_mask(larva_resized), (int(y_pos), int(x_pos)))
                
                annotation = create_coco_annotation(larva_resized, (int(y_pos), int(x_pos)), is_healthy, sample_id, annotation_id)
                image_annotations.append(annotation)
                annotation_id += 1
            else:
                print(f"Warning: Larva at position ({x_pos}, {y_pos}) doesn't fit within the placement mask. Skipping.")

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
