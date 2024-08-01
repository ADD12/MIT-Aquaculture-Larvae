"""
place_gravity.py

This module implements a gravity-based approach for generating synthetic images
of larvae on a background. It uses gravity points to create a probability distribution
for larvae placement.
"""

import cv2
import numpy as np
import random
import json
from pathlib import Path
from scipy.stats import beta, multivariate_normal
from .utils import load_coco_annotations, create_placement_mask
from .mask import create_larva_mask, apply_mask


def generate_larva_count():
    """
    Generate a random number of larvae between 10 and 300 using a normal distribution.

    Returns:
        int: The number of larvae to generate.
    """
    mean = 150
    std_dev = 60  # This is a moderately large SD; adjust as needed
    
    while True:
        count = int(np.random.normal(mean, std_dev))
        if 10 <= count <= 300:
            return count

def generate_gravity_points(placement_mask, num_points):
    """
    Generate random gravity points within the placement mask.

    Args:
        placement_mask (numpy.ndarray): Binary mask indicating valid placement areas.
        num_points (int): Number of gravity points to generate.

    Returns:
        numpy.ndarray: Array of gravity points.
    """
    valid_positions = np.argwhere(placement_mask == 1)
    points = valid_positions[np.random.choice(len(valid_positions), num_points, replace=False)]
    return points

def calculate_probability_distribution(placement_mask, gravity_points, sigma=200):
    """
    Calculate the probability distribution for larvae placement based on gravity points.

    Args:
        placement_mask (numpy.ndarray): Binary mask indicating valid placement areas.
        gravity_points (numpy.ndarray): Array of gravity points.
        sigma (float, optional): Standard deviation for the multivariate normal distribution. Defaults to 200.

    Returns:
        numpy.ndarray: Probability distribution for larvae placement.
    """
    y_coords, x_coords = np.mgrid[0:placement_mask.shape[0], 0:placement_mask.shape[1]]
    pos = np.dstack((y_coords, x_coords))
    probability_map = np.zeros_like(placement_mask, dtype=float)
    
    for point in gravity_points:
        rv = multivariate_normal(point, [[sigma, 0], [0, sigma]])
        probability_map += rv.pdf(pos)
    
    probability_map /= len(gravity_points)
    probability_map *= placement_mask  # Ensure probabilities are 0 outside the placement mask
    probability_map /= probability_map.sum()  # Normalize to sum to 1
    
    return probability_map

def check_overlap(position, larva_shape, placed_larvae, overlap_threshold=0.2):
    """
    Check if a new larva placement overlaps with existing larvae.

    Args:
        position (tuple): (y, x) position of the new larva.
        larva_shape (tuple): Shape of the larva image.
        placed_larvae (list): List of already placed larvae.
        overlap_threshold (float, optional): Maximum allowed overlap ratio. Defaults to 0.2.

    Returns:
        bool: True if there's significant overlap, False otherwise.
    """
    y, x = position
    h, w = larva_shape[:2]
    new_rect = [x, y, x + w, y + h]
    
    for placed_rect in placed_larvae:
        px, py, pw, ph = placed_rect
        if (x < px + pw and x + w > px and y < py + ph and y + h > py):
            # Calculate overlap area
            overlap_x = max(0, min(x + w, px + pw) - max(x, px))
            overlap_y = max(0, min(y + h, py + ph) - max(y, py))
            overlap_area = overlap_x * overlap_y
            min_area = min((w * h), (pw * ph))
            if overlap_area / min_area > overlap_threshold:
                return True
    return False

def find_valid_position(placement_mask, larva_shape, probability_map, placed_larvae, max_attempts=100):
    """
    Find a valid position for a larva based on the probability map and existing placements.

    Args:
        placement_mask (numpy.ndarray): Binary mask indicating valid placement areas.
        larva_shape (tuple): Shape of the larva image.
        probability_map (numpy.ndarray): Probability distribution for larvae placement.
        placed_larvae (list): List of already placed larvae.
        max_attempts (int, optional): Maximum number of attempts to find a valid position. Defaults to 100.

    Returns:
        tuple or None: (y, x) position if found, None otherwise.
    """
    bg_height, bg_width = placement_mask.shape[:2]
    
    for _ in range(max_attempts):
        # Choose position based on probability distribution
        position = np.unravel_index(np.random.choice(probability_map.size, p=probability_map.ravel()), probability_map.shape)
        y_pos, x_pos = position
        
        # Ensure position is within image bounds
        if y_pos + larva_shape[0] > bg_height or x_pos + larva_shape[1] > bg_width:
            continue
        
        larva_region = placement_mask[y_pos:y_pos+larva_shape[0], x_pos:x_pos+larva_shape[1]]
        if np.all(larva_region == 1) and not check_overlap((y_pos, x_pos), larva_shape, placed_larvae, overlap_threshold=0.2):
            return y_pos, x_pos
    
    return None

def place_larva(background, larva, position):
    """
    Place a larva on the background image at the specified position.

    Args:
        background (numpy.ndarray): Background image.
        larva (numpy.ndarray): Larva image.
        position (tuple): (y, x) position to place the larva.

    Returns:
        numpy.ndarray: Updated background image with the larva placed.
    """
    y_pos, x_pos = position
    mask = create_larva_mask(larva)
    return apply_mask(background, larva, mask, (y_pos, x_pos))

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
    
    # Create a more accurate polygon representation of the larva
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

    This function loads a background image, generates gravity points for larvae placement,
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
        
        num_gravity_points = random.randint(1, 10)
        gravity_points = generate_gravity_points(placement_mask, num_gravity_points)
        probability_map = calculate_probability_distribution(placement_mask, gravity_points)
        
        image_annotations = []
        placed_larvae = []

        for larva_id in range(num_larvae):
            is_healthy = random.choice([True, False])
            larva_path = random.choice(healthy_larvae if is_healthy else dead_larvae)
            larva = cv2.imread(str(larva_path), cv2.IMREAD_UNCHANGED)
            
            scale = random.uniform(0.9, 1.1)
            larva_resized = cv2.resize(larva, None, fx=scale, fy=scale)

            position = find_valid_position(placement_mask, larva_resized.shape, probability_map, placed_larvae)
            if position is None:
                print(f"  Could not place larva {larva_id} in sample {sample_id}")
                continue

            y_pos, x_pos = position
            placed_larvae.append([x_pos, y_pos, larva_resized.shape[1], larva_resized.shape[0]])
            new_image = place_larva(new_image, larva_resized, position)
            
            annotation = create_coco_annotation(larva_resized, position, is_healthy, sample_id, annotation_id)
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
