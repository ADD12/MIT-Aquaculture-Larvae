"""
examples.py

This script demonstrates various use cases of the synthetic larvae image generator.
It includes examples of generating images using different placement strategies and configurations.
"""

import os
from pathlib import Path
from generate_data.place_cell import add_larvae_to_background as add_larvae_cell
from generate_data.place_density import add_larvae_to_background as add_larvae_density
from generate_data.place_gravity import add_larvae_to_background as add_larvae_gravity
from generate_data.place_cluster import add_larvae_to_background as add_larvae_cluster
from generate_data.utils import view_random_generated_image, visualize_masks

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
BACKGROUND_PATH = BASE_DIR / "data" / "background" / "background.jpg"
BACKGROUND_ANNO_PATH = BASE_DIR / "data" / "background" / "_annotations.coco.json"
LARVAE_DIR = BASE_DIR / "data" / "larvae"
OUTPUT_DIR = BASE_DIR / "data" / "generated"

def example_cell_placement():
    """
    Generate synthetic images using the cellular automaton placement strategy.
    """
    output_dir = OUTPUT_DIR / "cell_placement"
    add_larvae_cell(BACKGROUND_PATH, BACKGROUND_ANNO_PATH, LARVAE_DIR, output_dir, num_samples=5)
    view_random_generated_image(output_dir, output_dir / "synthetic_annotations.json")

def example_density_placement():
    """
    Generate synthetic images using the density-based placement strategy.
    """
    output_dir = OUTPUT_DIR / "density_placement"
    add_larvae_density(BACKGROUND_PATH, BACKGROUND_ANNO_PATH, LARVAE_DIR, output_dir, num_samples=5)
    view_random_generated_image(output_dir, output_dir / "synthetic_annotations.json")

def example_gravity_placement():
    """
    Generate synthetic images using the gravity-based placement strategy.
    """
    output_dir = OUTPUT_DIR / "gravity_placement"
    add_larvae_gravity(BACKGROUND_PATH, BACKGROUND_ANNO_PATH, LARVAE_DIR, output_dir, num_samples=5)
    view_random_generated_image(output_dir, output_dir / "synthetic_annotations.json")

def example_cluster_placement():
    """
    Generate synthetic images using the cluster-based placement strategy.
    """
    output_dir = OUTPUT_DIR / "cluster_placement"
    add_larvae_cluster(BACKGROUND_PATH, BACKGROUND_ANNO_PATH, LARVAE_DIR, output_dir, num_samples=5)
    view_random_generated_image(output_dir, output_dir / "synthetic_annotations.json")

def example_visualize_masks():
    """
    Visualize the segmentation masks for a generated image.
    """
    output_dir = OUTPUT_DIR / "cell_placement"
    image_path = next(output_dir.glob("synthetic_image_*.png"))
    annotation_path = output_dir / "synthetic_annotations.json"
    visualization_path = output_dir / "mask_visualization.png"
    visualize_masks(str(image_path), str(annotation_path), str(visualization_path))

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating examples using cell placement strategy...")
    example_cell_placement()
    
    print("\nGenerating examples using density placement strategy...")
    example_density_placement()
    
    print("\nGenerating examples using gravity placement strategy...")
    example_gravity_placement()
    
    print("\nGenerating examples using cluster placement strategy...")
    example_cluster_placement()
    
    print("\nVisualizing segmentation masks...")
    example_visualize_masks()
    
    print("\nAll examples have been generated. Check the 'data/generated' directory for results.")
