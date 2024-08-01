from pathlib import Path
from generate_data.place_density import add_larvae_to_background
from generate_data.utils import visualize_masks

def generate_synthetic_images(background_path, background_anno_path, larvae_dir, output_dir, num_samples=10):
    add_larvae_to_background(
        background_path,
        background_anno_path,
        larvae_dir,
        output_dir,
        num_samples=num_samples
    )
    print(f"Generated {num_samples} synthetic images in {output_dir}")

def visualize_sample(output_dir):
    output_dir = Path(output_dir)  # Ensure output_dir is a Path object
    image_path = next(output_dir.glob("synthetic_image_*.png"))
    annotation_path = output_dir / "synthetic_annotations.json"
    visualization_path = output_dir / "visualization.png"
    
    visualize_masks(str(image_path), str(annotation_path), str(visualization_path))
    print(f"Visualization saved to {visualization_path}")

if __name__ == "__main__":
    background_path = "data/background/background.jpg"
    background_anno_path = "data/background/_annotations.coco.json"
    larvae_dir = "data/larvae"
    output_dir = Path("data/generated/test_density")  # Use Path object here

    # Generate images
    generate_synthetic_images(background_path, background_anno_path, larvae_dir, output_dir, num_samples=5)

    # Visualize all generated images
    for image_path in output_dir.glob("synthetic_image_*.png"):
        visualization_path = image_path.with_name(f"{image_path.stem}_visualization.png")
        visualize_masks(str(image_path), str(output_dir / "synthetic_annotations.json"), str(visualization_path))
        print(f"Visualization saved to {visualization_path}")
