# Synthetic Larvae Image Generator

## Project Overview

This project generates synthetic images of larvae on microscope backgrounds for training machine learning models. It creates realistic distributions of healthy and dead larvae, simulating various clustering patterns and densities.

## Key Features

- Generate synthetic images with configurable larvae counts and distributions
- Support for multiple placement strategies (density-based, gravity-based, cluster-based)
- COCO format annotation generation for each synthetic image
- Visualization tools for generated images and annotations

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/synthetic-larvae-generator.git
cd synthetic-larvae-generator

2. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. Install the required packages:

pip install -r requirements.txt

## Quick Start

1. Ensure you have a background image and its COCO annotations in the `data/background/` directory.
2. Place larva images in `data/larvae/healthy/` and `data/larvae/dead/` directories.
3. Run the main script:

4. Generated images and annotations will be saved in the `data/generated/` directory.

## Configuration
The project offers various configurable parameters for each placement strategy. For a comprehensive guide on these parameters, please refer to the configuration_guide.md file.

## Project Structure

synthetic-larvae-generator/
├── data/
│   ├── background/
│   │   ├── background.jpg
│   │   └── _annotations.coco.json
│   ├── larvae/
│   │   ├── healthy/
│   │   └── dead/
│   └── generated/
├── generate_data/
│   ├── init.py
│   ├── extract.py
│   ├── mask.py
│   ├── place_cell.py
│   ├── place_density.py
│   ├── place_gravity.py
│   ├── place.py
│   └── utils.py
├── configuration_guide.md
├── examples.py
├── main.py
├── README.md
└── requirements.txt

## Usage

To generate synthetic images, run the `main.py` script:

By default, this will:
1. Load the background image from `data/background/background.jpg`
2. Use annotations from `data/background/_annotations.coco.json`
3. Place larvae images from `data/larvae/`
4. Generate 5 synthetic images in `data/generated/test_density/`
5. Create visualizations for each generated image

## Key Components

1. `extract.py`: Extracts individual larvae from source images
2. `mask.py`: Handles creation and application of masks
3. `place_density.py`: Implements density-based larvae placement
4. `place_gravity.py`: Implements gravity-based larvae placement
5. `place_cluster.py`: Implements cluster-based larvae placement
6. `place_cell.py`: Implements cell-based larvae placement
7. `utils.py`: Provides utility functions for annotations and visualization

## Examples

Here's an example of how to use the `generate_synthetic_images` function:

```python
from generate_data.place_density import add_larvae_to_background

add_larvae_to_background(
    background_path="data/background/background.jpg",
    background_anno_path="data/background/_annotations.coco.json",
    larvae_dir="data/larvae",
    output_dir="data/generated/custom_output",
    num_samples=10
)

This will generate 10 synthetic images using the density-based placement strategy.
