# Configuration Guide for Synthetic Larvae Image Generator

This guide provides detailed information about all configurable parameters in the Synthetic Larvae Image Generator. Understanding these parameters will allow you to customize the image generation process to suit your specific needs.

## General Configuration

These parameters apply to all placement strategies:

### num_samples
- **Description**: Number of synthetic images to generate
- **Default value**: 10
- **Acceptable range**: Positive integer
- **Effect**: Determines how many synthetic images are produced in one run

### larvae_count
- **Description**: Number of larvae to place in each image
- **Default value**: Generated randomly between 10 and 300 using a beta distribution
- **Acceptable range**: 10 to 300
- **Effect**: Determines the density of larvae in each image
- **Note**: The beta distribution favors middle values, providing a more natural distribution

### larva_scale
- **Description**: Scale factor applied to each larva image
- **Default value**: Random value between 0.9 and 1.1
- **Acceptable range**: Positive float
- **Effect**: Adds size variation to larvae, increasing realism

## Cell Placement Strategy

### num_iterations
- **Description**: Number of iterations for the cellular automaton
- **Default value**: 100
- **Acceptable range**: Positive integer
- **Effect**: Higher values lead to more spread-out distributions
- **Note**: Very high values may significantly increase computation time

### reproduction_rate
- **Description**: Rate at which new larvae appear in neighboring cells
- **Default value**: 0.3
- **Acceptable range**: 0.0 to 1.0
- **Effect**: Higher values create denser clusters of larvae

## Density Placement Strategy

### num_centers
- **Description**: Number of density centers
- **Default value**: 15
- **Acceptable range**: Positive integer
- **Effect**: More centers create a more varied density distribution

### sigma_range
- **Description**: Range of standard deviations for Gaussian distributions
- **Default value**: (20, 80)
- **Acceptable range**: Tuple of positive floats
- **Effect**: Larger values create more spread-out distributions

### min_radius
- **Description**: Minimum radius between larvae in Poisson disc sampling
- **Default value**: max_larva_size * 0.3
- **Acceptable range**: Positive float
- **Effect**: Smaller values allow larvae to be placed closer together

### max_radius
- **Description**: Maximum radius between larvae in Poisson disc sampling
- **Default value**: max_larva_size
- **Acceptable range**: Positive float, greater than min_radius
- **Effect**: Larger values create more spacing between larvae

## Gravity Placement Strategy

### num_gravity_points
- **Description**: Number of gravity points
- **Default value**: Random integer between 1 and 10
- **Acceptable range**: Positive integer
- **Effect**: More gravity points create more varied larvae distributions

### sigma
- **Description**: Standard deviation for the multivariate normal distribution
- **Default value**: 200
- **Acceptable range**: Positive float
- **Effect**: Larger values create more spread-out distributions around gravity points

### overlap_threshold
- **Description**: Maximum allowed overlap between larvae
- **Default value**: 0.2
- **Acceptable range**: 0.0 to 1.0
- **Effect**: Higher values allow more overlap between larvae

## Cluster Placement Strategy

### num_clusters
- **Description**: Number of cluster centers
- **Default value**: max(1, num_larvae // 50)
- **Acceptable range**: Positive integer
- **Effect**: More clusters create more distributed groupings of larvae

### cluster_std
- **Description**: Standard deviation for normal distribution around cluster centers
- **Default value**: Calculated based on image size and number of clusters
- **Acceptable range**: Positive float
- **Effect**: Larger values create more spread-out clusters

## Usage Examples

To use these configurations, you can modify the `add_larvae_to_background` function calls in your scripts. Here are examples for each placement strategy:

### Cell Placement Strategy
```python
from generate_data.place_cell import add_larvae_to_background

add_larvae_to_background(
    background_path,
    background_anno_path,
    larvae_dir,
    output_dir,
    num_samples=15,
    num_iterations=150,
    reproduction_rate=0.4
)
