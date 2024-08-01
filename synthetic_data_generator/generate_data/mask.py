"""
mask.py

This module provides functions for creating and applying masks to larva images.
It is used in the process of placing larvae onto background images.
"""

import cv2
import numpy as np

def create_larva_mask(larva_image):
    """
    Create a binary mask for a larva image.

    Args:
        larva_image (numpy.ndarray): The input larva image, either in RGBA or RGB format.

    Returns:
        numpy.ndarray: A binary mask where the larva is white (255) and the background is black (0).
    """
    if larva_image.shape[2] == 4:  # RGBA
        return larva_image[:, :, 3]
    else:  # RGB
        # Create a binary mask based on non-black pixels
        return np.any(larva_image != [0, 0, 0], axis=2).astype(np.uint8) * 255

def apply_mask(background, larva, mask, position):
    """
    Apply a masked larva image onto a background image at a specified position.

    Args:
        background (numpy.ndarray): The background image to apply the larva to.
        larva (numpy.ndarray): The larva image to be applied.
        mask (numpy.ndarray): The binary mask of the larva.
        position (tuple): The (y, x) position to place the top-left corner of the larva on the background.

    Returns:
        numpy.ndarray: The background image with the larva applied.
    """
    y_pos, x_pos = position
    larva_height, larva_width = larva.shape[:2]
    mask_height, mask_width = mask.shape[:2]

    # Ensure mask and larva have the same dimensions
    if larva_height != mask_height or larva_width != mask_width:
        mask = cv2.resize(mask, (larva_width, larva_height))

    # Check if the larva fits within the background
    if y_pos + larva_height > background.shape[0] or x_pos + larva_width > background.shape[1]:
        larva_height = min(larva_height, background.shape[0] - y_pos)
        larva_width = min(larva_width, background.shape[1] - x_pos)
        larva = larva[:larva_height, :larva_width]
        mask = mask[:larva_height, :larva_width]

    # Apply the mask
    for c in range(3):  # RGB channels
        background[y_pos:y_pos+larva_height, x_pos:x_pos+larva_width, c] = \
            background[y_pos:y_pos+larva_height, x_pos:x_pos+larva_width, c] * (1 - mask/255.0) + \
            larva[:, :, c] * (mask/255.0)

    return background
