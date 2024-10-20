
import cv2
import numpy as np
import os
# Author: Santiago Borrego
# Contains functions for circle-related tasks used in other files

def circle_detection(image_paths):
    """
    Given a list of images detects all circles present in the image
    Returns list of circles in images
    """
    
    image_circles = []
    for path in image_paths:
        img = cv2.imread(path)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Denoising Technique 
        denoised = cv2.fastNlMeansDenoising(gray,None,10,7,21)
        adaptive_thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11,2)

        # Apply Hough transform for circle detection
        circles = cv2.HoughCircles(adaptive_thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=25,
                                param1=48, param2=26, minRadius=6, maxRadius=35)
        image_circles.append(circles[0])
    return image_circles

def draw_circles(img,circles):
    """
    Args: Image and list of circles identified in image
    Draws the circles in the image and displays the image with
    circle's edges colored in green and another with background removed
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create a mask with the same dimensions as the original image, initially all zeros (black)
    mask = np.zeros_like(gray)
    # Fill in circles on the mask with white color
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(mask, (i[0], i[1]), i[2], (255), -1)  # thickness=-1 fills the circle
    # Create a white canvas with the same dimensions as the original image
    white_canvas = np.full_like(img, 255)
    # Use the mask to combine the original image with the white canvas
    # Where the mask is black, use the white canvas; where the mask is white, use the original image
    background_removed = np.where(mask[:, :, np.newaxis] == 255, img, white_canvas)

    # Draw detected circles outline on the original image
    if circles is not None:
        circles = np.round(circles[0, :]).astype('int')
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    # Display the result
    cv2.imshow('Circled Image', img)  # Display image 1 in the 'Image 1' window
    key = cv2.waitKey(0)
    if key == ord('s'):  # If 's' is pressed
        cv2.imwrite('circled_image6.jpg', img)  # Save the image
        print("Image saved")
    cv2.imshow('Background Removed', background_removed)  # Display image 2 in the 'Image 2' window
    key = cv2.waitKey(0)
    if key == ord('s'):  # If 's' is pressed
        cv2.imwrite('background_removed.jpg', background_removed)  # Save the image
        print("Image saved")
    cv2.destroyAllWindows()


def get_circle_avg(image,circle):
    """
    Given an image and a circle annotation 
    Returns the 3 average color or 1 average grayscale value within that circle
    """
    x, y, r = map(int, circle)
    # Create a mask for the circle
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (x, y), r, 255, -1)
    
    if len(image.shape) == 3:
        # Image is in color (e.g., HSV, RGB)
        mean_values = cv2.mean(image, mask=mask)[:3]  # Exclude the alpha channel if present
    else:
        # Image is grayscale
        mean_values = cv2.mean(image, mask=mask)  # There's only one channel, so we get the first element

    # Convert mean_values to a numpy array and reshape it
    mean_values_array = np.array(mean_values).reshape(1, -1)  # Convert the tuple to a numpy array and reshape
    
    return mean_values_array
