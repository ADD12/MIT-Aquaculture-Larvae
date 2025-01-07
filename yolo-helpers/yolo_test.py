import cv2
import os
import glob
from ultralytics import YOLO
import time
import random

# Load the YOLOv8 model (You can choose yolov8n.pt, yolov8s.pt, etc. based on your requirements)
model = YOLO("yolov8n.pt")  # You can replace "yolov8n.pt" with a different model if you like

SPORTS_BALL_CLASS_ID = 37

def process_images(input_folder, conf_threshold=0.25):
    # Use glob to get all jpg/jpeg images
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.jpeg")) + glob.glob(os.path.join(input_folder, "*.JPG")) + glob.glob(os.path.join(input_folder, "*.JPEG"))

    # Check if any images are found
    if not image_paths:
        raise ValueError(f"No images found in the specified input folder: {input_folder}")

    random.shuffle(image_paths)
    # Loop through each image and apply YOLOv8 detection
    for image_path in image_paths:
        # Load image
        image = cv2.imread(image_path)

        # Run YOLOv8 on the image with a lower confidence threshold
        results = model.predict(source=image, conf=conf_threshold)  # Set confidence threshold

        # Get result image with bounding boxes and labels
        result_image = results[0].plot()  # .plot() returns a visualized image with bounding boxes

        # Display the result image
        cv2.imshow('YOLO Detection', result_image)

        # Wait for 2 seconds (2000 milliseconds)
        cv2.waitKey(2000)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Example usage
input_folder = "/Users/beckettdevoe/Desktop/Larvae Main/sorted_images/10x"  # Adjust path as necessary
process_images(input_folder, conf_threshold=0.01)
