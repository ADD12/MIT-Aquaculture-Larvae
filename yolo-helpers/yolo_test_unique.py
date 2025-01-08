import cv2
import os
import glob
from ultralytics import YOLO
import random

# Load your trained YOLO model (replace with your trained model path)
model = YOLO('/Path/to/model/best.pt')  # Path to your trained YOLO model

def process_images(input_folder, conf_threshold=0.25):
    # Use glob to get all jpg/jpeg images
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.jpeg")) + glob.glob(os.path.join(input_folder, "*.JPG")) + glob.glob(os.path.join(input_folder, "*.JPEG"))

    # Check if any images are found
    if not image_paths:
        raise ValueError(f"No images found in the specified input folder: {input_folder}")

    # Shuffle the images to randomize the display
    random.shuffle(image_paths)

    # Loop through each image and apply YOLOv8 detection
    for image_path in image_paths:
        # Load image
        image = cv2.imread(image_path)

        # Run YOLOv8 on the image with a lower confidence threshold
        results = model.predict(source=image, conf=conf_threshold)  # Set confidence threshold

        # Extracting bounding boxes without labels
        boxes = results[0].boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
        
        # Draw bounding boxes on the image (no labels)
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green bounding box

        # Display the result image with bounding boxes (no labels)
        cv2.imshow('YOLO Detection', image)

        # Wait for 2 seconds (2000 milliseconds)
        cv2.waitKey(2000)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Example usage
input_folder = '/Path/to/input/folder'  # Adjust path as necessary

process_images(input_folder, conf_threshold=0.5)
