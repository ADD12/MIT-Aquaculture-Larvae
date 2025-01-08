import cv2
import os
from ultralytics import YOLO

# Define paths (Adjust these paths to match your environment)
dataset_yaml_path = '/Path/to/data.yaml'  # Path to your data.yaml file
pretrained_model = "yolov8n.pt"  # You can choose other model sizes like yolov8s.pt, yolov8m.pt, etc.
train_results_folder = '/Folder for training/training_results'  # Folder where training results will be saved
dataset_images_path = '/Path/to/images'  # Path to images folder for running inference after training
prediction_output_folder = '/Path/to/prediction_output_folder'  # Folder where predictions will be saved

# Load the pre-trained YOLOv8 model
model = YOLO(pretrained_model)

# Step 1: Fine-tune the YOLOv8 model on your custom dataset with verbose progress logging
def train_model(epochs=50, imgsz=640, batch_size=16):
    print("Starting model training...")

    # Train the model with verbose logging (displays progress)
    model.train(
        data=dataset_yaml_path,  # Path to the dataset's data.yaml file from Roboflow
        epochs=epochs,           # Number of training epochs
        imgsz=imgsz,             # Image size for training (resized images)
        batch=batch_size,        # Batch size for training
        device='cpu',            # Use 'cpu' for CPU training (adjust as needed)
        verbose=True,             # Enable verbose mode to show detailed progress
        name="custom_v1",          # Experiment name
        project="/Project location/custom_yolo_models",  # Set the project path explicitly
    )

    print("Model training completed.")

# Step 2: Evaluate the trained model
def evaluate_model():
    print("Evaluating the model...")
    model.val()
    print("Model evaluation completed.")

# Step 3: Run inference on the dataset and save predictions with only bounding boxes (no labels)
def run_inference_on_dataset():
    print(f"Running inference on the dataset images from {dataset_images_path}...")
    
    # Run the model on the dataset images and get the results
    results = model.predict(
        source=dataset_images_path,  # Folder where your dataset images are located
        conf=0.1  # Confidence threshold
    )
    
    # Loop through each result
    for i, result in enumerate(results):
        # Get the original image
        img = result.orig_img

        # Get the bounding box coordinates from the result
        boxes = result.boxes.xyxy  # Bounding box coordinates

        # Loop through each box and draw it (no labels)
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green bounding box

        # Save the image with bounding boxes (no labels)
        output_image_path = os.path.join(prediction_output_folder, f"image_{i}.jpg")
        cv2.imwrite(output_image_path, img)

    print(f"Predictions saved in {prediction_output_folder}")

# Step 4: Combine the process (train, evaluate, and run inference)
if __name__ == "__main__":
    # Train the model with progress tracking
    train_model(epochs=50, imgsz=640, batch_size=16)
    
    # Evaluate the trained model
    evaluate_model()
    
    # Run inference on the dataset and save predictions
    run_inference_on_dataset()
