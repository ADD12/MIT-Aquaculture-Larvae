import sys
import json
import torch
import cv2
import base64
import numpy as np  # Added import
from PIL import Image
from ultralytics import YOLO

# Load your YOLO model (replace with the actual model path)
MODEL_PATH = 'backend/custom_v14 copy/weights/best.pt'
model = YOLO(MODEL_PATH)

def crop_to_square(image):
    width, height = image.size
    print(f'Processing image with dimensions: {width}x{height}')  # Log dimensions

    if width == 1696 and height == 2544:
        top = 424
        bottom = height - 424
        cropped_image = image.crop((0, top, width, bottom))
    elif width == 2544 and height == 1696:
        left = 424
        right = width - 424
        cropped_image = image.crop((left, 0, right, height))
    elif width == 1696 and height == 1696:
        cropped_image = image
    else:
        print(f"Skipping image with unexpected dimensions: {width}x{height}")
        return None

    return cropped_image

def process_image(image_path):
    # Open and process the image (crop to square and convert to grayscale)
    with Image.open(image_path) as img:
        square_image = crop_to_square(img)
        if square_image is None:
            print("Image has unexpected dimensions, skipping processing.")
            return

        bw_image = square_image.convert("L")
        bw_image_cv = cv2.cvtColor(np.array(bw_image), cv2.COLOR_GRAY2BGR)

    # Run YOLO model on processed image
    results = model.predict(source=bw_image_cv, conf=0.5)
    boxes = results[0].boxes.xyxy if results and results[0].boxes else []

    object_count = len(boxes)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(bw_image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    _, buffer = cv2.imencode('.jpg', bw_image_cv)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Create the result dictionary
    result = {
        'imageUrl': image_path,
        'objectCount': object_count,
        'annotatedImageBase64': encoded_image
    }

    # Print only the JSON result to stdout
    print(json.dumps(result))

if __name__ == '__main__':
    image_path = sys.argv[1]

    # Process the image
    process_image(image_path)
