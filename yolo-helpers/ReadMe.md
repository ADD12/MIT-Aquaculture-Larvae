# Project Overview

This repository includes several Python scripts for image processing, video-to-image conversion, and YOLO-based object detection and testing.

---

## Files and Descriptions

### 1. `image_processor.py`
- **Purpose**: Handles image processing tasks such as resizing, cropping, and formatting images for further analysis.
- **Usage**: 
  ```bash
  python image_processor.py --input <input_folder> --output <output_folder>
  ```
- **Features**:
  - Converts images to desired formats.
  - Crops or resizes images as required.

---

### 2. `train_test_yolo.py`
- **Purpose**: Provides functionality for training and testing a YOLO model.
- **Usage**:
  ```bash
  python train_test_yolo.py --train_data <path_to_training_data> --test_data <path_to_test_data>
  ```
- **Features**:
  - Supports training YOLO models on labeled datasets.
  - Evaluates model performance on test datasets.

---

### 3. `vid_to_image.py`
- **Purpose**: Converts video files into individual image frames for further processing.
- **Usage**:
  ```bash
  python vid_to_image.py --video <video_path> --output <output_folder>
  ```
- **Features**:
  - Extracts frames from videos at specified intervals.
  - Saves extracted frames in a designated folder.

---

### 4. `yolo_test_unique.py`
- **Purpose**: Tests a YOLO model on unique datasets or specific conditions.
- **Usage**:
  ```bash
  python yolo_test_unique.py --model <model_path> --data <data_path>
  ```
- **Features**:
  - Customizable test parameters for unique datasets.
  - Outputs performance metrics and visualizations.

---

### 5. `yolo_test.py`
- **Purpose**: General testing script for YOLO models.
- **Usage**:
  ```bash
  python yolo_test.py --model <model_path> --test_data <test_data_path>
  ```
- **Features**:
  - Tests YOLO models on standard datasets.
  - Provides evaluation metrics and optional image annotation.

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Set up and activate a virtual environment:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the desired script using the examples provided above.

---

## Notes
- Ensure you have the necessary dependencies and model weights for YOLO scripts.
- Modify the paths and parameters in each script to suit your specific use case.
