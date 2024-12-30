# Shellfish Counting and Classification System

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Future Work](#future-work)

---

## Introduction
This project addresses the challenge of counting and classifying larvae in images captured using a microscope and camera. The goal is to process these images to identify and categorize larvae as healthy or dead.

Currently, a YOLO-based model is providing the best results for detecting larvae. However, classification performance is limited due to an imbalanced dataset, with dead larvae being significantly underrepresented.

The **Spring-2024** directory includes the most recent work on this project. There is also a synthetic data generator that can generate synthetic data that is automatically labeled in COCO format. 

---

## Features
*To be filled in later.*

---

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/rvincent99/MIT-Aquaculture-Larvae.git
    ```

2. Install dependencies:
    ```bash
    pip install -r Spring-2024/requirements.txt
    ```

---

## Future Work
1. **Data Labeling and Balancing**:
    - Future UROP students should label additional data, with a focus on training the model on images containing dead larvae, which are currently underrepresented in the dataset.
    - Efforts should be made to balance the dataset to improve classification performance on the minority class.

2. **Detailed Record-Keeping**:
    - Keep detailed records of all iterations and changes to the model.
    - Ensure evaluation results are consistently recorded to build upon previous work.
    - Link to record of model iterations: https://docs.google.com/document/d/1nvsKUe01gV2PtSwyXfv_y1YS0jDedYZFQ-AhA69CAf0/edit?usp=sharing
