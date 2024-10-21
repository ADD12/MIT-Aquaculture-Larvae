import numpy as np
import cv2 
import os
from Circle_Detection import circle_detection,get_circle_avg
from shapely.geometry import Point,Polygon
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from HSV_RGB_Classifier import create_classifier
from All_Robo_Data import create_dataset
# Author: Santiago Borrego
# File goes to process to test the performance of baseline model which includes
# circle detection and then HSV/RGB classification 
# Prints multiple metrics and displays images with results included

def calculate_iou(circle,polygon):
    """
    Calculates the intersection over the union
    of a circle and a polygon shape 
    """
    intersection = circle.intersection(polygon).area
    union = circle.area + polygon.area - intersection
    return intersection / union if union != 0 else 0

def get_matchings(circles,annotations):
    """
    Args: List of circles in original image and dictionary of polygons 
    in the image from Roboflow
    Returns dictionary matching all circles with a corresponding polygon
    """
    # Create filled in circle objects
    all_circles = [[Point(x, y).buffer(r) for x, y, r in image] for image in circles]
    # Create dictionary matching circles to labeled polygons
    matchings = {i:{} for i in range(len(circles))}
    for i in range(len(all_circles)):
        for outline,shape in zip(circles[i],all_circles[i]):
            largest_iou = 0
            match_poly = None 
            for image_id,label, polygon in annotations: # Iterate through all polygons
                if image_id == i:
                    iou = calculate_iou(shape, polygon) # Calculate IOU
                    if iou >= 0.4 and iou > largest_iou: # If IOU is above threshold and is the best
                        match_poly = (image_id, label, polygon)
                        largest_iou = iou
            matchings[i][tuple(outline)] = match_poly # Match circles to the polygon that fits best
    return matchings

def analyze_matchings(matchings,image_paths,annotations):
    """
    Args: Matching dictionary which maps each circle to either
    a corresponding polygon or to None, original image which is being analyzed
    Returns number of true positives, false positives, false negatives,
    true labels, predicted labels labels, and edited image
    """
    true_positives, false_positives, false_negatives,healthy,dead,dead_matchings,healthy_matchings = 0, 0, 0, 0, 0, 0,0
    true_labels, predicted_labels, images = [], [], []
    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        grey_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        matched_polygons = set()
        for circle, data in matchings[i].items(): # Iterate through matchings
            # Get HSV Features
            circle_hsv = get_circle_avg(hsv_image,circle)
            # Get RBG Featues
            #circle_rgb = get_circle_avg(image,circle)
            
            # Use trained classifier to predict label 
            circle_label = classifier.predict(circle_hsv)
            #circle_label = classifier.predict(np.hstack((circle_hsv,circle_rgb)))
            if circle_label[0] == 0:
                dead += 1
            elif circle_label[0] == 1:
                healthy += 1
            x, y, r = map(int,circle)
            if data is None: # If no matching is found
                # Track performance metrics
                true_labels.append(0) 
                predicted_labels.append(1)
                false_positives += 1
                # Draw red circle for false positives
                cv2.circle(image, (x, y), r, (0, 0, 255), 2)
                
            else: # Matching was found
                # Track performance metrics
                true_labels.append(1)  
                predicted_labels.append(1)
                true_positives += 1
                image_id,label,polygon = data
                if circle_label[0]==label:
                    if circle_label[0] == 0:
                        dead_matchings +=1
                    else:
                        healthy_matchings +=1 
                # Draw polygon
                matched_polygons.add(polygon) # Add to matched polygon set
                points = [(int(x), int(y)) for x, y in polygon.exterior.coords]
                if circle_label[0] == 0:
                    cv2.polylines(image, [np.array(points)], True, (0, 0, 0), 2) # Outline Black if dead
                else:
                    cv2.polylines(image, [np.array(points)], True, (0, 255, 0), 2) # Outline Green if healthy
                
        for image_id,label, polygon in annotations:
            if image_id == i:
                if polygon not in matched_polygons: # Finds polygons that weren't matched
                    # Track performance metrics
                    true_labels.append(1)  
                    predicted_labels.append(0)
                    false_negatives += 1
                    # Draw False negatives in blue
                    points = [(int(x), int(y)) for x, y in polygon.exterior.coords]
                    cv2.polylines(image, [np.array(points)], True, (255, 0, 0), 2)
        images.append(image)
    return true_positives,false_positives, false_negatives,healthy, dead, dead_matchings,healthy_matchings,true_labels,predicted_labels, images

def main():
    # Load training dataset and extract circles 
    train_dataset = create_dataset('train')
    base_path = os.path.abspath(os.path.join('..', 'Aquaculture-Larvae-2', 'train'))
    image_paths = [os.path.join(base_path, image['file_name']) for image in train_dataset['images']]
    circles = circle_detection(image_paths)
    annotations = train_dataset['annotations']
    matchings = get_matchings(circles, annotations)
    # Retrieve all the measurements from analysis
    true_positives,false_positives, false_negatives,healthy, dead, dead_matchings,healthy_matchings,true_labels, predicted_labels, images = analyze_matchings(matchings, image_paths, annotations)

    # Display images with outlines one by one
    for i, image in enumerate(images):
        window_name = f'Image {i+1}'

        cv2.imshow(window_name, image)

        cv2.waitKey(0)

        cv2.destroyWindow(window_name)

    # Print performance metrics
    print(f"True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}, Healthy: {healthy}, Dead: {dead}, Dead Matchings: {dead_matchings}, Healthy Matchings: {healthy_matchings}")

    # Calculate and print precision, recall, F1 score, and accuracy
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    classifier_accuracy= (dead_matchings+healthy_matchings)/true_positives
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Accuracy: {accuracy}, Classifier Accuracy: {classifier_accuracy}" )

if __name__ == '__main__':
    # Create classifier using function imported from other file
    classifier = create_classifier()
    main()
    
