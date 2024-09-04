# Import necessary libraries
import cv2 as cv
import numpy as np
import joblib

'''Script Name: Instance Segmentation --> Classification full pipeline
Author: Unyi Usua
Date: 04/04/2024

Description:
This script take the masks created by the instance segmentation model and the trained classifier.
It is a pipeline to fully count and classify the the detected larvae larvae. 
'''

def make_detections(mask, img, model):
    # Ensure mask is single-channel and 8-bit
    real_mask = mask.astype(np.uint8)
    mask = mask.astype(np.uint8)

    if len(mask.shape) == 3:  # If mask has multiple channels, convert to single channel
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    # Find the min enclosing circle
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2:]
    if len(contours) == 0:
        print("No contours found")
        return

    # Calculate the distance transform and find the max value
    distance = cv.distanceTransform(mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    _, max_val, _, centre = cv.minMaxLoc(distance)
    mask1 = np.zeros_like(img[:, :, 0])
    mask = cv.circle(mask1, centre, int(max_val), (255, 255, 255), -1)

    # Ensure mask is single-channel and 8-bit
    mask = mask.astype(np.uint8)

    # Edit the ROI
    roi = cv.bitwise_and(img, img, mask=mask)
    roi2 = cv.bitwise_and(img, img, mask=real_mask)

    # Extract BGR and HSV mean values within the mask
    mean_bgr = cv.mean(roi, mask=mask)[:3]
    hsv_image = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mean_hsv = cv.mean(hsv_image, mask=mask)[:3]
    
    # Combine BGR and HSV mean values into features for prediction
    features = [[mean_bgr[0], mean_bgr[1], mean_bgr[2], mean_hsv[0], mean_hsv[1], mean_hsv[2]]]
    prediction = model.predict(features)
    
    # Determine the contour for the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv.findContours(real_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
   
    # Draw the contour and classification result on the image
    for cnt, cnt2 in zip(contours, contours2):
        x, y, w, h = cv.boundingRect(cnt)
        if prediction[0] == 1:
            img = cv.drawContours(img, [cnt2], -1, (0, 255, 255), 2)
            img = cv.drawContours(img, [cnt], -1, (0, 0, 255), 1)
        else:
            img = cv.drawContours(img, [cnt2], -1, (0, 255, 255), 2)
            img = cv.drawContours(img, [cnt], -1, (0, 255, 0), 1)
                   
    return img

# Load the masks produced by instance segmentation
masks = np.load('C:\\Users\\Unyim\\Downloads\\testttt\\GroundTruthMasks2.npy')

# Load the input image
image = cv.imread('C:\\Users\\Unyim\\Downloads\\testttt\\GT2zi.jpg')
num_instances = masks.shape[2]

# Initialize the custom classifier
modela = joblib.load('C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\\Code\\svm_more_data_model.sav')

# Copy the image to display results
masked_image = image.copy()

# Iterate through masks and apply detection
for i in range(num_instances):
    individual_mask = masks[:, :, i]
    masked_image = make_detections(individual_mask, masked_image, modela)
    print("On Mask: ", i)

# Display final result
cv.imshow(f"Masked Image", masked_image)
cv.waitKey(0)  # Wait for a key press to proceed to the next mask
cv.destroyAllWindows()  # Close all OpenCV windows
