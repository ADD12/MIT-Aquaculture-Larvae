import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from All_Robo_Data import create_dataset
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier



def calculate_polygon_avg(image,polygon):
    """
    Calculates the average rgb values for a polygon within image
    """
    # Create an empty mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Get exterior coordinates of the polygon
    exterior = np.array(polygon.exterior.coords).astype(np.int32)
    
    # Fill the polygon in the mask
    cv2.fillPoly(mask, [exterior], 255)
    
    if len(image.shape) == 3:
        # Image is in color (RGB)
        mean_values = cv2.mean(image, mask=mask)[:3]  # Exclude the alpha channel if present
    else:
        # Image is grayscale
        mean_values = cv2.mean(image, mask=mask)  # There's only one channel

    return mean_values


def prep_data(image_paths,annotations):
    """
    Takes in an image and the polygon_data dictionary
    Returns array of features and labels
    """
    features = []  # To hold the HSV or RGB values
    labels = []  # To hold the corresponding labels
    for i,path in enumerate(image_paths):
        image = cv2.imread(path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for annotation in annotations:
            image_id, label, polygon = annotation[0], annotation[1],annotation[2]
            if image_id == i:
                mean_hsv = calculate_polygon_avg(hsv_image, polygon)
                #mean_rgb = calculate_polygon_avg(image, polygon)
                features.append(mean_hsv)
                labels.append(label) # 0 is Dead and 1 is Healthy
    return np.array(features), np.array(labels)


def train_classifier(train_features, train_labels,test_features,test_labels):
    """
    Given training and testing features and labels 
    Returns a classifier training on the data and prints confusion matrix and accuracy
    """
    classes = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
    # param_grid = {'C': [0.1, 1, 10],
    #         'gamma': [0.0001, 0.001],
    #         'kernel': ['rbf', 'poly']
    #         }
    #svc = svm.SVC(C=1, gamma='scale', kernel='rbf', probability=True, class_weight=class_weights_dict, verbose=True)
    #svc = svm.SVC(probability=True,class_weight = class_weights_dict,verbose=True)
    #classifier_svm = GridSearchCV(svc,param_grid,cv=3,verbose=True)

    
    rf_classifier = RandomForestClassifier(n_estimators=100, class_weight=class_weights_dict, random_state=42)
    rf_classifier.fit(train_features, train_labels)
    # # Make predictions on the test set

    y_predictions = rf_classifier.predict(test_features)

    # # Evaluate the model
    cm = confusion_matrix(test_labels, y_predictions)
    accuracy = accuracy_score(test_labels, y_predictions)
    print(cm)
    print(f"Model Accuracy: {accuracy}")
    return rf_classifier


def cross_validate(classifiers,features,labels,n_folds = 5):
    """
    Function to cross validate different classifiers on the same data and labels
    Prints results by classifier
    """
    cv_results = {}

    for name, clf in classifiers.items():
        # Perform k-fold cross-validation
        scores = cross_val_score(clf, features, labels, cv=n_folds)
        
        # Store the mean and standard deviation of the scores
        cv_results[name] = {"mean_score": np.mean(scores), "std_score": np.std(scores)}

        # Print the results for each classifier
        print(f"{name}: Mean accuracy = {cv_results[name]['mean_score']:.4f} with std = {cv_results[name]['std_score']:.4f}")

def create_classifier():
    """
    Main function which loads the training and validation datasets
    Used to then train and return the classifier
    """
    train_dataset = create_dataset('train')
    train_base_path = os.path.abspath(os.path.join('..', 'Aquaculture-Larvae-2', 'train'))
    train_image_paths = [os.path.join(train_base_path, image['file_name']) for image in train_dataset['images']]
    train_annotations = train_dataset['annotations']
    train_features,train_labels = prep_data(train_image_paths,train_annotations)

    test_dataset = create_dataset('valid')
    test_base_path = os.path.abspath(os.path.join('..', 'Aquaculture-Larvae-2', 'valid'))
    test_image_paths = [os.path.join(test_base_path, image['file_name']) for image in test_dataset['images']]
    test_annotations = test_dataset['annotations']
    test_features,test_labels = prep_data(test_image_paths,test_annotations)

    classifier = train_classifier(train_features,train_labels,train_features,train_labels)
    return classifier
    
if __name__ == '__main__':
    create_classifier()
