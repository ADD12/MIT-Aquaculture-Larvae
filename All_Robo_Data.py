# Author: Santiago Borrego
# Code to connect with Roboflow API and retrieve JSON annotations
from roboflow import Roboflow
import json
from shapely.geometry import Polygon

# Loads data directly from Roboflow using API Key and project specifications
def load_roboflow_data(api_key, workspace_name, project_name, version):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)
    dataset = project.version(version).download("coco")
    return dataset

def open_dataset(annotations_path):
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data
    
# Cleans raw annotations and casts coordinates of annotations as a Polygon type
def clean_dataset(raw_dataset):
    clean = {'info': raw_dataset['info'], 'images':raw_dataset['images']}
    annotations = []
    for annotation in raw_dataset['annotations']:
        segmentation = annotation['segmentation'][0]
        coordinates = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
        annotations.append((int(annotation['image_id']),int(annotation['category_id'])-1, Polygon(coordinates)))
    clean['annotations'] = annotations
    return clean

def create_dataset(name):
    api_key = ""
    workspace_name = "river-herring"
    project_name = "aquaculture-larvae"
    version = 2
    #dataset = load_roboflow_data(api_key,workspace_name,project_name,version)
    data_path = f"../Aquaculture-Larvae-2/{name}/_annotations.coco.json"
    raw_train = open_dataset(data_path)
    clean_train = clean_dataset(raw_train)
    return clean_train
    # test_annotations = open_dataset(test_annotations_path)
    # valid_annotations = open_dataset(valid_annotations_path)
    


if __name__ =="__main__":
    create_dataset('train')
