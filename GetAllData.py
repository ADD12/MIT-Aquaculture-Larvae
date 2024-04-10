from roboflow import Roboflow
from PIL import Image
import json
from pathlib import Path
import os
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 
import random as r
import pandas as pd

def load_roboflow_data(api_key, workspace_name, project_name, version):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)
    dataset = project.version(version).download("coco")
    return dataset

def load_annotations(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data

def load_image_paths(images_path,annotations):
    image_paths = {}
    im_ids = annotations['images']

    for file_path in glob.glob(os.path.join(images_path, '*.jpg')):
        name = str(file_path).split("\\")[-1]

        #match annotations with their respective images
        for im in im_ids:
            if im['file_name'] == name:
                image_paths[im['id']] = file_path
        
    return image_paths


def images_and_features(current,images,labels,feature=None):
    df_features = pd.DataFrame(columns=['B','G','R','H','S','V','Target'])
    cnt = 0

    annotations = labels['annotations']
    for id in range(len(images)):
        if not annotations: 
            print('not notations found')
            break

        print('AT IMAGE------------', id)
        im = cv.imread(str(images[id]))

        for node in annotations:
            if node['image_id'] == id:
                poly_mask = np.full((im.shape[0], im.shape[1]), 0, dtype=np.uint8)
                pts = node['segmentation'][0]
                pts = np.array(pts, np.int32)
                pts = pts.reshape((-1,1,2))
                
                cv.fillPoly(poly_mask, pts=[pts], color=(255, 0, 0))


                masked_img = cv.bitwise_and(im,im,mask = poly_mask)

                

                #box each image then resize each image to 128x128
                heights = pts[:,0][:,1]
                widths = pts[:,0][:,0]
                maxh,minh =  max(heights),min(heights)
                maxw,minw = max(widths),min(widths)
                masked_img = masked_img[minh:maxh,minw:maxw]
                try:
                    masked_img = cv.resize(masked_img,(128,128))
                except:
                    break

                #extract features
                masked_img =  get_center_square(masked_img)

                #extract features
                b,g,r,h,s,v = get_bgr(masked_img,poly_mask) +  get_hsv(masked_img,poly_mask)
                df_features.loc[cnt] = [b,g,r,h,s,v,node["category_id"]]
                cnt+=1
                print('at image, ',cnt)
                print('color',b,g,r,h,s,v )
                try:
                    masked_img = cv.resize(masked_img,(128,128))
                    #save image to a folder in current dir                                                                                                            
                    #save_image(current,masked_img,cnt)
                except:
                    print('FAILURE')

            
    return df_features

def get_bgr(masked_img,poly_mask):
    #mean_bgr = cv.mean(masked_img, mask=poly_mask)[:3]
    mean_bgr = cv.mean(masked_img)[:3]
    return mean_bgr

def get_hsv(masked_img,poly_mask):
    hsv_image = cv.cvtColor(masked_img, cv.COLOR_BGR2HSV)
    #mean_hsv = cv.mean(hsv_image, mask=poly_mask)[:3]
    mean_hsv = cv.mean(hsv_image)[:3]
    return mean_hsv

def get_gray(masked_img,poly_mask):
    gray = cv.cvtColor(masked_img, cv.COLOR_BGR2GRAY)
    mean_gray = cv.mean(gray, mask=poly_mask)[:1]
    return mean_gray,gray

def get_center_square(masked_img):
    masked_img = masked_img[32:32+64, 32:32+64]
    return masked_img

    
def save_image(current,photo,cnt):
    try:
        os.makedirs(os.path.join(current, 'bw_larvae_images'))
        path = os.path.join(current, 'bw_larvae_images')
    except FileExistsError:
        # directory already exists
        path = os.path.join(current, 'bw_larvae_images')
    cv.imwrite(os.path.join(path , 'larvae_{}.jpg'.format(str(cnt))), photo)

def main():
    # Roboflow data loading

    api_key = "GTZOIW8tEAHwMJFcuMCM"
    workspace_name = "river-herring"
    project_name = "aquaculture-larvae"
    version = 2
    load_roboflow_data(api_key, workspace_name, project_name, version) # only needs to be run once

    # training Annotations
    script_dir = os.path.dirname(__file__) #get current directory

    # training paths
    annotations_path = os.path.join(script_dir, 'Aquaculture-Larvae-2\\train\\_annotations.coco.json')
    annotations_data = load_annotations(annotations_path)
    images_path = os.path.join(script_dir, "Aquaculture-Larvae-2\\train")
    image_paths = load_image_paths(images_path,annotations_data)

    #val paths
    annotations_path_2 = os.path.join(script_dir, 'Aquaculture-Larvae-2\\valid\\_annotations.coco.json')
    annotations_data_2 = load_annotations(annotations_path_2)
    images_path_2 = os.path.join(script_dir, "Aquaculture-Larvae-2\\valid")
    image_paths_2 = load_image_paths(images_path_2,annotations_data_2)
    

    #test paths
    annotations_path_3 = os.path.join(script_dir, 'Aquaculture-Larvae-2\\test\\_annotations.coco.json')
    annotations_data_3 = load_annotations(annotations_path_3)
    images_path_3 = os.path.join(script_dir, "Aquaculture-Larvae-2\\test")
    image_paths_3 = load_image_paths(images_path_3,annotations_data_3)


    #combine data into one
    print('getting images----------')
    print('im paths, ', image_paths)
    a1 = images_and_features(script_dir,image_paths,annotations_data)
    a2 = images_and_features(script_dir,image_paths_2,annotations_data_2)
    a3 = images_and_features(script_dir,image_paths_3,annotations_data_3)

    #hsv df
    print('saving gray df')
    a = pd.concat([a1, a2,a3], ignore_index=True)
    a.to_csv('gray_data.csv')
    print(a.head(20))


if __name__ == "__main__":
    main()
