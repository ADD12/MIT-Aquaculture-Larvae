import cv2 as cv
import os
import glob
import numpy as np
import json
import skimage
import tifffile
import matplotlib.pyplot as plt
import shutil
from roboflow import Roboflow


'''Script Name: COCO JSON Annotation to Binary Mask Converter
Author: Unyi Usua
Date: 04/04/2024

Description:
This script processes Roboflow annotations, generating binary masks 
from them. It downloads data from Roboflow, loads annotations from COCO JSON
files, matches annotations with their respective images, creates masks,
and saves them as TIFF files.
'''


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

def create_mask(image_info, annotations, output_folder):
    # Create an empty mask as a numpy array
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint16)

    # Counter for the object number
    class1 = 1
    class2 = 2

    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            # Extract segmentation polygon
            for seg in ann['segmentation']:
                if seg[0] != seg[-2] or seg[1] != seg[-1]:
                  seg.append(seg[0])
                  seg.append(seg[1])

                # Convert polygons to a binary mask and add it to the main mask
                rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)

                if ann['category_id'] == class2:
                    mask_np[rr, cc] = class2
                else:
                    mask_np[rr, cc] = class1        


    # Visualize the mask
    plt.imshow(mask_np, cmap='gray')  # You can change the colormap as needed
    plt.colorbar()
    plt.title('Mask Visualization')
    plt.show()

    # Save the numpy array as a TIFF using tifffile library
    mask_path = os.path.join(output_folder, image_info['file_name'].replace('.jpg', '_mask.tif'))
    #mask_np_normalized = (mask_np * 255).astype(np.uint8)
    mask_np_normalized = mask_np.astype(np.float32) / np.max(mask_np)
    tifffile.imwrite(mask_path, mask_np_normalized)
    #tifffile.imwrite(mask_path, mask_np)

    print(f"Saved mask for {image_info['file_name']} to {mask_path}")


def images_and_features(current,images,labels,output):
    cnt = 0
    images = labels['images']
    annotations = labels['annotations']
    for img in images:

        #print('AT IMAGE------------', img)
        #im = cv.imread(str(images[id]))
        #im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

        create_mask(img,annotations,output)




def main():

    api_key = "GTZOIW8tEAHwMJFcuMCM"
    workspace_name = "river-herring"
    project_name = "aquaculture-larvae"
    version = 2
    load_roboflow_data(api_key, workspace_name, project_name, version) # only needs to be run once


    annotations_path = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\\Code\\Load_Data\\Aquaculture-Larvae-2\\train\\_annotations.coco.json'
    annotations_data = load_annotations(annotations_path)
    images_path = 'Aquaculture-Larvae-2\\train'
    image_paths = load_image_paths(images_path,annotations_data)

    mask_output_folder = 'binary_output' # Modify this as needed. Using val2 so my data is not overwritten
    image_output_folder = '/content/drive/MyDrive/Code Projects/Data/train_im_output'  #
    print(image_paths)
    a1 = images_and_features('its a go',image_paths,annotations_data,mask_output_folder)