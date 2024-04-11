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

def create_mask(og_image_path,image_info, annotations, output_folder,im_output,cnt):
    # Create an empty mask as a numpy array
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint16)

    # Counter for the object number
    class1 = 1 #class for not healthy
    class2 = 2 #class for healthy
    id = image_info['id']

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


    # Visualize the mask while making it to make sure it is correct
    visualize_mask(mask_np)

    # Save the numpy array as a TIFF using tifffile library
    mask_name = "mask_{}.tif".format(cnt)
    mask_path = os.path.join(output_folder, mask_name)
    tifffile.imwrite(mask_path, mask_np)

    #colored image
    image_name = "image_{}.tif".format(cnt)
    original = image_info['file_name']
    path_im = os.path.join(og_image_path,original)
    get_image = cv.imread(path_im, cv.IMREAD_COLOR)
    im_path = os.path.join(im_output, image_name)
    tifffile.imwrite(im_path, get_image)

    print('shapes')
    print(mask_np.shape)
    print(get_image.shape)
    print('--------------')

    #print(f"Saved mask and image for {image_info['file_name']} to {mask_path}")


def visualize_mask(mask_np):
    # Visualize the mask
    plt.imshow(mask_np, cmap='gray')  # You can change the colormap as needed
    plt.colorbar()
    plt.title('Mask Visualization')
    plt.show()


def loop_images(og_image_path,labels,output_bi,output_im):
    cnt = 0
    images = labels['images']
    annotations = labels['annotations']

    for img in images:

        create_mask(og_image_path,img,annotations,output_bi,output_im,cnt)
        cnt+=1


def main():

    api_key = "GTZOIW8tEAHwMJFcuMCM"
    workspace_name = "river-herring"
    project_name = "aquaculture-larvae"
    version = 5
    print('loading from roboflow...')
    load_roboflow_data(api_key, workspace_name, project_name, version) # only needs to be run once

    #get annotation data
    annotations_path = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\\Code\\Load_Data\\Aquaculture-Larvae-5\\train\\_annotations.coco.json' #path to annotations
    annotations_data = load_annotations(annotations_path)

    #get image data
    images_path = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\\Code\\Load_Data\\Aquaculture-Larvae-5\\train'  #path to images
    image_paths = load_image_paths(images_path,annotations_data)

    #output folders
    mask_output_folder = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\\Code\\binary_output' # choose output folder for the masks
    image_output_folder = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\\Code\\im_output'  # choose output folder for the images

    print('looping through images')
    loop_images(images_path,annotations_data,mask_output_folder,image_output_folder)

main()
