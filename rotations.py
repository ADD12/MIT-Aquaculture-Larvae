import tifffile as tiff
import albumentations as A
import cv2 as cv
import glob
import numpy as np
import os

images_to_generate = 1000


#organize the images and masks in order based on naming criteria
images = []
masks = []


#define the transformations that you want done
transform = A.Compose([
    A.Rotate(p=1),
    #A.RandomBrightnessContrast(p=0.3),
    A.RandomRotate90(p=1),
])

def save_transformation(img_name,mask_name,save_mask,save_img,aug,t_img,t_mask):
    save_im_path = os.path.join(save_img,  img_name + '_' + str(aug) +".tif")
    save_mask_path = os.path.join(save_mask,  mask_name + '_' + str(aug) + ".tif")
    tiff.imwrite(save_im_path, t_img)
    tiff.imwrite(save_mask_path, t_mask)


def main():
    mask_path = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\\Code\\binary_output'
    im_path = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\Code\\im_output'

    save_mask = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\\Code\\album_masks'
    save_img = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\Code\\album_images'

    for mask_file_path,im_file_path in zip(glob.glob(os.path.join(mask_path, '*.tif')), glob.glob(os.path.join(im_path, '*.tif'))):


        color = tiff.imread(im_file_path)
        mask = tiff.imread(mask_file_path)
        name_mask = str(mask_file_path).split("\\")[-1].split('.')[0]
        name_im = str(im_file_path).split("\\")[-1].split('.')[0]

        for i in range(15):
            aug = i
            print(color.shape)
            print(mask.shape)
            transformed = transform(image=color, mask=mask)
            transformed_image = transformed["image"]
            transformed_mask = transformed["mask"]
            save_transformation(name_im,name_mask,save_mask,save_img,aug,transformed_image,transformed_mask)


main()