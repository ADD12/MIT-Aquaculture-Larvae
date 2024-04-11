'''create 'patches'''

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff
import glob
import os

def create_patches(im,name,save_path):

    patches_img = patchify(im, (128, 128,3), step=128)  #Step=256 for 256 patches means no overlap
    head_path = save_path

    for i in range(patches_img.shape[0]):

        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]

            save_path = os.path.join(head_path,  name + '_' + str(i)+str(j)+ ".tif")
            tiff.imwrite(save_path, single_patch_img)


def create_patches_mask(im,name,save_path):

    patches_img = patchify(im, (256, 256), step=236)  #Step=256 for 256 patches means no overlap
    head_path = save_path

    for i in range(patches_img.shape[0]):

        for j in range(patches_img.shape[1]):
            #print('j==', j)
            
            single_patch_img = patches_img[i,j,:,:]
            #visualize_mask(single_patch_img)
            save_path = os.path.join(head_path,  name + '_' + str(i)+str(j)+ ".tif")
            tiff.imwrite(save_path, single_patch_img)

def visualize_mask(mask_np):
    # Visualize the mask
    plt.imshow(mask_np, cmap='gray')  # You can change the colormap as needed
    plt.colorbar()
    plt.title('Mask Visualization')
    plt.show()

def main():
    mask_path = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\\Code\\binary_output'
    im_path = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\\Code\\im_output'

    save_mask = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\\Code\\bi_patches'
    save_img = 'C:\\Users\\Unyim\\Downloads\\CSDownloads\\IAP_UROP_DATA\Code\\im_patches'

    for mask_file_path,im_file_path in zip(glob.glob(os.path.join(mask_path, '*.tif')), glob.glob(os.path.join(im_path, '*.tif'))):


        color = tiff.imread(im_file_path)
        mask = tiff.imread(mask_file_path)
        name_mask = str(mask_file_path).split("\\")[-1].split('.')[0]
        name_im = str(im_file_path).split("\\")[-1].split('.')[0]

        create_patches(color,name_im,save_img)
        create_patches_mask(mask,name_mask,save_mask)

main()
    
