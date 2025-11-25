import tifffile
import sklearn.feature_extraction
import numpy as np
import glob
import cv2
import os
import dataset_utils


def determine_patch_validity(patch):
    if np.sum(patch <= 10) > 0.1 * np.sum(patch >= 0):
        return False
    return True

RESOLUTION = 256

def generate_dataset_patches(image_folder, mask_folder, patch_image_folder, patch_mask_folder):
    # combine images with 12 bands and 13th channel as the mask
    # patch the image
    # store the 12 bands and 13th band separately
    labelled_images = glob.glob(f'{mask_folder}/*.png')
    print(len(labelled_images))
    for img in labelled_images:
        split_image_with_mask(img, image_folder, mask_folder, patch_image_folder, patch_mask_folder)
    
def extract_patches_with_mask(filename, image_folder, mask_folder):
    mask_filename = f'{mask_folder}/{os.path.basename(filename)[:-4]}.png'
    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
    mask = np.expand_dims(mask, axis=-1)
    if not os.path.exists(f'{image_folder}/{os.path.basename(filename)[:-4]}.tif'): return np.array([])
    img = dataset_utils.load_tif_image(f'{image_folder}/{os.path.basename(filename)[:-4]}.tif', [], all_bands = True)
    combined_image = np.concatenate((img, mask), axis=-1)

    if 'test' in image_folder:
        patches = []
        for y in range(0, combined_image.shape[0], RESOLUTION):
            for x in range(0, combined_image.shape[1], RESOLUTION):
                # Define the bounding box for the piece
                if x  + RESOLUTION > combined_image.shape[1] or y + RESOLUTION > combined_image.shape[0]:
                    continue
                piece = combined_image[y:y + RESOLUTION, x:x + RESOLUTION]
                patches.append(piece)
    else:
        patches = sklearn.feature_extraction.image.extract_patches_2d(combined_image, (RESOLUTION,RESOLUTION), max_patches=2000)
    return np.array(patches)

def split_image_with_mask(filename, image_folder, mask_folder, patch_image_folder, patch_mask_folder):
    patches = extract_patches_with_mask(filename, image_folder, mask_folder,)
    patch_num = 0
    for i in range(patches.shape[0]):
        patch = patches[i][:,:,:-1]
        mask = patches[i][:,:,-1]
        if determine_patch_validity(patch):
            patch_name = f'{patch_image_folder}/PATCH_{patch_num}_{os.path.basename(filename)[:-4]}.tif'
            mask_name = f'{patch_mask_folder}/PATCH_{patch_num}_{os.path.basename(filename)[:-4]}.png'
            tifffile.imwrite(patch_name, patch)
            cv2.imwrite(mask_name, mask * 255)
            patch_num += 1

def union_masks(mask_folder):
    masks = glob.glob(f'{mask_folder}/*.png')
    unique_names = set(['_'.join(m.split('_')[-12:-2]) for m in masks])
    print(unique_names)
    for n in unique_names:
        images = []
        for m in masks:
            if '_'.join(m.split('_')[-12:-2]) == n:
                images.append(cv2.imread(m))
        im = cv2.bitwise_or(images[0], images[1])
        im = cv2.bitwise_or(im, images[2])
        cv2.imwrite(f'{mask_folder}/{n}_mask.png',im )
    
def isolate_labelled_images(orig_image_folder, dest_image_folder, mask_folder):
    masks = glob.glob(f'{mask_folder}/*.png')
    images = glob.glob(f'{orig_image_folder}/*.tif')
    print(len(images))
    unique_names = set(['_'.join(m.split('/')[-1].split('_')[-11:-1]) for m in masks])
    print(unique_names)
    for n in unique_names:
        for m in images:
            print('_'.join(m.split('/')[-1].split('_')[-11:]))
            if '_'.join(m.split('/')[-1].split('_')[-10:])[:-4] == n:
                new_image_name = f'{dest_image_folder}/{n}.tif'
                image =dataset_utils.load_tif_image(m,bands = [], all_bands=True)
                tifffile.imwrite(new_image_name, image)



