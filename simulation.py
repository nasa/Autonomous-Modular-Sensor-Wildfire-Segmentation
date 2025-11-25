import combined_algorithm
import classifier_model
import segmentation_model
import dataset_utils
import math
import numpy as np
import cv2

def run_inference(full_img, input_img, classifier_model, segmenter_model, frame_rate, display_mode='visual_spectrum'):
    H,W = input_img.shape[:2]
    new_H, new_W = int(math.ceil(H/256)*256), int(math.ceil(W/256)*256)
    new_full_img = np.pad(full_img, ((0,new_H - H), (0,new_W - W), (0,0)))
    new_input_img = np.pad(input_img, ((0,new_H - H), (0,new_W - W), (0,0)))

    display_img = np.zeros((new_H, new_W, 3)).astype(np.uint8)
    display_mask = np.zeros((new_H, new_W, 1)).astype(np.uint8)


    for row in range(new_H//256):
        for col in range(new_W//256):
            patch = new_input_img[row*256:(row+1)*256, col*256:(col+1)*256]/255
            patch = patch[None,:,:,:]
            text, result = combined_algorithm.complete_algorithm(patch, classifier_model, segmenter_model)
            if display_mode=='visual_spectrum':
                display_img[row*256:(row+1)*256, col*256:(col+1)*256, :] = new_full_img[row*256:(row+1)*256, col*256:(col+1)*256,[1,2,4]]
            if display_mode=='color_infrared':
                display_img[row*256:(row+1)*256, col*256:(col+1)*256, :] = new_full_img[row*256:(row+1)*256, col*256:(col+1)*256,[2,4,6]]
            if display_mode=='fire_emphasizing':
                display_img[row*256:(row+1)*256, col*256:(col+1)*256, :] = new_full_img[row*256:(row+1)*256, col*256:(col+1)*256,[2,6,9]]
            display_mask[row*256:(row+1)*256, col*256:(col+1)*256] = result[0]*255
            cv2.imshow('original', display_img)
            cv2.imshow('fire mask', display_mask)
            cv2.waitKey(int(1/frame_rate*1000))

    return 

def main():
    print('hi')
    classification_model_all_bands = classifier_model.build_model(3, 'simple', None)
    classification_model_all_bands.summary()
    classification_model_all_bands.load_weights('./data/model_weights/enc_16_classification_bands_11,9,2_trial1.keras')
    seg_model = segmentation_model.get_trimmed_unet_model(3,1)
    seg_model.summary()
    seg_model.load_weights('./data/model_weights/enc_16_segmentation_bands_11,9,2_trial1.keras')
    img = dataset_utils.load_tif_image('./data/ams_data/processed_images/labelled_images/complete_images/test/ENHANCED_CLEAN_TIF_AMSL1B_1180104_05_20110722_2000_2003_V01.tif', [], all_bands=True)
    frame_rate = 5
    
    # change the image bands depending on which model is used
    run_inference(img, img[:,:, [10,8,1]], classification_model_all_bands, seg_model, frame_rate, display_mode='fire_emphasizing')

if __name__ == '__main__':
    main()











