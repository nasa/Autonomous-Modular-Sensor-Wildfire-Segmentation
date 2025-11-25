import requests
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import tifffile
import os

DATASET_FOLDER= '../data/tensorflow_datasets'
MIN_FIRE_PIXELS = 3
DEFAULT_BATCH_SIZE = 10
LANDSAT_MIN_FIRE_PIXELS = 10


K_to_C = 273.15
MIN_TEMP = 250
MAX_TEMP = 500

def display_ams_patch(ams_image, true_mask, pred_mask = None):
    # get RGB
    rgb = np.stack([ams_image[:,:,i] for i in (4,2,1)], axis=-1)

def load_tif_image(filename, bands, all_bands = False):
    try:
        image = tifffile.imread(filename)
    except:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if len(image.shape) == 2:
        image=np.expand_dims(image, -1)

    bands = np.array(bands)

    if image is None:
        raise Exception(f'{filename} not found')

    if image.shape[-1] == 16:
        bands[bands == 9] = 13
        bands[bands == 10] = 14
        bands[bands == 11] = 15
        bands[bands == 12] = 16

    if not all_bands: image = image[:,:,bands - 1]

    return image

def process_mask(mask):
    if tf.reduce_sum(tf.cast(mask>0, dtype=tf.uint8)) > 0:
        return 1
    return 0

def normalize(image):
    return (image - np.min(image,keepdims=True)) / (np.max(image, keepdims=True) - np.min(image,keepdims=True))

def get_dataset(bands, image_folder, mask_folder, dataset_name=None, dataset_source='AMS', dataset_type='classification', test=False):
    if dataset_name == None:
        filename = os.path.join(DATASET_FOLDER, f"{dataset_source}_{dataset_type}_bands_{str(bands)}_{'test' if test else 'train'}.ds")
    else: filename = os.path.join(DATASET_FOLDER, dataset_name)
    if not os.path.exists(filename) or True:
        with tf.device('/cpu:0') :
            num_images, dataset = generate_dataset(bands, image_folder, mask_folder,dataset_source, dataset_type)
            # dataset.save(filename)
            return num_images, dataset
    dataset = tf.data.Dataset.load(filename)
    num_images = dataset.cardinality().numpy()
    return num_images, dataset

def generate_dataset(bands, image_folder, mask_folder,dataset_source, dataset_type):
    if dataset_source == 'AMS' and dataset_type=='classification':
        return generate_AMS_classification_dataset(bands, image_folder, mask_folder)
    if dataset_source == 'AMS' and dataset_type=='segmentation':
        return generate_AMS_segmentation_dataset(bands, image_folder, mask_folder)
    if dataset_source == 'Landsat8' and dataset_type=='classification':
        return generate_Landsat8_classification_dataset(bands, image_folder, mask_folder)
    if dataset_source == 'Landsat8' and dataset_type=='segmentation':
        return generate_Landsat8_segmentation_dataset(bands, image_folder, mask_folder)
    if dataset_source == 'USFS' and dataset_type=='classification':
        return generate_USFS_classification_dataset(bands, image_folder, mask_folder)
    if dataset_source == 'Generative' and dataset_type=='classification':
        return generate_AMS_classification_dataset(bands, image_folder, mask_folder)
    if dataset_source == 'Generative' and dataset_type=='segmentation':
        return generate_AMS_segmentation_dataset(bands, image_folder, mask_folder)

def generate_AMS_classification_dataset(bands, image_folder, mask_folder):
    num_channels = len(bands)
    images = glob.glob(f'{image_folder}/*.tif')
    masks = glob.glob(f'{mask_folder}/*.png')
    images.sort()
    masks.sort()
    num_images = len(images)
    x_shape = (num_images, 256,256,num_channels)
    y_shape = (num_images, 1)

    x =  np.zeros(x_shape, dtype=np.float16)
    y =  np.zeros(y_shape, dtype=np.uint8)
    pos_avg_activation = []
    neg_avg_activation = []

    for i,image in enumerate(images):
        img = load_tif_image(image, bands)
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        x[i] = img
        if np.sum(mask > 0) > MIN_FIRE_PIXELS:
            y[i][0] = 1
            pos_avg_activation.append(np.max(img))
        else:
            neg_avg_activation.append(np.max(img))
    # print('pos:', len(pos_avg_activation), 'neg:', len(neg_avg_activation))
    # print(np.mean(pos_avg_activation), np.mean(neg_avg_activation))
    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x/255), tf.convert_to_tensor(y)))
    return num_images, dataset

def generate_AMS_segmentation_dataset(bands, image_folder, mask_folder):
    num_channels = len(bands)
    images = glob.glob(f'{image_folder}/*.tif')
    masks = glob.glob(f'{mask_folder}/*.png')
    images.sort()
    masks.sort()
    valid_images = 0
    for i,image in enumerate(images):
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        if np.sum(mask > 0) > MIN_FIRE_PIXELS:
            valid_images+=1
    num_images = valid_images
    x_shape = (num_images, 256,256,num_channels)
    y_shape = (num_images, 256,256, 1)

    x =  np.zeros(x_shape, dtype=np.float16)
    y =  np.zeros(y_shape, dtype=np.float16)

    valid_images = 0
    print(len(images))
    for i,image in enumerate(images):
        img = load_tif_image(image, bands).astype(np.float16)
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        if np.sum(mask > 0) > MIN_FIRE_PIXELS:
            x[valid_images] = img
            y[valid_images,:,:,0]=mask
            valid_images += 1

    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x/255), tf.convert_to_tensor(y/255)))
    return num_images, dataset

def process_Landsat8_image(image_file):
    GC_L8 = 'https://storage.googleapis.com/gcp-public-data-landsat/LC08/01/'
    MTL_EXTENSION = '_MTL.txt'

    def getMTLParameters(MTL):
        '''Parses the given metadata (MTL) text, and returns several independent
    parameters.'''

        Mref = []
        Aref = []
        Mrad = []
        Arad = []
        K1 = []
        K2 = []

        MTL = MTL.splitlines()

        for ln in MTL:

            if 'RADIANCE_MULT_BAND_' in ln:
                Mrad.append(float(ln.split(' = ')[1]))
            if 'RADIANCE_ADD_BAND_' in ln:
                Arad.append(float(ln.split(' = ')[1]))
            if 'REFLECTANCE_MULT_BAND_' in ln:
                Mref.append(float(ln.split(' = ')[1]))
            if 'REFLECTANCE_ADD_BAND_' in ln:
                Aref.append(float(ln.split(' = ')[1]))
            if 'K1_CONSTANT_BAND_' in ln:
                K1.append(float(ln.split(' = ')[1]))
            if 'K2_CONSTANT_BAND_' in ln:
                K2.append(float(ln.split(' = ')[1]))

            if 'SUN_ELEVATION' in ln:
                SE = float(ln.split(' = ')[1])

            if 'LANDSAT_SCENE_ID' in ln:
                L8ID = (ln.split(' = ')[1])
            if 'FILE_DATE' in ln:
                FDATE = str(ln.split(' = ')[1])
            if 'DATE_ACQUIRED' in ln:
                DATEAC = str(ln.split(' = ')[1])
            if 'SCENE_CENTER_TIME' in ln:
                SceneTIME = str(ln.split(' = ')[1])
            if 'CLOUD_COVER' in ln:
                CC = float(ln.split(' = ')[1])
            if 'MAP_PROJECTION' in ln:
                MP = str(ln.split(' = ')[1])
            if 'DATUM' in ln:
                DT = str(ln.split(' = ')[1])
            if 'ELLIPSOID' in ln:
                EL = str(ln.split(' = ')[1])
            if 'UTM_ZONE' in ln:
                ZONE = int(ln.split(' = ')[1])

        return Mrad, Arad, Mref, Aref, K1, K2, SE, L8ID, FDATE, DATEAC, SceneTIME, CC, MP, DT, EL, ZONE
    image_name = '_'.join(os.path.basename(image_file).split('_')[:-1])
    aws_path = GC_L8 + image_name[10:13] + '/' + image_name[13:16] + '/' + image_name + '/' + image_name + MTL_EXTENSION
    MTL = requests.get(aws_path).text
    Mrad, Arad, Mref, Aref, K1, K2, SE, L8ID, FDATE, DATEAC, SceneTIME, CC, MP, DT, EL, ZONE = getMTLParameters(MTL)
    Mrad, Arad, Mref, Aref, K1, K2 = np.array(Mrad), np.array(Arad), np.array(Mref), np.array(Aref), np.array(K1), np.array(K2)

    img = tifffile.imread(image_file)
    img = np.insert(img, 7, np.zeros_like(img[..., 0]), axis=-1)

    radiance = img * Mrad[None, None, :] + Arad[None, None, :]
    brightness_temp = K2[None, None, :] / np.log(K1[None, None, :] / radiance[..., -2:] + 1)
    brightness_temp = (np.clip(brightness_temp, MIN_TEMP, MAX_TEMP) - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)
    reflectance = img[..., :-2] * Mref[None, None, :] + Aref[None, None, :]
    scaled_radiance = reflectance / np.pi * 2
    scaled_radiance = np.clip(scaled_radiance, 0, 1)
    normalized_output_image = np.concatenate((scaled_radiance, brightness_temp), axis=-1)
    print(normalized_output_image.shape)

    return normalized_output_image


def generate_Landsat8_segmentation_dataset(bands, image_folder, mask_folder):
    num_channels = len(bands)
    images = glob.glob(f'{image_folder}/*.tif')
    images.sort()
    masks = glob.glob(f'{mask_folder}/*.tif')
    masks.sort()
    num_images = len(masks)
    x_shape = (num_images, 256,256,num_channels)
    y_shape = (num_images, 256,256, 1)

    x =  np.zeros(x_shape, dtype=np.float16)
    y =  np.zeros(y_shape, dtype=np.uint8)

    num_images = 0

    for i,image_name in enumerate(images):
        landsat_img = process_Landsat8_image(image_name)
        H,W = landsat_img.shape[:2]
        potential_mask_name = [mask_name for mask_name in masks if os.path.basename(image_name) == os.path.basename(mask_name).replace('v1_','')]
        if len(potential_mask_name) <= 0:
            # y[i][0] = 0
            pass
        else:
            x[num_images] = landsat_img[:,:,np.array(bands) - 1]
            y[num_images, :, :, 0] = cv2.imread(potential_mask_name[0], cv2.IMREAD_GRAYSCALE)
            num_images += 1
            print(np.sum(y[num_images, :, :, 0]))

    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x), tf.convert_to_tensor(y)))
    return num_images, dataset

def generate_Landsat8_classification_dataset(bands, image_folder, mask_folder):
    num_channels = len(bands)
    images = glob.glob(f'{image_folder}/*.tif')
    images.sort()
    masks = glob.glob(f'{mask_folder}/*.tif')
    masks.sort()
    num_images = len(images)
    x_shape = (num_images, 256,256,num_channels)
    y_shape = (num_images, 1)

    x =  np.zeros(x_shape, dtype=np.float16)
    y =  np.zeros(y_shape, dtype=np.uint8)

    for i,image_name in enumerate(images):
        landsat_img = tifffile.imread(image_name) / 255
        if np.max(landsat_img) > 1:
            landsat_img = process_Landsat8_image(image_name)
            tifffile.imwrite(image_name, landsat_img * 255)
        H,W = landsat_img.shape[:2]
        x[i] = landsat_img[:,:,np.array(bands) - 1]

        potential_mask_name = [mask_name for mask_name in masks if os.path.basename(image_name) == os.path.basename(mask_name).replace('v1_','')]
        if len(potential_mask_name) == 0:
            y[i][0] = 0
        else:
            mask = tifffile.imread(potential_mask_name[0])
            if np.sum(mask) >= LANDSAT_MIN_FIRE_PIXELS:
                y[i][0] = 1

    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x), tf.convert_to_tensor(y)))
    return num_images, dataset

def generate_USFS_classification_dataset(bands, image_folder, mask_folder):
    num_channels = len(bands)
    images = glob.glob(f'{image_folder}/*.tif')
    images.sort()
    masks = glob.glob(f'{mask_folder}/*.png')
    masks.sort()
    num_images = len(images)
    x_shape = (num_images, 256,256,num_channels)
    y_shape = (num_images, 1)

    x =  np.zeros(x_shape)
    y =  np.zeros(y_shape)

    for i,image_name in enumerate(images):
        usfs_img = load_tif_image(images[i], [], all_bands=True )
        usfs_img = usfs_img/255
        H,W = usfs_img.shape[:2]
        ams_approximation_img=np.rollaxis(np.array([    np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        usfs_img[:,:,0],
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        np.random.rand(H,W),
                                                        ]), 0, 3)
        x[i] = ams_approximation_img[:,:,np.array(bands) - 1]
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        if np.sum(mask > 0) > MIN_FIRE_PIXELS:
            y[i][0] = 1

    dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x), tf.convert_to_tensor(y)))
    return num_images, dataset

def visualize_dataset(dataset, bands = None):
    i = 0
    class_names = ['No Fire', 'Fire']
    fig = plt.figure(figsize=(30, 30))
    fig.patch.set_facecolor('xkcd:black')
    for images, labels in dataset:
        if i > 0:
            break
        i+=1
        images = (images * 255).numpy().astype("uint8")
        if bands is not None: images = images [:,:,:, np.array(bands) - 1]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i], vmin=0, vmax=255, cmap='viridis')
            plt.title(class_names[labels[i].numpy()[0].astype(np.uint8)], fontsize=40,color = 'xkcd:white')
            plt.axis("off")

def calculate_data_distribution_from_folder(mask_folder):
    masks = glob.glob(f'{mask_folder}/*.png')
    num_images = len(masks)
    y_shape = (num_images, 1)

    y =  np.zeros(y_shape)

    for i,image in enumerate(masks):
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        if np.sum(mask > 0) > 0:
            y[i][0] = 1

    print('Positive Examples:', np.sum(y))
    print('Negative Examples:', num_images - np.sum(y))
    print('Total Examples:', num_images)
    return np.sum(y),num_images - np.sum(y)

def calculate_data_distribution_from_dataset(dataset):
    print('Total examples: ', dataset.cardinality())
    total = dataset.cardinality()
    @tf.function
    def count_class(counts, batch):
        y, _, c = tf.unique_with_counts(batch[1])
        y = tf.cast(y, dtype=tf.int32)
        return tf.tensor_scatter_nd_add(counts, tf.expand_dims(y, axis=1), c)

    counts = dataset.reduce(
        initial_state=tf.zeros(2, tf.int32),
        reduce_func=count_class)

    return counts.numpy()

def balance_dataset(dataset, num_images = 1000, batch_size = DEFAULT_BATCH_SIZE, batched=True):
    if batched: dataset = dataset.unbatch()
    negative_ds = (
        dataset
        .filter(lambda features, label:tf.reduce_all(label==0))).shuffle(1024).repeat()
    positive_ds = (
        dataset
        .filter(lambda features, label: tf.reduce_all(label==1))).shuffle(1024).repeat()
    balanced_ds = tf.data.Dataset.sample_from_datasets([negative_ds, positive_ds], [0.5, 0.5])
    balanced_ds = balanced_ds.shuffle(10000).take(num_images)
    return balanced_ds


def split_dataset(num_images, dataset, val_fraction = 0.2, test_fraction = 0.1, batch_size = DEFAULT_BATCH_SIZE, batched = True):
    if batched: dataset = dataset.unbatch()
    train_size = tf.cast( tf.cast(num_images, dtype=tf.float64) *(1 - val_fraction - test_fraction), dtype=tf.int64)
    val_size = tf.cast( tf.cast(num_images, dtype=tf.float64) *(val_fraction), dtype=tf.int64)
    test_size = tf.cast( tf.cast(num_images, dtype=tf.float64) *(test_fraction), dtype=tf.int64)

    train_ds = dataset.take(train_size).shuffle(10000).batch(batch_size)
    val_ds = dataset.skip(train_size).take(val_size).shuffle(10000).batch(batch_size)
    test_ds = dataset.skip(train_size + val_size).take(test_size).shuffle(10000).batch(batch_size)

    return train_ds, val_ds, test_ds










