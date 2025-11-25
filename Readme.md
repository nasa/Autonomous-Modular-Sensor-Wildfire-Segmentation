## Authors & Contact Information
Yajvan Ravan (NASA Langley Intern): yajvanravan@gmail.com

Aref Malek (NASA Langley Intern): malek.aref1@gmail.com

Chester Dolph (NASA Langley): chester.v.dolph@nasa.gov

Nikhil Behari (NASA Langley Intern): nikhilbehari@gmail.com


## Creating Environment
To run the code, first create a Conda environment with the necessary dependencies by running the following code at the terminal

    conda create --file windows_environment.yml

Replace the yaml file with the mac one if you are using a mac (the only difference is the usage of cuda)

## Dataset
Our custom dataset can be found at [here]. It includes 4875 patches of size 256 x 256 for training and 489 for testing. *generate_data_for_annotation.ipynb* was used to isolate images that were then manually annotated using Roboflow. It also includes 18 train and 2 test images from the AMS flights (in data/ams_data/processed_images_new/labelled_images) that were then patched to create the dataset. Note that these are TIFF images with 12-16 channels each, and 3 channel views are also provided in color_infrared, fire_emphasizing, and visual_spectrum. To use this dataset, download it and add it as a folder in this repository. You can patch the dataset on your own, to generate a custom random sample of patches using patch_dataset.ipynb. 

The dataset also contains our trained model weights, which you can use in the simulation.

## Data Preprocessing

1. AMS_dataset_processing.py
2. dataset_patching.py
3. process_ams_data.ipynb
4. patch_dataset.ipynb
5. dataset_utils.py 

These files contain methods and their usage to clean the raw images captured by the AMS sensor. This includes reorienting images so that they are compass aligned, resampling images to a consistent resolution, normalizing images, and splitting images, along with their masks, into random patches. The last one contains code to turn our dataset into a Tensorflow dataset. The tensorflow datasets in our folder already include all of this processing.

## Models
1. classifier_model.py
2. segmentation_model.py
3. combined_algorithm.py

These files contain the network architectures for our model and the last one combines the networks into a complete wildfire localization model. 

## Training/Testing
1. training_utils.py
2. train_classifier.ipynb
3. train_segmenter.ipynb
4. optimize_classifier_hyperparameters.ipynb
5. test_models.ipynb
6. combine_models.ipynb
7. simulation.py

These files contain code to train, test, and combine our models. 1 contains utilities to visualize training metrics. 2 and 3 were used to train our networks. 4 was used to optimize network hyperparameters. 5 was used for testing. 6 and 7 were used to combine the networks and simulate the entire model. Trained model weights are in the data folder. The simulation can be run with 7 (simulation.py), and the model weights & image can be changed if desired.

## Benchmarking/Results
1. generate_figures.ipynb
2. benchmark_landsat_results.ipynb

These files contain code to generate figures and results for our paper. The latter contains a reference to a file 'pereira_segmentation_models.py' that uses code from *[Active Fire Detection in Landsat-8 Imagery: a Large-Scale Dataset and a Deep-Learning Study](https://github.com/pereira-gha/activefire)* to compare our results. The file itself has been removed for copyright, but the code can be found in the original authors' Github


