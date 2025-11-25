import training_utils
import segmentation_model
import classifier_model
import dataset_utils
import os
import tensorflow as tf
from tensorflow import keras

def add_sample_weights(image, label):
  class_weights = tf.constant([1.0, 50.0])
  class_weights = class_weights/tf.reduce_sum(class_weights)
  sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
  return image, label, sample_weights


BATCH_SIZE = 100
num_test_images , gen_test_dataset = dataset_utils.get_dataset([12], './data/ams_data/processed_images_new/patches/test/images', './data/ams_data/processed_images_new/patches/test/labels', dataset_name=None, dataset_source='AMS', dataset_type='segmentation', test=True)
gen_test_dataset = gen_test_dataset.shuffle(5000).batch(BATCH_SIZE, drop_remainder=False)
print(num_test_images)

num_train_images , gen_train_dataset = dataset_utils.get_dataset([0], './data/diffusion_dataset/trial_1/images', './data/diffusion_dataset/trial_1/labels',dataset_name=None, dataset_source='Generative', dataset_type='segmentation', test=False)
# num_train_images , gen_train_dataset = dataset_utils.get_dataset([12], './data/ams_data/processed_images_new/patches/train/images', './data/ams_data/processed_images_new/patches/train/labels', dataset_name=None, dataset_source='AMS', dataset_type='segmentation', test=False)
gen_train_dataset = gen_train_dataset.shuffle(5000).batch(BATCH_SIZE, drop_remainder=False)
print(num_train_images)

gen_test_dataset = gen_test_dataset.cache().apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
gen_train_dataset = gen_train_dataset.cache().apply(tf.data.experimental.prefetch_to_device("/gpu:0"))

segmentation_model_ir_generative = segmentation_model.get_trimmed_unet_model(input_channel_count=1, output_channels=1)
segmentation_model_ir_generative.compile(
                optimizer= tf.keras.optimizers.Adam(learning_rate=1e-03), # tf.keras.optimizers.SGD(learning_rate=0.01),
                loss= tf.keras.losses.BinaryCrossentropy(), # tf.keras.losses.BinaryCrossentropy(),
                metrics=[segmentation_model.IOU(num_classes=2,name='mean_iou'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall')],
                weighted_metrics=[]
            )
segmentation_model_ir_generative.evaluate(gen_test_dataset.map(add_sample_weights), verbose=2)
history = segmentation_model_ir_generative.fit(gen_train_dataset.map(add_sample_weights),
                                    epochs=150,
                                    # class_weight={0:1,1:100},
                                    validation_data=gen_test_dataset.map(add_sample_weights),
                                    # callbacks=[keras.callbacks.EarlyStopping( monitor="val_mean_iou",min_delta=5e-3,patience=50,verbose=1),],
                                               # tf.keras.callbacks.ModelCheckpoint(
                                               #              filepath='./data/model_weights/diffusion_segmentation_bands_11_trial1.weights.h5',
                                               #              save_weights_only=True,
                                               #              monitor='val_mean_iou',
                                               #              mode='max',
                                               #              save_best_only=True)],
                                    verbose=1
                                    )
segmentation_model_ir_generative.evaluate(gen_test_dataset.map(add_sample_weights), verbose=2)


