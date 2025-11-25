import training_utils
import segmentation_model
import classifier_model
import dataset_utils
import os
import tensorflow as tf
from tensorflow import keras


class StepMetricLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_accuracy_per_step = []
        self.val_loss_per_step = []
        self.steps = []
        self.global_step = 0

    def on_test_batch_end(self, batch, logs=None):
        # Logs contain batch-level validation metrics
        self.val_accuracy_per_step.append(logs.get('val_accuracy'))
        self.val_loss_per_step.append(logs.get('val_loss'))
        self.steps.append(self.global_step)
        self.global_step += 1

    def on_epoch_end(self, epoch, logs=None):
        # Logs at the end of each epoch
        print(f"Epoch {epoch + 1} ended.")


# Instantiate the callback
step_logger = StepMetricLogger()


BATCH_SIZE = 100
# num_test_images , gen_test_dataset = dataset_utils.get_dataset([11], './data/ams_data/processed_images_new/patches/test/images', './data/ams_data/processed_images_new/patches/test/labels', dataset_name=None, dataset_source='AMS', dataset_type='classification', test=True)
num_test_images , gen_test_dataset = dataset_utils.get_dataset([7], './data/landsat_data/manual_annotations/patches/landsat_patches', './data/landsat_data/manual_annotations/patches/manual_annotations_patches', dataset_name=None, dataset_source='Landsat8', dataset_type='classification', test=True)
gen_test_dataset = gen_test_dataset.batch(BATCH_SIZE, drop_remainder=False)
gen_test_dataset = dataset_utils.balance_dataset(gen_test_dataset, num_images=num_test_images)
gen_test_dataset = gen_test_dataset.batch(BATCH_SIZE, drop_remainder=False)
print(num_test_images)

num_train_images , gen_train_dataset = dataset_utils.get_dataset([10], './data/ams_data/processed_images_new/patches/train/images', './data/ams_data/processed_images_new/patches/train/labels', dataset_name=None, dataset_source='AMS', dataset_type='classification', test=False)
# num_train_images , gen_train_dataset = dataset_utils.get_dataset([0], './data/diffusion_dataset/trial_1/images', './data/diffusion_dataset/trial_1/labels',dataset_name=None, dataset_source='Generative', dataset_type='classification', test=False)
# num_train_images , gen_train_dataset = dataset_utils.get_dataset([0], './data/diffusion_dataset/trial_2/images', './data/diffusion_dataset/trial_2/labels',dataset_name=None, dataset_source='Generative', dataset_type='classification', test=False)
# num_train_images , gen_train_dataset = dataset_utils.get_dataset([0], './data/diffusion_dataset/trial_3/images', './data/diffusion_dataset/trial_3/labels',dataset_name=None, dataset_source='Generative', dataset_type='classification', test=False)
gen_train_dataset = gen_train_dataset.batch(BATCH_SIZE, drop_remainder=False)
gen_train_dataset = dataset_utils.balance_dataset(gen_train_dataset, num_images=num_train_images)
gen_train_dataset = gen_train_dataset.batch(BATCH_SIZE, drop_remainder=False)
print(num_train_images)

gen_test_dataset = gen_test_dataset.cache().apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
gen_train_dataset = gen_train_dataset.cache().apply(tf.data.experimental.prefetch_to_device("/gpu:0"))

classification_model= classifier_model.build_model(1, 'simple', None)
classification_model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3,beta_1=0.9,beta_2=0.999),
    metrics=[
        keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'),
    ],
)
classification_model.evaluate(gen_test_dataset, verbose=2)
history = classification_model.fit(gen_train_dataset,
                                    epochs=100,
                                    class_weight={0:1, 1:1},
                                    validation_data=gen_test_dataset,
                                    callbacks=[step_logger,
                                        # keras.callbacks.EarlyStopping( monitor="val_accuracy",min_delta=5e-3,patience=10,verbose=1)],
                                        #        tf.keras.callbacks.ModelCheckpoint(
                                        #                     filepath='./data/model_weights/gen_classifier_test.keras',
                                        #                     save_weights_only=True,
                                        #                     monitor='val_accuracy',
                                        #                     mode='max',
                                        #                     save_best_only=True)
                                            ],
                                    verbose=1
                                    )
classification_model.evaluate(gen_test_dataset, verbose=2)

val_accuracy = step_logger.val_accuracy_per_step
val_loss = step_logger.val_accuracy_per_step
steps = step_logger.steps
import numpy as np
print(np.mean(val_accuracy))
print(np.mean(steps))

import matplotlib.pyplot as plt
plt.plot(steps, val_accuracy, label='val_accuracy')
plt.show()
