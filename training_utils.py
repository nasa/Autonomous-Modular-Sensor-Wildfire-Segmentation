import tensorflow as tf
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import seaborn as sns
import time
import dataset_utils
import segmentation_model
import classifier_model
from tensorflow import keras
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' for even less output


WEIGHTS_FOLDER = './data/model_weights'
HYPERPARAMETER_CHECKPOINTS_FOLDER = './data/hyperparameter_checkpoints'
SEGMENTATION_BATCH_SIZE = 4
CLASSIFIER_BATCH_SIZE = 100
DATA_ROOT = './data/ams_data/processed_images_new/patches'
GPU_DEVICE = "/gpu:0"

# tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.test.is_gpu_available ())
tf.device('/gpu:0')



mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_roc(name, labels, predictions, figure_num=1, **kwargs):
  f1 = plt.figure(figure_num)
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

def plot_prc(name, labels, predictions, figure_num=1, **kwargs):
    f1 = plt.figure(figure_num)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

def plot_cm(labels, predictions, p=0.5, figure_num=1):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figure_num,figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('No Fire Detected (True Negatives): ', cm[0][0])
  print('No Fire Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fire Missed (False Negatives): ', cm[1][0])
  print('Fire Detected (True Positives): ', cm[1][1])
  print('Total Fire: ', np.sum(cm[1]))

def plot_metrics(history, figure_num=1):
  f1 = plt.figure(figure_num)
  metrics = ['loss', 'prc', 'precision', 'recall', 'accuracy', 'auc' ]
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,3,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()

def plot_loss(history, label = '', n=0, figure_num=1):
  f1 = plt.figure(figure_num)
  # Use a log scale on y-axis to show the wide range of values.
  plt.semilogy(history.epoch, history.history['loss'],
               color=colors[n], label='Train ' + label)
  plt.semilogy(history.epoch, history.history['val_loss'],
               color=colors[n], label='Val ' + label,
               linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

def get_model_predictions(model, dataset):
   y_pred = []  # store predicted labels
   y_true = []  # store true labels

   total_time = 0

   # iterate over the dataset
   for image_batch, label_batch in dataset:   # use dataset.unbatch() with repeat
      # append true labels
      y_true.append(label_batch)
      # compute predictions
      start_time = time.perf_counter()
      preds = tf.math.sigmoid(model.predict(image_batch))
      total_time += time.perf_counter() - start_time
      # append predicted labels
      y_pred.append(preds)

   # convert the true and predicted labels into tensors
   correct_labels = tf.concat([item for item in y_true], axis = 0)
   predicted_labels = tf.concat([item for item in y_pred], axis = 0)

   avg_time = total_time / tf.size(correct_labels).numpy()

   return correct_labels, predicted_labels, avg_time

def visualize_training_results(history, figure_num=1):
    plot_loss(history, label = '', n=0, figure_num=figure_num)
    figure_num += 1
    plot_metrics(history, figure_num=figure_num)
    return figure_num

def visualize_test_results(labels = None, predictions = None, figure_num=1, model = None, test_dataset = None):
    if model is not None:
       labels, predictions, avg_time = get_model_predictions(model, test_dataset)
       print("Average Inference Time (s): ", avg_time)
    if labels is None:
        return figure_num
    plot_cm(labels, predictions, p=0.5, figure_num=figure_num)
    figure_num += 1
    plot_prc('Precision Recall Curve', labels, predictions, figure_num=figure_num)
    figure_num += 1
    plot_roc('Precision Recall Curve', labels, predictions, figure_num=figure_num)
    return figure_num

def visualize_results(history, labels = None, predictions = None, model = None, test_dataset = None):
    figure_num=1
    figure_num = visualize_training_results(history, figure_num=figure_num)
    if labels is not None:
      figure_num += 1
      figure_num = visualize_test_results(labels = labels, predictions = predictions, figure_num=figure_num)
    if model is not None:
      figure_num += 1
      figure_num = visualize_test_results(model = model, test_dataset = test_dataset, figure_num=figure_num)
    plt.show()

def add_sample_weights(image, label):
  class_weights = tf.constant([1.0, 50.0])
  class_weights = class_weights/tf.reduce_sum(class_weights)
  sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
  return image, label, sample_weights


def train_and_store(bands, segmentation=True, runs=5):
    """
    Trains the model multiple times and averages the final evaluation metrics.

    Args:
        bands (List[int]): List of band indices to use.
        segmentation (bool): Whether to use segmentation or classification model.
        runs (int): Number of repeated trainings for averaging.

    Returns:
        str: Path to the final saved model weights.
    """

    def prepare_dataset(split: str, test=False):
        num_images, dataset = dataset_utils.get_dataset(
            bands,
            os.path.join(DATA_ROOT, split, 'images'),
            os.path.join(DATA_ROOT, split, 'labels'),
            dataset_name=None,
            dataset_source='AMS',
            dataset_type='segmentation' if segmentation else 'classification',
            test=test
        )
        print(f"{split} images:", num_images)
        dataset = dataset.shuffle(5000).batch(
            SEGMENTATION_BATCH_SIZE if segmentation else CLASSIFIER_BATCH_SIZE,
            drop_remainder=not test
        )
        dataset = dataset.cache().apply(tf.data.experimental.prefetch_to_device(GPU_DEVICE))
        return dataset

    # Load datasets once
    train_ds = prepare_dataset("train", test=False)
    test_ds = prepare_dataset("test", test=True)

    if segmentation:
        train_ds = train_ds.map(add_sample_weights)
        test_ds = test_ds.map(add_sample_weights)

    all_metrics = []
    metrics_names = None

    for run in range(1, runs + 1):
        print(f"\n🚀 Starting training run {run}/{runs}")

        MODEL_PATH = f"./data/model_weights/bands_{'_'.join(map(str, bands))}_{'segmenter' if segmentation else 'classifier'}_trial{run}.h5"
        if not os.path.exists(MODEL_PATH):
            model = (
                segmentation_model.build_segmentation_model(bands, load_weights=False, MODEL_PATH=MODEL_PATH)
                if segmentation
                else classifier_model.build_classifier_model(bands, load_weights=False, MODEL_PATH=MODEL_PATH)
            )

            if segmentation:
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_mean_iou",
                        min_delta=5e-3,
                        patience=10,
                        verbose=1
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath=MODEL_PATH,
                        save_weights_only=True,
                        monitor='val_mean_iou',
                        mode='max',
                        save_best_only=True
                    )
                ]
            else:
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        min_delta=5e-3,
                        patience=20,
                        verbose=1
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath=MODEL_PATH,
                        save_weights_only=True,
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True
                    )
                ]

            model.fit(
                train_ds,
                epochs=500,
                validation_data=test_ds,
                callbacks=callbacks,
                verbose=1
            )

        # Reload best weights before evaluation
        model = (
            segmentation_model.build_segmentation_model(bands, load_weights=True, MODEL_PATH=MODEL_PATH)
            if segmentation
            else classifier_model.build_classifier_model(bands, load_weights=True, MODEL_PATH=MODEL_PATH)
        )

        print(f"\n📊 Evaluating model from run {run} on test dataset:")
        metrics = model.evaluate(test_ds, verbose=0)
        all_metrics.append(metrics)
        metrics_names = metrics_names or model.metrics_names

        print(f"✅ Run {run} results:")
        for name, value in zip(metrics_names, metrics):
            print(f"   🔹 {name}: {value:.4f}")

        # Clean up before next run
        del model
        gc.collect()

    # Average evaluation metrics across runs
    all_metrics = np.array(all_metrics)
    mean_metrics = np.mean(all_metrics, axis=0)
    std_metrics = np.std(all_metrics, axis=0)

    print("\n📈 Final Averaged Evaluation Metrics Across Runs:")
    for name, mean, std in zip(metrics_names, mean_metrics, std_metrics):
        print(f"   🔸 {name}: {mean:.4f} ± {std:.4f}")

    # Final cleanup
    del train_ds
    del test_ds
    gc.collect()

    print(metrics_names)

    return MODEL_PATH, mean_metrics, std_metrics