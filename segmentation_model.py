import tensorflow as tf
import os
from tensorflow import keras

class IOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.where(y_pred > 0.5, 1.0,0.0)
        return super().update_state(y_true,y_pred,sample_weight)

def encoding_block(inputs, output_channels):
    conv1 = tf.keras.layers.Conv2D(filters=output_channels, 
                                   kernel_size=(3, 3), 
                                   strides=1,
                                   padding='same',
                                   activation='relu'
                                   )(inputs)
    conv1 = tf.keras.layers.Conv2D(filters=output_channels,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding='same',
                                   activation='relu'
                                   )(conv1)
    batchNorm = tf.keras.layers.BatchNormalization()(conv1)
    # activated = tf.keras.layers.relu()(conv1)
    pooled = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(batchNorm)
    return pooled, conv1 # both the pooled for next layer, and conv1 layer for skip connection

def decoding_block(inputs, skip_connection, out_channels):
    # upsample our inputs guy
    inputs = tf.keras.layers.Convolution2DTranspose(
        filters=out_channels,
        kernel_size=(2,2),
        strides=(2,2),
        padding="same"
    )(inputs)
    b, w, h, c = inputs.shape

    # crop the guysies
    cropped = tf.keras.layers.CenterCrop(
        height=h,
        width=w
    )(skip_connection)
    cropped = tf.cast(cropped, dtype=tf.float32)

    concatenate = tf.concat([cropped, inputs], axis=3)

    # convolve over this guy
    conv1 = tf.keras.layers.Conv2D(filters=out_channels,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding='same',
                                   activation='relu'
                                   )(concatenate)
    conv1 = tf.keras.layers.Conv2D(filters=out_channels,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding='same',
                                   activation='relu'
                                   )(conv1)
    # activated = tf.keras.layers.relu()(conv1)
    
    return conv1


def get_unet_model(input_channel_count: int, output_channels: int):

    inputs = tf.keras.layers.Input(shape=(256,256,input_channel_count))

    x, c1 = encoding_block(inputs, 64)
    x, c2 = encoding_block(x, 128)
    x, c3 = encoding_block(x, 256)
    x, c4 = encoding_block(x, 512)
    _, x = encoding_block(x, 1024)

    x = decoding_block(x, c4, 512)
    x = decoding_block(x, c3, 256)
    x = decoding_block(x, c2, 128)
    x = decoding_block(x, c1, 64)

    outputs = tf.keras.layers.Conv2D(output_channels, 1, padding="same", activation = "softmax")(x)
    unet = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet

def get_trimmed_unet_model(input_channel_count: int, output_channels: int):

    inputs = tf.keras.layers.Input(shape=(256,256,input_channel_count))
    x, c1 = encoding_block(inputs, 32)
    x, c2 = encoding_block(x, 64)
    _, x = encoding_block(x, 128)

    x = decoding_block(x, c2, 64)
    x = decoding_block(x, c1, 32)

    outputs = tf.keras.layers.Conv2D(output_channels, 1, padding="same", activation = "sigmoid")(x)
    unet = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet

def build_segmentation_model(bands, load_weights=True, MODEL_PATH=None):
    """
    Builds and compiles a segmentation model for the given band(s).
    If pretrained weights exist, they are loaded.

    Args:
        bands (List[int]): List of band indices.

    Returns:
        tf.keras.Model: Compiled U-Net segmentation model.
    """
    MODEL_PATH = MODEL_PATH or f'./data/model_weights/bands_{"_".join(map(str, bands))}_segmenter.h5'

    model = get_trimmed_unet_model(
        input_channel_count=len(bands),
        output_channels=1
    )

    # Compile before loading weights (not strictly necessary, but helps debugging)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            IOU(num_classes=2, name='mean_iou'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
        weighted_metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
        ]
    )

    if os.path.exists(MODEL_PATH) and load_weights:
        print(f"Loading pretrained weights from {MODEL_PATH}")
        model.load_weights(MODEL_PATH)
    else:
        print(f"No pretrained weights found at {MODEL_PATH}")

    return model