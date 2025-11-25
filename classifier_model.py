import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


def build_simple_encoder(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=1,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool1')(x1)
    x3 = layers.BatchNormalization()(x2)

    x4 = layers.Conv2D(filters=1,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x5 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool2')(x4)
    x6 = layers.BatchNormalization()(x5)

    x7 = layers.Conv2D(filters=1,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool3')(x7)
    x9 = layers.BatchNormalization()(x8)

    x10 = layers.Conv2D(filters=1,kernel_size=5,strides=1,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    x11 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool4')(x10)
    x12 = layers.BatchNormalization()(x11)
    
    # x13 = layers.Conv2D(filters=1,kernel_size=5,strides=1,activation='relu', padding = 'same', name=f'conv9', kernel_initializer=initializer)(x12)
    # x14 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool5')(x13)
    # x15 = layers.BatchNormalization()(x14)

    x16 = layers.Flatten()(x12) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17

def build_simple_encoder_2(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=8,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=4,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=2,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool3')(x8)

    x10 = layers.Conv2D(filters=1,kernel_size=5,strides=1,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    x11 = layers.BatchNormalization()(x10)
    x115 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool4')(x11)
    x12= layers.Dropout(0.1)(x115)
    # x13 = layers.Conv2D(filters=1,kernel_size=5,strides=1,activation='relu', padding = 'same', name=f'conv9', kernel_initializer=initializer)(x12)
    # x14 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool5')(x13)
    # x15 = layers.BatchNormalization()(x14)

    x16 = layers.Flatten()(x12) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17


def build_simple_encoder_3(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=12,kernel_size=5,strides=1,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=6,kernel_size=5,strides=1,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=3,kernel_size=5,strides=1,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool3')(x8)

    x10 = layers.Conv2D(filters=1,kernel_size=5,strides=1,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    x11 = layers.BatchNormalization()(x10)
    x115 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x115) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17

def build_simple_encoder_4(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=12,kernel_size=5,strides=1,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=6,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=3,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool3')(x8)

    x10 = layers.Conv2D(filters=1,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    x11 = layers.BatchNormalization()(x10)
    x115 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x115) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17

def build_simple_encoder_5(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=12,kernel_size=5,strides=1,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=6,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=3,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool3')(x8)

    x10 = layers.Conv2D(filters=1,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    x11 = layers.BatchNormalization()(x10)
    x115 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x115) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17

def build_simple_encoder_6(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=12,kernel_size=3,strides=1,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=6,kernel_size=3,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=3,kernel_size=3,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool3')(x8)

    x10 = layers.Conv2D(filters=1,kernel_size=3,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    x11 = layers.BatchNormalization()(x10)
    x115 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x115) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17

def build_simple_encoder_7(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=12,kernel_size=7,strides=1,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=6,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=3,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool3')(x8)

    x10 = layers.Conv2D(filters=1,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    x11 = layers.BatchNormalization()(x10)
    x115 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x115) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17


def build_simple_encoder_8(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=12,kernel_size=7,strides=1,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=6,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=3,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool3')(x8)

    x10 = layers.Conv2D(filters=1,kernel_size=3,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    x11 = layers.BatchNormalization()(x10)
    x115 = layers.MaxPool2D(pool_size=3,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x115) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17

def build_simple_encoder_9(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=6,kernel_size=7,strides=1,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=3,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=1,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool3')(x8)

    x10 = layers.Conv2D(filters=1,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    x11 = layers.BatchNormalization()(x10)
    x115 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x115) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17

def build_simple_encoder_10(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=2,kernel_size=7,strides=1,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=4,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=8,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool3')(x8)

    x10 = layers.Conv2D(filters=16,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    x11 = layers.BatchNormalization()(x10)
    x115 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x115) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17

def build_simple_encoder_11(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=2,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=4,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=8,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool3')(x8)

    # x10 = layers.Conv2D(filters=16,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    # x11 = layers.BatchNormalization()(x10)
    # x115 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x9) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17

def build_simple_encoder_12(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=2,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x25 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool1')(x2)
    x3 = layers.BatchNormalization()(x25)

    x4 = layers.Conv2D(filters=4,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x55 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool2')(x3)
    x6 = layers.BatchNormalization()(x55)

    x7 = layers.Conv2D(filters=8,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x85 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool3')(x8)
    x9 = layers.BatchNormalization()(x85)

    # x10 = layers.Conv2D(filters=16,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    # x11 = layers.BatchNormalization()(x10)
    # x115 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x9) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17

def build_simple_encoder_13(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=1,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=2,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=4,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool3')(x8)

    # x10 = layers.Conv2D(filters=16,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    # x11 = layers.BatchNormalization()(x10)
    # x115 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x9) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17

def build_simple_encoder_14(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=12,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=6,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=3,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool3')(x8)

    # x10 = layers.Conv2D(filters=16,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    # x11 = layers.BatchNormalization()(x10)
    # x115 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x9) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17


def build_simple_encoder_15(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x = layers.Conv2D(filters=6,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer, kernel_regularizer='l1')(input)
    x = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool1', padding = 'same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=3,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer, kernel_regularizer='l1')(x)
    x = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool2', padding = 'same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=1,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer, kernel_regularizer='l1')(x)
    x = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool3', padding = 'same')(x)
    x = layers.BatchNormalization()(x)

    # x = layers.Conv2D(filters=16,kernel_size=5,strides=1,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer, kernel_regularizer='l1')(x)
    # x = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool4', padding = 'same')(x)
    # x = layers.BatchNormalization()(x)

    x = layers.GlobalMaxPool2D()(x) 
    x = layers.BatchNormalization()(x)

    x17 = layers.Dense(1, kernel_initializer=initializer)(x)

    return x17

def build_simple_encoder_16(input):
    initializer = None #tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=1,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=2,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Conv2D(filters=4,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv5', kernel_initializer=initializer)(x6)
    x8 = layers.BatchNormalization()(x7)
    x9 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool3')(x8)

    # x10 = layers.Conv2D(filters=16,kernel_size=5,strides=2,activation='relu', padding = 'same', name=f'conv7', kernel_initializer=initializer)(x9)
    # x11 = layers.BatchNormalization()(x10)
    # x115 = layers.MaxPool2D(pool_size=2,strides=2,name=f'maxpool4')(x11)

    x16 = layers.GlobalMaxPool2D()(x9) 

    x17 = layers.Dense(1, kernel_initializer=initializer)(x16)

    return x17


def build_simple_encoder_17(input):
    initializer = None# tf.keras.initializers.RandomNormal(mean=0., stddev=1e-2, seed=4)
    x1 = layers.Conv2D(filters=1,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv1', kernel_initializer=initializer)(input)
    x2 = layers.BatchNormalization()(x1)
    x3 = layers.MaxPool2D(pool_size=4,strides=2,name=f'maxpool1')(x2)

    x4 = layers.Conv2D(filters=2,kernel_size=7,strides=2,activation='relu', padding = 'same', name=f'conv3', kernel_initializer=initializer)(x3)
    x3 = layers.BatchNormalization()(x4)
    x6 = layers.MaxPool2D(pool_size=4,strides=2,name=f'maxpool2')(x3)

    x7 = layers.Flatten()(x6)
    x17 = layers.Dense(1, kernel_initializer=initializer)(x7)

    return x17


def build_resnet(input):
    return tf.keras.applications.resnet50.ResNet50(
    include_top=True,
    weights='imagenet',
    input_tensor=input,
    classes=2,
    )



def build_model(num_input_channels, model_name, output_bias = None):
    input = keras.Input(shape=(256,256,num_input_channels))
    
    # if model_name == 'custom':
    #     squeeze_net_output = build_custom_squeeze_net(input, output_bias)
    # if model_name == 'original':
    #     squeeze_net_output = build_original_squeeze_net(input, output_bias)
    # if model_name == 'resnet':
    #     squeeze_net_output = build_resnet(input)
    # if model_name == 'lenet':
    #     squeeze_net_output = build_lenet(input)
    # if model_name == 'custom_2':
    #     squeeze_net_output = build_custom_squeeze_net_2(input)
    # if model_name == 'unet':
    #     squeeze_net_output = build_unet_encoder(input)
    # if model_name == 'unet_2':
    #     squeeze_net_output = build_unet_encoder_2(input)
    if model_name == 'simple':
        # squeeze_net_output = build_simple_encoder_17(input)
        squeeze_net_output = build_simple_encoder_16(input)
        # squeeze_net_output = build_simple_encoder_15(input)
        # squeeze_net_output = build_simple_encoder_14(input)
        # squeeze_net_output = build_simple_encoder_13(input)
        # squeeze_net_output = build_simple_encoder_12(input)
        # squeeze_net_output = build_simple_encoder_11(input)
        # squeeze_net_output = build_simple_encoder_10(input)
        # squeeze_net_output = build_simple_encoder_9(input)
        # squeeze_net_output = build_simple_encoder_8(input)
        # squeeze_net_output = build_simple_encoder_7(input)
        # squeeze_net_output = build_simple_encoder_6(input)
        # squeeze_net_output = build_simple_encoder_5(input)
        # squeeze_net_output = build_simple_encoder_4(input)
        # squeeze_net_output = build_simple_encoder_3(input)
        # squeeze_net_output = build_simple_encoder_2(input)
        # squeeze_net_output = build_simple_encoder(input)

    squeeze_net_model = keras.Model(inputs=input, outputs=squeeze_net_output)

    return squeeze_net_model

def build_classifier_model(bands, load_weights=True, MODEL_PATH=None):
    """
    Builds and compiles a classification model for the given band(s).
    Loads pretrained weights if available.

    Args:
        bands (List[int]): List of band indices.

    Returns:
        tf.keras.Model: Compiled classification model.
    """
    MODEL_PATH = MODEL_PATH or f'./data/model_weights/bands_{"_".join(map(str, bands))}_classifier.h5'

    # Define the model
    model = build_model(len(bands), 'simple', None)

    # Compile the model
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(
            learning_rate=7.5e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
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
        ]
    )

    # Load weights if they exist
    if os.path.exists(MODEL_PATH) and load_weights:
        print(f"🔄 Loading pretrained weights from {MODEL_PATH}")
        model.load_weights(MODEL_PATH)
    else:
        print(f"📭 No pretrained weights found at {MODEL_PATH}")

    return model

if __name__=='__main__':
    num_channels = 1
    inputs = keras.Input(shape=(192,192,num_channels))

    squeeze_net_output = build_custom_squeeze_net(inputs)

    squeeze_net_model = keras.Model(inputs=inputs, outputs=squeeze_net_output)

    squeeze_net_model.summary()