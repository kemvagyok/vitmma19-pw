import tensorflow as tf
import keras
from keras.layers import Conv2D, Input, MaxPool2D, GlobalAveragePooling2D, Dense
import numpy as np

import config

def baseline_model():
    def predict():
        return np.array([0,1,0], dtype=np.float32)
    return predict

def simpleCNNModel(num_classes=3, input_shape=(128,128,3)):
    model = keras.Sequential([
        Input(shape=input_shape),

        Conv2D(8, 3, activation='relu'),
        MaxPool2D(),

        Conv2D(16, 3, activation='relu'),
        MaxPool2D(),

        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')  # konzisztens output!
    ])

    return model



def advancedCNNModel(num_classes=3, input_shape=(config.TARGET_IMAGE_SIZE[0], config.TARGET_IMAGE_SIZE[1],3)):
    model = keras.Sequential([
        keras.Input(shape = input_shape),

        Conv2D(32, 3, activation='relu'),
        MaxPool2D(),

        Conv2D(64, 3, activation='relu'),
        MaxPool2D(),

        GlobalAveragePooling2D(),

        Dense(64, activation='relu'),

        Dense(num_classes, activation='softmax')
    ])

    return model