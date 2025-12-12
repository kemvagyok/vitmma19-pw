import tensorflow as tf
import keras
from keras.layers import Conv2D, Input, MaxPool2D, GlobalAveragePooling2D, Dense

def SimpleCNNModel():
    model = keras.Sequential(
    [      
        Input((256,256,3)),
        Conv2D(4, (3,3), activation='relu'),
        MaxPool2D((2,2)),
        Conv2D(8, (3,3), activation='relu'),
        GlobalAveragePooling2D(),
        Dense(3)
    ])
    return model
