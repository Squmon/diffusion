import tensorflow as tf
import keras
import keras.layers as layers

def Down():
    return layers.MaxPool2D((2, 2))

def upConv(out_channels, kernel_size, activation = 'leaky_relu'):
    return layers.Conv2DTranspose(out_channels, kernel_size)

class Unet(keras.Model):
    def __init__(self, context_res = 1, ):
        super().__init__()

    def call(self, x, context_vector):
        pass