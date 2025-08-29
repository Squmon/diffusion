import tensorflow as tf
import keras
import keras.layers as layers

class Unet(keras.Model):
    def __init__(self):
        super(Unet, self).__init__()
        
        # Encoder
        self.conv1_1 = layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same')
        self.conv1_2 = layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2, 2))
        
        self.conv2_1 = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')
        self.conv2_2 = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2))
        
        # Bottleneck
        self.conv3_1 = layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same')
        self.conv3_2 = layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same')
        
        # Decoder
        self.up4 = layers.UpSampling2D((2, 2))
        self.conv4_1 = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')
        self.conv4_2 = layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')
        
        self.up5 = layers.UpSampling2D((2, 2))
        self.conv5_1 = layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same')
        self.conv5_2 = layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same')
        self.output_conv = layers.Conv2D(1, (1, 1), activation='linear', padding='same')
        
    def call(self, inputs):
        x, time = inputs
        
        time_scalar = tf.squeeze(time)
        time_channel = tf.fill(tf.shape(x), time_scalar)
        
        x_with_time = tf.concat((x, time_channel), axis=-1)
        
        # Encoder
        c1 = self.conv1_1(x_with_time)
        c1 = self.conv1_2(c1)
        p1 = self.pool1(c1)
        
        c2 = self.conv2_1(p1)
        c2 = self.conv2_2(c2)
        p2 = self.pool2(c2)
        
        # Bottleneck
        c3 = self.conv3_1(p2)
        c3 = self.conv3_2(c3)
        
        # Decoder
        u4 = self.up4(c3)
        u4 = layers.concatenate([u4, c2])
        c4 = self.conv4_1(u4)
        c4 = self.conv4_2(c4)
        
        u5 = self.up5(c4)
        u5 = layers.concatenate([u5, c1])
        c5 = self.conv5_1(u5)
        c5 = self.conv5_2(c5)
        
        output = self.output_conv(c5)
        
        return output