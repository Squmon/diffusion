import tensorflow as tf
import keras
import keras.layers as layers

def Down():
    return layers.MaxPool2D((2, 2))

def upConv(out_channels, kernel_size, activation='relu'):
    return layers.Conv2DTranspose(out_channels, kernel_size, strides=2, padding='same', activation=activation)

class CrossAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, out_dim):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads, embed_dim, out_dim)
        self.norm = layers.LayerNormalization()

    def call(self, x, context):
        if context.shape[-1] < x.shape[-1]:
            self.context_up = layers.Dense(x.shape[-1])
            context = self.context_up(context)
        elif context.shape[-1] > x.shape[-1]:
            self.x_up = layers.Dense(context.shape[-1])
            x = self.x_up(x)
        # x: (batch, h, w, c), context: (batch, context_len, c)
        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_flat = tf.reshape(x, [b, h * w, c])
        attn_out = self.mha(x_flat, context, x_flat)
        attn_out = self.norm(attn_out + x_flat)
        return tf.reshape(attn_out, [b, h, w, c])

class Unet(keras.Model):
    def __init__(self, bottle_neck_dim = 32, context_dim=1):
        super().__init__()
        # Encoder
        self.enc1 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.enc2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.enc3 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.pool = Down()
        # Bottleneck
        #self.bottleneck = layers.Conv2D(256, 3, activation='relu', padding='same')
        self.bottleneck = keras.Sequential([
            layers.Flatten(),
            layers.Dense(bottle_neck_dim),
        ])
        # Cross Attention at bottleneck
        self.cross_attn = CrossAttention(context_dim)
        # Decoder
        self.up1 = upConv(128, 3)
        self.dec1 = layers.Conv2D(128, 3, activation='relu', padding='same')
        self.up2 = upConv(64, 3)
        self.dec2 = layers.Conv2D(64, 3, activation='relu', padding='same')
        self.up3 = upConv(32, 3)
        self.dec3 = layers.Conv2D(32, 3, activation='relu', padding='same')
        # Output
        self.out_conv = layers.Conv2D(1, 1, activation='sigmoid')

    def call(self, x, context_vector=None):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        # Bottleneck
        b = self.bottleneck(p3)
        # Cross Attention
        if context_vector is not None:
            # context_vector: (batch, context_len, context_dim)
            # Project context to bottleneck dim if needed
            if context_vector.shape[-1] != b.shape[-1]:
                context_proj = layers.Dense(b.shape[-1])
                context_vector = context_proj(context_vector)
            b = self.cross_attn(b, context_vector)
        # Decoder
        u1 = self.up1(b)
        c1 = layers.Concatenate()([u1, e3])
        d1 = self.dec1(c1)
        u2 = self.up2(d1)
        c2 = layers.Concatenate()([u2, e2])
        d2 = self.dec2(c2)
        u3 = self.up3(d2)
        c3 = layers.Concatenate()([u3, e1])
        d3 = self.dec3(c3)
        # Output
        out = self.out_conv(d3)
        return out