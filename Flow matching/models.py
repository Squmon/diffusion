import tensorflow as tf
from keras import layers, Model
import keras


def block_down(x, filter, factor, config, name):
    cfg = config["model"]["encoder_block"]

    # 1. projection
    protected_params = {
        "filters": filter,
        "name": f"{name}_conv_down",
        "strides": factor,
        "padding": "same"
    }
    x = layers.Conv2D(**(cfg['base_down'] | protected_params))(x)

    if cfg["use_bn"]:
        x = layers.BatchNormalization(name=f"{name}_bn_1")(x)
    x = layers.Activation(cfg["activation"], name=f"{name}_act_1")(x)

    # 2. Another non-linearity
    if "extra_nonlin" in cfg:
        protected_params = {
            "filters": filter,
            "strides": 1,  # Явно фиксируем, чтобы случайно не сжать второй раз
            "padding": "same",
            "name": f"{name}_conv_extra"
        }
        x = layers.Conv2D(**(cfg['extra_nonlin'] | protected_params))(x)

        if cfg["use_bn"]:
            x = layers.BatchNormalization(
                name=f"{name}_bn_2")(x)  # Имя изменено на bn_2
        x = layers.Activation(cfg["activation"], name=f"{name}_act_2")(
            x)  # Имя изменено на act_2

    return x


def block_up(x, filter, factor, config, name):
    cfg = config["model"]["decoder_block"]

    if factor > 1:
        # UpSampling
        up_protected = {
            "name": f"{name}_upsample",
            "size": (factor, factor)
        }
        x = layers.UpSampling2D(**(cfg['upsample'] | up_protected))(x)

    # 2. Details
    conv_protected = {
        "filters": filter,
        "strides": 1,
        "padding": "same",
        "name": f"{name}_conv"
    }
    x = layers.Conv2D(**(cfg['base_up'] | conv_protected))(x)

    if cfg['use_bn']:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation(cfg['activation'], name=f"{name}_act")(x)

    return x


def build_vae(config, name="vae", block_down=block_down, block_up=block_up):
    input_shape = config["model"]['input_shape']
    filters = config["model"]['filters']
    factors = config["model"]['factors']
    latent_channels = config["model"]['latent_channels']
    channels_num_input = input_shape[-1]
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for n, (f, s) in enumerate(zip(filters, factors)):
        x = block_down(x, f, s, config, name=f"{name}_down_block_{n}_")

    z_mean = layers.Conv2D(
        latent_channels, 1, padding='same', name="z_mean")(x)
    z_log_var = layers.Conv2D(
        latent_channels, 1, padding='same', name="z_log_var")(x)
    encoder = keras.Model(inputs, [z_mean, z_log_var], name="encoder")
    encoder.build(input_shape)

    # latent shape
    latent_shape = encoder.output[0].shape[1:]

    # Decoder
    dec_inputs = layers.Input(shape=latent_shape)
    x = dec_inputs
    for n, (f, s) in enumerate(zip(filters[::-1], factors[::-1])):
        x = block_up(x, f, s, config, name=f"{name}_up_block_{n}_")

    outputs = layers.Conv2D(channels_num_input, 3,
                            padding='same', activation='tanh')(x)
    decoder = keras.Model(dec_inputs, outputs, name=f"{name}_decoder")
    decoder.build(latent_shape)
    return encoder, decoder


class VAE(Model):
    def __init__(self, encoder, decoder, beta=0.1, log_clipping=(-10, 10), **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = tf.Variable(beta, trainable=False, dtype=tf.float32)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.beta_tracker = keras.metrics.Mean(name="current_beta")
        self.log_clipping = log_clipping

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker, self.beta_tracker]

    def spatial_reparameterization(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        h = tf.shape(z_mean)[1]
        w = tf.shape(z_mean)[2]
        c = tf.shape(z_mean)[3]
        epsilon = tf.random.normal(shape=(batch, h, w, c))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs, training=False):
        z_mean, z_log_var = self.encoder(inputs, training=training)
        z_log_var = tf.clip_by_value(
            z_log_var, self.log_clipping[0], self.log_clipping[1])
        z = self.spatial_reparameterization([z_mean, z_log_var])
        reconstruction = self.decoder(z, training=training)

        return reconstruction, z_mean, z_log_var

    def predict_step(self, data):
        outputs = self.call(data, training=False)
        return outputs[0]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            reconstruction, z_mean, z_log_var = self.call(data, True)
            # loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=[1, 2, 3])
            )
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=[1, 2, 3]))
            total_loss = reconstruction_loss + self.beta * kl_loss

        # updating weigths
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # updating metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        self.beta_tracker.update_state(self.beta)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        reconstruction, z_mean, z_log_var = self.call(data, training=False)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(data - reconstruction), axis=[1, 2, 3])
        )
        kl_loss = -0.5 * (1 + z_log_var -
                          tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=[1, 2, 3]))
        total_loss = reconstruction_loss + self.beta * kl_loss

        # Обновляем метрики (те же трекеры, что в train_step)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.beta_tracker.update_state(self.beta)

        return {m.name: m.result() for m in self.metrics}


def get_vae_from_config(config):
    opt_config = config["training"].get("optimizer_config", None)
    if opt_config is None:
        optimizer = 'adam'
    else:
        optimizer = keras.optimizers.get(opt_config)

    encoder, decoder = build_vae(config)
    vae = VAE(encoder, decoder, beta=config["training"]["KL_beta"])
    vae.build(config["model"]['input_shape'])
    vae.compile(optimizer=optimizer)
    return vae
