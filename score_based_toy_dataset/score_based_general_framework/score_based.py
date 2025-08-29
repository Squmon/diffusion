import numpy as np
import tensorflow as tf
import keras


class forward_process:
    def sample_weight(self, t):
        raise NotImplementedError

    def forward(self, t, x0):  # возвращает зашумленную версию и скор функцию
        raise NotImplementedError

    def map_function(self, x, y=None, w=None):
        t = tf.random.uniform([], 0, 1)
        noised, score_function = self.forward(x, t)
        if w is None:
            w = self.sample_weight(t)
        else:
            w = self.sample_weight(t) * w
        if y is None:
            return (t, noised), score_function, w
        else:
            return (t, y, noised), score_function, w


class VP_SDE(forward_process):
    def __init__(self, betas=None):
        super().__init__()
        if betas is None:
            betas = tf.linspace(0.001, 0.02, 100)
        self.betas = betas
        self.num_steps = self.betas.shape[0]
        self.alphas = 1.0 - self.betas
        self.alphas_bar = tf.math.cumprod(self.alphas)

    def forward(self, t, x0):
        index = tf.math.floor(t * (self.num_steps - 1))
        noise = tf.random.normal(tf.shape(x0))
        noised = x0 * tf.sqrt(self.alphas_bar[index]) + noise * tf.sqrt(
            1 - self.alphas_bar[index]
        )
        score_function = noise / tf.sqrt(1 - self.alphas_bar[index])
        return noised, score_function

def langevin_sampling(x_shape, score_function, steps):
    T = tf.linspace(1, 0, steps)
    dt = 1/steps
    x = tf.random.uniform(x_shape)
    for t in T:
        x = x + dt*score_function(x, t) + tf.sqrt(2*dt) * tf.random.uniform(x_shape)