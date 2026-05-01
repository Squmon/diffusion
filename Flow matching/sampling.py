import keras
import tensorflow as tf
import datetime
from stdMeans import cats
from models import get_vae_from_config
from keras.preprocessing.image import array_to_img
import numpy as np
import json
import argparse
import os


parser = argparse.ArgumentParser(description="Flow sampling")
parser.add_argument("--steps", type=str, default=200, help="Folder with config and weights")
parser.add_argument("--seed", type=str, default=None, help="Folder with config and weights")
parser.add_argument("--num", type=str, default=10, help="Folder with config and weights")
parser.add_argument("--dest", type=str, default="/output", help="Folder with config and weights")
parser.add_argument("--trajectory", type=str, default='false', help="Folder with config and weights") # если True, то нужно сохранить в gif формате анимацию

with open("models/VAE/config.json", 'r') as j:
    config = json.load(j)

unet = keras.models.load_model("models/flow matching/small_flow_cats.keras")
vae = get_vae_from_config(config)
vae.load_weights("models/VAE/StrongSon.weights.h5")
time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
latent_shape = vae.decoder.input_shape[1:]

std, mean = cats()

def sample(model, noise, num_steps=50):
    dt = 1.0 / num_steps
    x = noise
    for i in range(num_steps):
        t = tf.cast(i / num_steps, tf.float32)
        t_input = tf.fill((noise.shape[0], 1), t)
        v = model({"x": x, "t": t_input}, training=False)
        x = x + v * dt

    return x

def sample_and_save_states(model, noise, num_steps=50):
    dt = 1.0 / num_steps
    states = []
    x = noise
    for i in range(num_steps):
        t = tf.cast(i / num_steps, tf.float32)
        t_input = tf.fill((noise.shape[0], 1), t)
        v = model({"x": x, "t": t_input}, training=False)
        x = x + v * dt
        states.append(np.copy(x.numpy())[0])
    return np.array(states)


args = parser.parse_args()

# Установка случайного сидирования для воспроизводимости
if args.seed is not None:
    tf.random.set_seed(int(args.seed))

# Проверка и создание директории для сохранения результатов
os.makedirs(args.dest, exist_ok=True)

# Функция для сохранения анимации (GIF)
if args.trajectory.lower() == "true":
    import imageio

def save_trajectory_as_gif(frames, path):
    imageio.mimsave(path, frames, fps=10, loop=0)

# Основной функционал в зависимости от режима
num_samples = int(args.num)
num_steps = int(args.steps)


noise = tf.random.normal(shape=(num_samples, *latent_shape))
if args.trajectory.lower() == "true":
    for i, n in enumerate(noise):
        x = sample_and_save_states(unet, n[np.newaxis, :], num_steps=num_steps)* std + mean
        f = (vae.decoder(x) + 1.0) / 2.0
        frames = []
        for a in f:
            frames.append(array_to_img(a * 255))
        save_trajectory_as_gif(frames, os.path.join(args.dest, f"{time}_trajectory_{i}.gif"))

else:
    x = sample(unet, noise, num_steps=num_steps) * std + mean
    samples = (vae.decoder(x) + 1.0) / 2.0
    for n, s in enumerate(samples):
        img = array_to_img(s * 255)
        img.save(os.path.join(args.dest, f"{time}_{n}.png"))