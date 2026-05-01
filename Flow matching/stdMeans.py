import numpy as np

def cats():
    latent = np.load("latent_datasets/cats_latents_16x16x8.npy")
    return latent.std(), latent.mean()