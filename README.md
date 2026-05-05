# Diffusion-based Image Generation via Flow Matching in Latent Space

## Project Overview

This project implements a generative model pipeline combining Variational Autoencoders (VAE) with Flow Matching to generate images from noise. The architecture operates in two stages: (1) learning a compressed latent representation via VAE, and (2) training a neural ODE-based flow matching model to generate valid samples within the learned latent space.

## Methodology

### Stage 1: Variational Autoencoder (VAE)

The VAE component learns an efficient latent representation of images through reconstruction loss and KL divergence regularization. The encoder maps input images to a latent distribution parameterized by mean (z_mean) and log-variance (z_log_var). During inference, samples are drawn from this distribution and decoded back to the image space.

A total of 31 autoencoder architectures were explored during development, with configuration variations tested across spatial resolutions and channel depths in latent space. The optimal model was selected based on reconstruction quality and latent space coverage.

### Stage 2: Flow Matching

Flow Matching is a recent generative modeling technique that directly learns the velocity field of an optimal transport map from data to noise. Unlike traditional diffusion models, Flow Matching provides exact likelihood estimation and deterministic trajectories, making it more efficient for sampling.

The implementation uses a UNet-based velocity field predictor conditioned on time. The latent space is normalized (zero mean, unit variance) to ensure stable training and consistent velocity field magnitude across the space.

## Project Structure

```
Flow matching/
  models.py              - Core VAE architecture and utilities
  sampling.py            - Inference pipeline with command-line interface
  stdMeans.py            - Latent space normalization statistics
  
  models/
    flow_matching/       - Trained Flow Matching velocity field model
    VAE/                 - Optimal VAE configuration and weights
  
  latent_datasets/       - Pre-encoded latent representations for training
  generations/           - Generated samples and trajectories
  
  additional_scripts/
    dataset.py           - Data loading and preprocessing
    train.py             - Flow Matching model training
    main.py              - VAE training pipeline
  
  training/
    flow-matching.ipynb  - Flow Matching training experiments
    vae.ipynb            - VAE model exploration and validation
```

## Key Findings

1. **Texture vs. Object Representation**: The VAE, trained exclusively on pixel art datasets, generalizes well to arbitrary images. Analysis suggests the model learns texture and color distributions rather than object-specific features, enabling cross-domain applicability.

2. **Latent Spatial Resolution Impact**: Increasing latent spatial resolution significantly improved reconstruction quality relative to increasing channel dimension. This suggests that spatial information density is more critical than feature channel capacity for this domain.

## Dependencies

- TensorFlow
- Keras
- NumPy
- Python 3.7+

## Usage

### Loading the VAE

```python
import json
from models import get_vae_from_config

with open("models/VAE/config.json", 'r') as j:
    config = json.load(j)

vae = get_vae_from_config(config)
vae.load_weights("models/VAE/StrongSon.weights.h5")
latent_shape = vae.decoder.input_shape[1:]
```

### Encoding and Decoding

```python
# Encode image to latent distribution
reconstruction, z_mean, z_log_var = vae.call(original_image)

# Alternative: separate encoder/decoder access
latent_params = vae.encoder(original_image)  # Returns z_mean, z_log_var
reconstructed = vae.decoder(latent_samples)  # Expects batched latent tensors
```

### Flow Matching Model

```python
import tensorflow as tf
import keras

# Load pre-trained velocity field model
unet = keras.models.load_model("models/flow_matching/small_flow_cats.keras")

def sample(model, noise, num_steps=50):
    """Sample from the generative model via ODE integration."""
    dt = 1.0 / num_steps
    x = noise
    for i in range(num_steps):
        t = tf.cast(i / num_steps, tf.float32)
        t_input = tf.fill((noise.shape[0], 1), t)
        v = model({"x": x, "t": t_input}, training=False)
        x = x + v * dt
    return x

# Retrieve latent space normalization parameters
from stdMeans import cats
std, mean = cats()

# Generate samples
num_samples = 16
noise = tf.random.normal(shape=(num_samples, *latent_shape))
x = sample(unet, noise, num_steps=100) * std + mean
images = (vae.decoder(x) + 1.0) / 2.0
```

### Quick Start: Command-line Interface

Generate 20 images:
```bash
python sampling.py --num 20 --steps 50 --dest generations
```

Generate 4 videos showing the generation trajectory from noise to image:
```bash
python sampling.py --num 4 --steps 50 --trajectory True --dest generations
```

The `--steps` parameter controls ODE integration granularity. Values above 50 provide diminishing returns in sample quality while increasing computation time.

## Results

Generated samples are saved to the `generations/` directory. Historical experiments and ablation studies are archived in the `old/` directory, including alternate VAE configurations and model checkpoints from various training runs.