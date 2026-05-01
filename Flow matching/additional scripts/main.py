import argparse
import json
import tensorflow as tf
from models import build_vae, VAE
from dataset import get_image_dataset, samples_from_dataset
import keras
import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.preprocessing.image import array_to_img

def main():
    parser = argparse.ArgumentParser(description="VAE running")
    parser.add_argument("--mode", type=str, default=None, help="sample/encode")
    parser.add_argument("--folder", type=str, default=None, help="Folder with config and weights")
    parser.add_argument("--image", type=str, default=None, help="Target image for encoding")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="./output", help="Output directory for saving results")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory with images for encoding")
    parser.add_argument("--decode_image", type=str, default=None, help="Path to an image to decode")

    args = parser.parse_args()

    if args.folder is None:
        raise Exception("No folder specified for config and weights")

    config_path = os.path.join(args.folder, "config.json")
    weights_path = None

    for file in os.listdir(args.folder):
        if file.endswith(".h5"):
            weights_path = os.path.join(args.folder, file)
            break

    if not os.path.exists(config_path) or weights_path is None:
        raise Exception("Config or weights not found in the specified folder")

    with open(config_path, 'r') as j:
        config = json.load(j)

    encoder, decoder = build_vae(config)
    vae = VAE(encoder, decoder)
    vae.build(config['model']['input_shape'])
    vae.load_weights(weights_path)

    # Automatically retrieve latent shape from decoder
    latent_shape = vae.decoder.input_shape[1:]  # Exclude batch dimension
    print("Expected latent shape for decoder:", latent_shape)

    os.makedirs(args.output, exist_ok=True)

    if args.mode == "sample":
        z = tf.random.normal(shape=(args.samples, *latent_shape))
        generated_images = vae.decoder(z)
        print("Generated samples:")
        for i, img in enumerate(generated_images):
            img_path = os.path.join(args.output, f"sample_{i}.png")
            tf.keras.preprocessing.image.save_img(img_path, (img + 1.0) * 127.5)
        np.save(os.path.join(args.output, "sample_latents.npy"), z.numpy())

    elif args.mode == "encode":
        if args.input_dir is None:
            raise Exception("No input directory specified for encoding")

        encoded_results = {}
        for file_name in os.listdir(args.input_dir):
            file_path = os.path.join(args.input_dir, file_name)
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img = load_img(file_path, target_size=(config['model']['input_shape'], config['model']['input_shape']))
            img_array = img_to_array(img) / 127.5 - 1.0
            img_array = tf.expand_dims(img_array, axis=0)

            z_mean, z_log_var, z = vae.encode(img_array)
            encoded_results[file_name] = {
                "z_mean": z_mean.numpy(),
                "z_log_var": z_log_var.numpy(),
                "z": z.numpy()
            }

            output_img_path = os.path.join(args.output, file_name)
            array_to_img((vae.decode(z)[0] + 1.0) * 127.5).save(output_img_path)

        np.save(os.path.join(args.output, "encoded_results.npy"), encoded_results)

    elif args.mode == "decode_image":
        if args.decode_image is None:
            raise Exception("No image specified for decoding")

        img = load_img(args.decode_image, target_size=(config['model']['input_shape'][0], config['model']['input_shape'][1]))
        img_array = img_to_array(img) / 127.5 - 1.0
        img_array = tf.expand_dims(img_array, axis=0)

        reconstruction, z_mean, z_log_var = vae.call(img_array)

        output_img_path = os.path.join(args.output, "decoded_image.png")
        array_to_img((reconstruction[0] + 1.0) * 127.5).save(output_img_path)
        print(f"Decoded image saved to {output_img_path}")

    else:
        raise Exception("Invalid mode. Use 'sample' or 'encode'")

if __name__ == "__main__":
    main()