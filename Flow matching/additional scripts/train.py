import argparse
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import json
import datetime
import tensorflow as tf
from models import get_vae_from_config
import keras
from dataset import get_image_dataset, samples_from_dataset


class CyclicalBetaScheduler(keras.callbacks.Callback):
    def __init__(self, target_beta=0.01, n_cycles=4, ratio=0.5):
        super().__init__()
        self.target_beta = target_beta
        self.n_cycles = n_cycles
        self.ratio = ratio  # half plato half increasing

    def on_epoch_begin(self, epoch, logs=None):
        total_epochs = self.params['epochs']
        period = total_epochs / self.n_cycles
        relative_pos = (epoch % period) / period

        if relative_pos <= self.ratio:
            new_beta = self.target_beta * (relative_pos / self.ratio)
        else:
            new_beta = self.target_beta

        self.model.beta.assign(new_beta)


class VisualizerCallback(keras.callbacks.Callback):
    def __init__(self, sample_data, log_dir, every = 10):
        super().__init__()
        self.sample_data = (sample_data + 1.0)/2.0
        self.file_writer = tf.summary.create_file_writer(log_dir + "/images")
        self.n = len(sample_data)
        self.every = every

    def on_train_begin(self, logs=None):
        with self.file_writer.as_default():
            tf.summary.image("Input image", self.sample_data, step=0)

    def on_epoch_end(self, epoch, logs=None):
        latent_shape = self.model.decoder.input_shape[1:]
        if epoch % self.every == 0:
            reconstruction = (self.model.predict(self.sample_data) + 1.0)/2.0
            noise = tf.random.normal(shape=(self.n, *latent_shape))
            samples = self.model.decoder(noise)

            with self.file_writer.as_default():
                tf.summary.image("Reconstruction", reconstruction, step=epoch)
                tf.summary.image("Samples", samples, step=epoch)


def main():
    parser = argparse.ArgumentParser(description="VAE training")
    parser.add_argument("--config-path", type=str, default=None, help="Path to json file")
    parser.add_argument("--config", type=str, default=None, help="JSON string")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=50, help="patience")
    parser.add_argument("--save-best", type=int, default=True, help="best only")
    parser.add_argument("--samples", type=int, default=10, help="samplesF")
    

    args = parser.parse_args() 
    if args.config_path is None: 
        if args.config is None:
            raise Exception("no config specified")
        else:
            config = json.loads(args.config)
    else:
        with open(args.config_path, 'r') as j:
            config = json.load(j)

    name = f"{config['name']}_input{config["model"]['input_shape']}_latent{config["model"]["latent_channels"]}_filters{config["model"]["filters"]}_beta{config["training"]["KL_beta"]}"
    log_dir = config["log_dir"] + name + "/" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    check_points = log_dir + "/checkpoints/"

    train_ds, val_ds = get_image_dataset(config['training']["image_path"], config['training']["batch_size"], (config['model']["input_shape"][0], config['model']["input_shape"][1]))
    samples = samples_from_dataset(val_ds)

    visualizer_callback = VisualizerCallback(samples, log_dir, args.samples)

    b_callback = CyclicalBetaScheduler(
        target_beta=config["training"]["KL_beta"], n_cycles=4, ratio=0.5)

    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
    )

    early_stop = EarlyStopping(
        monitor='rec_loss',
        patience=args.patience,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=check_points + name + "_" +
        '{epoch:02d}_{rec_loss:.4f}.weights.h5',
        save_weights_only=True,
        monitor='rec_loss',
        mode='min',
        save_best_only=args.save_best
    )

    vae = get_vae_from_config(config)
    vae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[tensorboard_callback, visualizer_callback,
                   early_stop, checkpoint_callback, b_callback]
    )


if __name__ == "__main__":
    main()