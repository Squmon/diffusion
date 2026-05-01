import tensorflow as tf
import keras

def get_image_dataset(path, batch_size=32, image_size=(256, 256), val_split=0.2):
    # Общие параметры для обоих подмножеств
    common_params = dict(
        directory=path,
        label_mode=None,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42 # Важно для идентичного сплита
    )

    train_ds = keras.utils.image_dataset_from_directory(
        subset="training",
        validation_split=val_split,
        **common_params
    )

    val_ds = keras.utils.image_dataset_from_directory(
        subset="validation",
        validation_split=val_split,
        **common_params
    )
    train_ds = train_ds.map(lambda x: (x / 127.5) - 1.0)
    val_ds = val_ds.map(lambda x: (x / 127.5) - 1.0)

    return train_ds, val_ds

def samples_from_dataset(dataset, samples_amount=3):
    for images in dataset.take(1):
        viz_samples = images[:samples_amount]
        break
    return viz_samples.numpy()