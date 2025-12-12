# Model training script
# This script defines the model architecture and runs the training loop.
import models 
import config
from utils import setup_logger
import tensorflow as tf

logger = setup_logger()


def getData(dataset_dir):
    dataset = tf.data.Dataset.load(dataset_dir)
    classes = ['1_Pronacio', '2_Neutralis', '3_Szupinacio']
    class_to_index = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    # string -> index TF-ben
    lookup_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(classes),
            values=tf.constant(list(range(num_classes)), dtype=tf.int32),
        ),
        default_value=-1
    )

    def encode_one_hot_tf(x, y):
        y_idx = lookup_table.lookup(y)            # tf.string -> int
        y_one_hot = tf.one_hot(y_idx, depth=num_classes)
        x.set_shape([256,256,3])                  # fix input shape
        return x, y_one_hot

    dataset = dataset.map(encode_one_hot_tf)

    return dataset

def splittingDataset(dataset, batch_size, train_size):
    dataset_size = len(dataset)
    dataset = dataset.shuffle(buffer_size=dataset_size, seed=42)

    train_size = int(train_size * dataset_size)
    val_size = int((1 - train_size) * dataset_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    val_dataset   = val_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

    return train_dataset, val_dataset

def train():
    logger.info("Starting training process...")
    logger.info(f"Loaded configuration. Epochs: {config.EPOCHS}")
    
    dataset = getData(f"{config.DATA_DIR}/dataset")
    train_dataset, val_dataset = splittingDataset(dataset, config.BATCH_SIZE, config.TRAINING_SIZE)
    
    # Simulation of training loop
    model = models.SimpleCNNModel()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_dataset, 
        validation_data=(val_dataset), 
        epochs=config.EPOCHS)
    
    model.save(config.MODEL_SAVE_PATH)  
    logger.info("Training complete.")

if __name__ == "__main__":
    train()
