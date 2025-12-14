# Model training script
# This script defines the model architecture and runs the training loop.
import models 
import config
from utils import setup_logger
import tensorflow as tf
import keras
from dataUtils import split_dataset 
from metricsUtils import OrdinalDistanceLoss, OrdinalMAE
logger = setup_logger()


def getData(dataset_dir):
    dataset = tf.data.Dataset.load(dataset_dir)
    classes = ['1_Pronacio', '2_Neutralis', '3_Szupinacio']
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
        shape = [config.TARGET_IMAGE_SIZE[0], config.TARGET_IMAGE_SIZE[1], 3]
        x.set_shape(shape)                  # fix input shape
        return x, y_one_hot

    dataset = dataset.map(encode_one_hot_tf)

    return dataset

def train():
    logger.info("Starting training process...")
    logger.info(
        "Loaded configuration:\n"
        f"Epochs: {config.EPOCHS}\n"
        f"Batch size: {config.BATCH_SIZE}\n"
        f"Training size: {config.TRAINING_SIZE}\n"
        f"Learning rate: {config.LEARNING_RATE}\n"
        f"Weight decay: {config.WEIGHT_DECAY}\n"
    )    
    logger.info("Loading train, validation dataset...")
    dataset = getData(f"{config.DATA_DIR}/dataset")
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, config.BATCH_SIZE, config.TRAINING_SIZE)
    logger.info(f"{len(list(train_dataset))} samples in the train dataset.")
    logger.info(f"{len(list(val_dataset))} samples in the validation dataset.")
    # Simulation of training loop
    model = models.advancedCNNModel()
    logger.info(f"About the advanced model: {model.summary()} ")
    model.compile(
        optimizer=keras.optimizers.Adam(
            weight_decay = config.LEARNING_RATE,
            learning_rate = config.WEIGHT_DECAY
            ),
        loss = OrdinalDistanceLoss(),
        metrics = ['accuracy', OrdinalMAE()]
    )
    history = model.fit(
        train_dataset, 
        validation_data=(val_dataset), 
        epochs=config.EPOCHS)
    
    model.save(config.MODEL_SAVE_PATH)  
    logger.info("Training complete.")

if __name__ == "__main__":
    train()
