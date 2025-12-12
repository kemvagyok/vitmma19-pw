# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
from utils import setup_logger
from keras.models import load_model
import tensorflow as tf
import config
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
        x.set_shape([256,256,3])                  # fix input shape
        return x, y_one_hot

    dataset = dataset.map(encode_one_hot_tf)

    return dataset

def splittingDataset(dataset, batch_size, train_ratio):
    # Dataset elemszám meghatározása
    dataset_size = sum(1 for _ in dataset)

    # Shuffling
    dataset = dataset.shuffle(buffer_size=dataset_size, seed=42, reshuffle_each_iteration=False)

    # Train/val/test arányok
    train_size = int(train_ratio * dataset_size)
    val_size   = int(0.1 * dataset_size)  # tetszőlegesen 10% validáció
    test_size  = dataset_size - train_size - val_size

    # Split
    train_dataset = dataset.take(train_size)
    val_dataset   = dataset.skip(train_size).take(val_size)
    test_dataset  = dataset.skip(train_size + val_size).take(test_size)

    # Batch (testnél NEM használunk drop_remainder=True)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    val_dataset   = val_dataset.batch(batch_size, drop_remainder=False)
    test_dataset  = test_dataset.batch(batch_size, drop_remainder=False)

    return train_dataset, val_dataset, test_dataset

def evaluate():
    logger.info("Evaluating model...")

    dataset = getData(f"{config.DATA_DIR}/dataset")
    train_dataset, val_dataset, test_dataset = splittingDataset(dataset, config.BATCH_SIZE, config.TRAINING_SIZE)

    model = load_model(config.MODEL_SAVE_PATH)
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

if __name__ == "__main__":
    evaluate()
