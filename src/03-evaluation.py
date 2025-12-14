# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
from utils import setup_logger
from models import baseline_model
import config
from dataUtils import split_dataset
from metricsUtils import Evaluator
from metricsUtils import OrdinalDistanceLoss, OrdinalMAE

import numpy as np
from keras.models import load_model
import tensorflow as tf


logger = setup_logger()

def getBaselinePredictions(dataset):
    baselineModel = baseline_model()
    y_true = []
    baselinePredictions = []
    for x, y in dataset:
        for index in range(len(x)):        
            baselinePredictions.append(baselineModel())
            y_true.append(y[index])
    y_true = np.array(y_true)
    np_baselinePredictions = np.array(baselinePredictions, dtype=np.float32)
    return y_true, np_baselinePredictions

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
        x.set_shape(shape)                 # fix input shape
        return x, y_one_hot

    dataset = dataset.map(encode_one_hot_tf)

    return dataset

def evaluate():    
    
    logger.info("\nEvaluating models...")

    logger.info("Loading test dataset...")
    dataset = getData(f"{config.DATA_DIR}/dataset")
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, config.BATCH_SIZE, config.TRAINING_SIZE)
    logger.info("Loaded...")

    logger.info("Evaluating baselinemodel...")
    y_true, baselinePredictions = getBaselinePredictions(test_dataset)
    evaluator = Evaluator()
    evaluator = evaluator.evaluate(np.argmax(y_true, axis=1), baselinePredictions)
    logger.info(f"Test baselinemodel accuracy: {evaluator['accuracy']}")
    logger.info(f"Test baselinemodel ordinal_mae: {evaluator['ordinal_mae']}")


    logger.info("Evaluating advanced model...")
    model = load_model(config.MODEL_SAVE_PATH)
    model = load_model(
        config.MODEL_SAVE_PATH,
        custom_objects={
            "OrdinalDistanceLoss": OrdinalDistanceLoss
        }
    )
    logger.info("Advanced model loaded successfully.")
    logger.info(f"{len(list(test_dataset))} samples in the test dataset.")
    loss, acc, ordinal_mae = model.evaluate(test_dataset)
    logger.info(f"Test advanced model accuracy: {acc}")
    logger.info(f"Test advanced model ordinal MAE: {ordinal_mae}")
    logger.info("Evaluating ended")
if __name__ == "__main__":
    evaluate()
