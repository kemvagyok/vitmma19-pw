# Configuration settings as an example

# Training hyperparameters
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAINING_SIZE = 0.64
WEIGHT_DECAY = 0.001
# Paths
DATA_ROOT_DIR = "/app/data"
DATA_DIR = "/app/data/anklealign"
MODEL_SAVE_PATH = "/app/model.keras"

# Image settings
TARGET_IMAGE_SIZE = (128, 128)

#Settings
WITH_OWN_LABELS = True # Whether to include own labels in preprocessing