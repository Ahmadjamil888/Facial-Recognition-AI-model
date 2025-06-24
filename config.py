import torch
import os

# -------------------------
# Device and Seed
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# -------------------------
# Dataset Configuration
# -------------------------
NUM_CLASSES = 7  # Change based on number of expression classes

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"
TRAIN_IMG_DIR = "data/raw/train"
VAL_IMG_DIR = "data/raw/test"

# -------------------------
# Training Hyperparameters
# -------------------------
BATCH_SIZE = 8
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# -------------------------
# Checkpoint Paths
# -------------------------
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# -------------------------
# Output (e.g. Confusion Matrix)
# -------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
