# src/config.py

# ---------------- SISA CONFIG ----------------

S = 5          # number of shards
R = 10         # number of slices per shard


# ---------------- TRAINING CONFIG ----------------

BATCH_SIZE = 64

EPOCHS_PER_SLICE = 1

LEARNING_RATE = 0.001


# ---------------- PATHS ----------------

DATA_DIR = "./data"

BASE_MODEL_DIR = "./models"

UNLEARN_SHARDED_MODEL_DIR = "./unlearn_sharded_models"


# ---------------- DEVICE ----------------

DEVICE = "cuda"