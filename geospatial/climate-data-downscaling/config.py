import torch

DATA_DIR = "/beegfs/halder/DATA/climate_data_(kaushik)/"
CHECKPOINT_DIR = "/beegfs/halder/GITHUB/RESEARCH/climate-data-downscaling/checkpoint"
LOG_DIR = "/beegfs/halder/GITHUB/RESEARCH/climate-data-downscaling/log"
LOAD_MODEL = True
SAVE_MODEL = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
BATCH_SIZE = 32
NUM_WORKERS = 2
HIGH_RES = 512
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 17