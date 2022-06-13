import torch

MAX_ITER = 55 # tbd
BATCH_SIZE = 64
SUBDIVISIONS = 8
NUM_CLASSES = 20
LEARNING_RATE = 0.001
DECAY = 2e-5
GRAD_NORM = 1.0
IMG_SIZE = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_DIR = 'dataset'
