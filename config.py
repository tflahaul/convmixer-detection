import torch

MAX_ITER = 55
BATCH_SIZE = 64
SUBDIVISIONS = 16
NUM_CLASSES = 20
LEARNING_RATE = 0.001
DECAY = 5e-4
GRAD_NORM = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_DIR = 'dataset'
