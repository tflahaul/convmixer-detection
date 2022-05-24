import torch

MAX_ITER = 12
BATCH_SIZE = 64
SUBDIVISIONS = 4
NUM_CLASSES = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_DIR = 'dataset'
