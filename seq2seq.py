from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
embedding_dim = 200
hidden_dim = 200
epochs = 5