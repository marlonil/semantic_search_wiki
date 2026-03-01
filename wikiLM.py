import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim as optim
import torch.utils.data as data
import math
import copy

class WikiDataset(Dataset):
    pass

class MultiHeadAttention(nn.Module):
    pass

class PositionWiseFeedForward(nn.Module):
    pass

class PositionalEncoding(nn.Module):
    pass

class DecoderLayer(nn.Module):
    # In our case we just use the decoder only. We want a model that can generate token by token.
    # It means the modell predicts the next token based on the current and previous tokens.
    # Therefore we use the Multi-head attention. We ensure that the each token only attends to the previous token.

    pass

class Transformer(nn.Module):
    pass

# training loop down here