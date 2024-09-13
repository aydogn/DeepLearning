"""import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_epochs =4
batch_size = 4
learning_rate = 0.001


transform = transform.Compose()
"""
import torch 
x = torch.rand(2,3)
y = x.view(-1,2)
print(type(x.size()))
print(type(x.shape))
print(y)