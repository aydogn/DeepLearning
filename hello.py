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

x = torch.tensor([1,2,3,4], dtype = torch.float32)
y = torch.tensor([2,4,6,8], dtype = torch.float32)

w = torch.tensor(0.0, requires_grad=True)

def forward(x):
    return w * x

def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

n_iter = 100
l_r = 0.01


for epoch in range(n_iter):
    y_pred = forward(x)
    l = loss(y, y_pred)    
    l.backward()
    with torch.no_grad():
        w -= l_r * w.grad
    w.grad.zero_()
    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
print(f'prediction after training: f(5) = {forward(5):.3f}')
