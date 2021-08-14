from model import FeedForward
from hyperparameters import *

import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# 0) data prepration MNIST
train_dataset = tv.datasets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
test_dataset = tv.datasets.MNIST(root='./data',
                                train=False,
                                transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True)
test_loader = DataLoader(dataset=test_dataset, 
                        batch_size=BATCH_SIZE,
                        shuffle=False)

# 1) model
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model = FeedForward(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)

# 2) loss and optimizer
crit = nn.CrossEntropyLoss()
opt = t.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3) training loop
n_total_steps = len(train_loader)
for epoch in range(NUM_EPOCHS):
    for i, (images, lables) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784
        images = images.reshape(-1, 28*28).to(device)
        lables = lables.to(device)

        # forward
        outputs = model(images)
        loss = crit(outputs, lables)
        
        # backward
        opt.zero_grad()
        loss.backward()
        opt.step()

        if(i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {NUM_EPOCHS}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

# test
with t.no_grad():
    n_correct = 0
    n_samples = 0
    for images, lables in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        lables = lables.to(device)
        outputs = model(images)

        # value, index
        _, predicts = t.max(outputs, 1)
        n_samples += lables.shape[0]
        n_correct += (predicts == lables).sum().item()
    
    acc = 100.0 * n_correct/n_samples
    print(f'accuracy = {acc}')