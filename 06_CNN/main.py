import torch as t
from torch import optim
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as transforms
from torchvision import datasets

from model import ConvNet
from hyperparameters import *

# 0) data prepration
_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

train_dataset = datasets.CIFAR10(root='./data',
                                download=True,
                                transform=_transforms)
test_dataset = datasets.CIFAR10(root='./data',
                                download=True,
                                transform=_transforms)
train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)
# 1) model
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model = ConvNet().to(device)

# 2) loss and optimizer
crit = nn.CrossEntropyLoss()
optimizer = t.optim.SGD(model.parameters(), lr=lr)

# 3) training loop
n_total_steps = len(train_dataloader)
for epoch in range(num_epochs):
    for i, (images, lables) in enumerate(train_dataloader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input_channels, 6 output_channels, 5 kernel_size
        images = images.to(device)
        labels = lables.to(device)

        # forward
        outputs = model(images)
        loss = crit(outputs, labels)

        # backward
        loss.backward()

        # update 
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('finished training!')

with t.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for iamges, lables in test_dataloader:
        images = iamges.to(device)
        lables = lables.to(device)
        outputs = model(images)

        # max returns (value, index)
        _, predicted = t.max(outputs, 1)
        n_samples += lables.size(0)
        n_correct += (predicted == lables).sum().item()

        for i in range(batch_size):
            lable = lables[i]
            pred = predicted[i]
            if (lable == pred):
                n_class_correct[lable] += 1
            n_class_samples[lable] += 1
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')