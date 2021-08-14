import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from hyperparameters import *
from model import NeuralNet

writer = SummaryWriter('runs/mnist')

# 0) data prepration: MNIST dataset
train_dataset = tv.datasets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = tv.datasets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)
train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)

# simple demo
examples = iter(test_dataloader)
example_data, example_targets = examples.next()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()
img_grid = tv.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)
writer.close()

# 1) model
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model = NeuralNet(in_size, hidden_size, num_classes).to(device)

# 2) loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

# add loss function to graph
writer.add_graph(model, example_data.reshape(-1, 28*28).to(device))
writer.close()

# 3) training loop
n_total_steps = len(train_dataloader)
running_loss = 0.0
running_correct = 0.0
for epoch in range(num_epoches):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward  
        loss.backward()

        # update
        optimizer.zero_grad()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = t.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epoches}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct/ 100, epoch * n_total_steps + i)
            running_loss = 0
            running_correct = 0

_labels = list()
preds = list()
with t.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_dataloader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = t.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    
        _labels.append(predicted)
        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        preds.append(class_predictions)
    
    _labels = t.cat(_labels)
    preds = t.cat([t.stack(batch) for batch in preds])

    acc = 100.0 * n_correct/n_samples
    print(f'accuracy of the network on the test iamges: {acc} %')

    classes = range(10)
    for i in classes:
        labels_i = _labels == i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()