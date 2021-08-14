# official documentation
# https://pytorch.org/docs/stable/torchvision/transforms.html

import torch as t
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms

class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt("data/wine/wine.csv",
                    delimiter=',',
                    dtype=np.float32,
                    skiprows=1)

        self.n_samples = xy.shape[0]
        
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.transform = transform
        
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
    
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return self.n_samples

# custom transform
class ToTensor():
    def __call__(self, sample):
        inputs, target = sample
        return t.from_numpy(inputs), t.from_numpy(target)

class MulTransform():
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
feature, labels = first_data
print(type(feature), type(labels))

composed = tv.transforms.Compose([
    ToTensor(),
    MulTransform(2)
])

dataset = WineDataset(transform=composed)
first_data = dataset[0]
feature, labels = first_data
print(type(feature), type(labels))