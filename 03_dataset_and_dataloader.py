import torch as t
import torchvision as tv
from torch.utils.data import Dataset, DataLoader, dataloader, dataset

import os
import numpy as np

class WineDataset(Dataset):
    def __init__(self):
        # load data
        xy = np.loadtxt('./data/wine/wine.csv',
                        delimiter=',',
                        dtype=np.float32,
                        skiprows=1)
        self.x = t.from_numpy(xy[:, 1:])
        self.y = t.from_numpy(xy[:, [0]])  # n_samples, 1
        self.n_samples = xy.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

if __name__ == '__main__':
    db = WineDataset()
    db_loader = DataLoader(dataset=db,
                            batch_size=4,
                            shuffle=True,
                            num_workers=2)
    data_iter = iter(db_loader)
    # data = data_iter.next()
    # features, labels = data
    # print(features, labels)

    # trainig loop
    num_epochs = 2
    total_samples = len(db)
    n_iterations = np.ceil(total_samples/4)
    print(total_samples, n_iterations)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(db_loader):
            # forward, backward, update
            if (i+1) % 5 == 0:
                print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}")