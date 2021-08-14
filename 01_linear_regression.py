import torch as t
from torch import optim
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) data prepration
X_numpy, y_numpy = datasets.make_regression(n_samples=100,
                                            n_features=1,
                                            noise=20,
                                            random_state=1)

X = t.from_numpy(X_numpy.astype(np.float32))
y = t.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) Model
in_size = n_features
out_size = 1
model = nn.Linear(in_size, out_size)

# 2) loss and optimizer
learning_rate = .01
crit = nn.MSELoss()
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = crit(y_predicted, y)

    # backward
    loss.backward()

    # update
    optimizer.step()

    # empty our gradient
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')

# plot the result
prediction = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro', label='real')
plt.plot(X_numpy, prediction, 'b', label='predicted')
plt.legend(loc='upper left')
plt.show()