from scipy.sparse import data
import torch as t
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) data prepration
bc_dataset = datasets.load_breast_cancer()
X, y = bc_dataset.data, bc_dataset.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.2,
                                                    random_state=123)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = t.from_numpy(X_train.astype(np.float32))
X_test = t.from_numpy(X_test.astype(np.float32))
y_train = t.from_numpy(y_train.astype(np.float32))
y_test = t.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = t.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate = .01
crit = nn.BCELoss()
optim = t.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward path and loss
    y_predicted = model(X_train)
    loss = crit(y_predicted, y_train)

    # backward path
    loss.backward()

    # updates
    optim.step()

    # make gradient empty again
    optim.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss: {loss.item():.4f}")

with t.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy = {acc:.4f}")