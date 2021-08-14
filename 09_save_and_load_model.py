import torch as t
import torch.nn as nn
import torchvision as tv

# some model to test
class FeedFowrad(nn.Module):
    def __init__(self, in_features, hidden_sizes, num_classes):
        super(FeedFowrad, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_sizes)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_sizes, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

model = FeedFowrad(in_features=2,
                    hidden_sizes=3,
                    num_classes=2)

FILE = 'testModel.pth'
# save the model
t.save(model.state_dict(), FILE)

print('----- original model -----')
for p in model.parameters():
    print(p)

# load the model
loaded_model = FeedFowrad(2, 3, 2)
loaded_model.load_state_dict(t.load(FILE))
loaded_model.eval()

print('----- laoded model -----')
for p in loaded_model.parameters():
    print(p)