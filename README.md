# torchplus

## 1. Implements the + operator on PyTorch modules, returning sequences.
## 2. Sequence step layers with no args don't require parentheses.

### Original model:

```py
from torch import nn

class OrigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```

### Rewritten using torchplus:

```py
from torchplus import nn

class TorchPlusModel(nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv2d(1, 10, kernel_size=5) + nn.MaxPool2d(2) + nn.ReLU
        conv2 = nn.Conv2d(10, 20, kernel_size=5) + nn.Dropout2d() + \
                nn.MaxPool2d(2) + nn.ReLU
        fc1 = nn.Linear(320, 50) + nn.ReLU + nn.Dropout
        fc2 = nn.Linear(50, 10) + nn.LogSoftmax(1)
        self.seq = conv1 + conv2 + nn.Flatten + fc1 + fc2

    def forward(self, x):
        return self.seq(x)
```
