import torch.nn as nn
import torch.nn.functional as F


class UKR_LETTERS_NET(nn.Module):
    def __init__(self):
        super(UKR_LETTERS_NET, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(20*6*6, 50)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, 32)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 36 -> 16
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # 16 -> 6
        x = self.flatten(x)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x
