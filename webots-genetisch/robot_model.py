import torch
import torch.nn as nn
import torch.functional as F

class PoleRobot(nn.Module):

    def __init__(self):
        super(PoleRobot, self).__init__()
        self.polePos = polePos

        self.fc1 = nn.Linear(1, 15)
        self.fc2 = nn.Linear(15, 35)
        self.fc3 = nn.Linear(35, 12)
        self.fc4 = nn.Linear(12, 4)

        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        #x = nn.Softmax(self.fc4(x), dim=1)
        return self.fc4(x)