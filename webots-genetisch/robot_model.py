import torch
import torch.nn as nn
import torch.nn.functional as F

class PoleRobot(nn.Module):

    def __init__(self):
        super(PoleRobot, self).__init__()

        #linear layer definieren
        self.fc1 = nn.Linear(1, 15)#eingabe 1, da vorhersage basierend auf einem wert --> pole position gemacht wird
        #folgende linear layer enthalten weniger neuronen, um rechenlast gering zu halten
        self.fc2 = nn.Linear(15, 35)
        self.fc3 = nn.Linear(35, 12)
        #ausgabe von einem wert --> wird dann auf alle räder applied
        self.fc4 = nn.Linear(12, 1)

    def forward(self, x):
        #definieren des forward pass (hinzufügen von aktivierungsfunktionen)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return torch.tanh(self.fc4(x))# normiert auf bereich [-1,1]