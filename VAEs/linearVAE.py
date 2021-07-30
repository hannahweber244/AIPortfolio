import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torch.optim as optim

import random 
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s = 0
random.seed(s)
torch.manual_seed(s)
np.random.seed(s)

class linearVAE(nn.Module):

    def __init__(self, input_neurons = 64*64, latent_dim = 30):
        self.encoder = nn.Sequential(nn.Linear(input_neurons, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, 250),
                                    nn.ReLU(),
                                    nn.Linear(250, 75),
                                    nn.ReLU(),
                                    nn.Linear(75,latent_dim*2),
                                    nn.ReLU())

        self.mean = nn.Linear(latent_dim*2, latent_dim)
        self.var = nn.Linear(latent_dim*2, latent_dim)

        self.decoder = nn.Sequential(nn.Linear(latent_dim, latent_dim*2),
                                    nn.ReLU(),
                                    nn.Linear(latent_dim*2, 75),
                                    nn.ReLU(),
                                    nn.Linear(75, 250),
                                    nn.ReLU(),
                                    nn.Linear(250, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, input_neurons),
                                    nn.Sigmoid())

    def reparameterization_trick(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z 

    def forward(self, x):
        #input encoden
        x = self.encoder(x.float())

        #mittelwert, std berechnen
        mean_ = self.mean(x)
        var_ = self.var(x)
        var_ = torch.exp(0.5 * var_)
        
        #sample aus latentem raum ziehen
        sample = self.reparameterization_trick(mean_, var_)

        #basierend auf sample rekonstruktion mit Hilfe von decoder anwenden
        x = self.decoder(sample)

        return x