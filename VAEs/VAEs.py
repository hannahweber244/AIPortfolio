import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random 
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s = 0
random.seed(s)
torch.manual_seed(s)
np.random.seed(s)

class VAE_1Conv(nn.Module):

    def __init__(self, sigmoid = False):
        super(VAE_1Conv, self).__init__()

        #True/False Wert ob Sigmoid auf letztes Layer
        self.sigmoid = sigmoid
        self.encoder = nn.Sequential(nn.Conv2d(3, 12, kernel_size=4, stride=2),
                                     nn.ReLU())#31x31 pixel bild

        self.encode = nn.Linear(12*31*31, 48)
        self.decode = nn.Linear(24,12*31*31)

        #decoder convolutions 
        self.decode_conv = nn.ConvTranspose2d(12, 3, kernel_size=4, stride=2)
        
        
        ## latenter raum = 24
        self.mean = nn.Linear(48, 24)
        self.var = nn.Linear(48, 24)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z     

    def decoder(self, x):
        batch_size = x.shape[0]
        if self.sigmoid:
            x = self.decode(x)
            x = torch.reshape(x, (batch_size, 12,31,31))
            x = self.decode_conv(x)
            return F.sigmoid(x)
        else:
            x = self.decode(x)
            x = torch.reshape(x, (batch_size, 12,31,31))
            x = self.decode_conv(x)
            return F.relu(x)


    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim = 1)# 64 x ... großen vektor (64, 12*31*31) shape
        x = self.encode(x)

        mean_ = self.mean(x)
        var_ = self.var(x)
        var_ = torch.exp(0.5 * var_)

        sample = self.reparameterization(mean_, var_)
        reconstructed = self.decoder(sample)
        return reconstructed, mean_, var_

class VAE_2Conv(nn.Module):

    def __init__(self, sigmoid = False):
        super(VAE_2Conv, self).__init__()

        #True/False Wert ob Sigmoid auf letztes Layer
        self.sigmoid = sigmoid
        self.encoder = nn.Sequential(nn.Conv2d(3, 12, kernel_size=4, stride=2),#31x31 pixel bild
                                     nn.ReLU(),
                                     nn.Conv2d(12, 48, kernel_size=3, stride=2), #15x15
                                     nn.ReLU())

        self.encode = nn.Linear(48*15*15, 48)
        self.decode = nn.Linear(24,48*15*15)

        #decoder convolutions 
        self.decode_conv = nn.Sequential(nn.ConvTranspose2d(48, 12, kernel_size=3, stride=2),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(12, 3, kernel_size=4, stride=2))
        
        
        ## latenter raum = 24
        self.mean = nn.Linear(48, 24)
        self.var = nn.Linear(48, 24)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z     

    def decoder(self, x):
        batch_size = x.shape[0]
        if self.sigmoid:
            x = self.decode(x)
            x = torch.reshape(x, (batch_size, 48,15,15))
            x = self.decode_conv(x)
            return F.sigmoid(x)
        else:
            x = self.decode(x)
            x = torch.reshape(x, (batch_size, 48,15,15))
            x = self.decode_conv(x)
            return F.relu(x)


    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim = 1)# 64 x ... großen vektor (64, 12*31*31) shape
        x = self.encode(x)

        mean_ = self.mean(x)
        var_ = self.var(x)
        var_ = torch.exp(0.5 * var_)

        sample = self.reparameterization(mean_, var_)
        reconstructed = self.decoder(sample)
        return reconstructed, mean_, var_

class VAE_3Conv(nn.Module):

    def __init__(self, sigmoid = False):
        super(VAE_3Conv, self).__init__()

        #True/False Wert ob Sigmoid auf letztes Layer
        self.sigmoid = sigmoid
        self.encoder = nn.Sequential(nn.Conv2d(3, 12, kernel_size=4, stride=2),#31x31 pixel bild
                                     nn.ReLU(),
                                     nn.Conv2d(12, 48, kernel_size=3, stride=2), #15x15
                                     nn.ReLU(),
                                     nn.Conv2d(48, 64, kernel_size=3,stride=1),#13x13
                                     nn.ReLU())

        self.encode = nn.Linear(64*13*13, 48)
        self.decode = nn.Linear(24,64*13*13)

        #decoder convolutions 
        self.decode_conv = nn.Sequential(nn.ConvTranspose2d(64, 48, kernel_size=3, stride=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(48, 12, kernel_size=3, stride=2),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(12,3, kernel_size=4, stride=2))
        
        
        ## latenter raum = 24
        self.mean = nn.Linear(48, 24)
        self.var = nn.Linear(48, 24)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z     

    def decoder(self, x):
        batch_size = x.shape[0]
        if self.sigmoid:
            x = self.decode(x)
            x = torch.reshape(x, (batch_size, 64,13,13))
            x = self.decode_conv(x)
            return F.sigmoid(x)
        else:
            x = self.decode(x)
            x = torch.reshape(x, (batch_size, 64,13,13))
            x = self.decode_conv(x)
            return F.relu(x)


    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim = 1)# 64 x ... großen vektor (64, 12*31*31) shape
        x = self.encode(x)

        mean_ = self.mean(x)
        var_ = self.var(x)
        var_ = torch.exp(0.5 * var_)

        sample = self.reparameterization(mean_, var_)
        reconstructed = self.decoder(sample)
        return reconstructed, mean_, var_
