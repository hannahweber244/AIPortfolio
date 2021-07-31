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
    #modellklasse definieren für einen linearen VAE --> ist als simples gegenbeispiel zum conv VAE gedacht
    def __init__(self, input_neurons = 64*64, latent_dim = 30, sigmoid = False):
        #input_neurons ist 64*64 wenn geflattetes 64x64 Pixelbild eingelsen wird --> Pixel characters 
        #encoder und decoder werden modular in sequentials aufgebaut --> so kann decoder auch von außerhalb der klasse zur 
        #generierung neuer bilder, basierend auf zufallsvektor, auch einfach aufgerufen werden

        super(linearVAE, self).__init__()

        #encoder als sequential defineiren
        self.encoder = nn.Sequential(nn.Linear(input_neurons, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, 250),
                                    nn.ReLU(),
                                    nn.Linear(250, 75),
                                    nn.ReLU(),
                                    nn.Linear(75,latent_dim*2),
                                    nn.ReLU())

        #layer für mean und var definieren
        self.mean = nn.Linear(latent_dim*2, latent_dim)
        self.var = nn.Linear(latent_dim*2, latent_dim)
        if sigmoid:#auf letztem layer soll eine sigmoid funktion angewendet werden
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
        else:#auf letztem layer wird keine sigmoid funktion angewendet, aber eine relu, um pixel dann wenigstens positiv zu haben
            self.decoder = nn.Sequential(nn.Linear(latent_dim, latent_dim*2),
                                    nn.ReLU(),
                                    nn.Linear(latent_dim*2, 75),
                                    nn.ReLU(),
                                    nn.Linear(75, 250),
                                    nn.ReLU(),
                                    nn.Linear(250, 1000),
                                    nn.ReLU(),
                                    nn.Linear(1000, input_neurons),
                                    nn.ReLU())

    def reparameterization_trick(self, mean, var):
        #trick um mit sampling backpropagaieren u können --> fehler rückeärts rechnen können
        epsilon = torch.randn_like(var).to(device)      # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick --> ziehen einer zufalls stichprobe
        return z 

    def forward(self, x):
        #input encoden
        x = self.encoder(x.float())

        #mittelwert, std berechnen
        mean_ = self.mean(x)
        var_ = self.var(x)#ausgabe der log varianz
        var_ = torch.exp(0.5 * var_)#umrechnen der varianz nach bekannter formel
        
        #sample aus latentem raum ziehen
        sample = self.reparameterization_trick(mean_, var_)

        #basierend auf sample rekonstruktion mit Hilfe von decoder anwenden
        x = self.decoder(sample)

        return x