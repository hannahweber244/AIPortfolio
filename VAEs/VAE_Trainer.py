import torch.optim as optim
import torch
import torch.nn as nn

import random 
import numpy as np

s = 0
random.seed(s)
torch.manual_seed(s)
np.random.seed(s)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAETrainer:

    def __init__(self, train_batches):
        self.train_batches = train_batches

    def combined_loss(self, loss, mu, logvar):
        '''
        Funktion kombiniert den Rekonstruktionsloss (BCE/MSE etc.)
        mit Nebenbedingung der KL Divergenz, um eine Normalverteilung
        im latenten Raum zu erzwingen
        '''
        LOSS = loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return LOSS+KLD

    def train(self, model, epochs = 100, lr = 0.0001):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()#weil keine Sigmoid auf letztem Layer
        model.train()
        for epoch in range(0,epochs):
            print('epoche:', epoch+1)
            loss_ = 0
            for batch_id, batch in enumerate(self.train_batches):
                optimizer.zero_grad()
                batch = batch.to(device)
                image, mu, std = model(batch.float())

                loss = criterion(image, batch.float())
                loss = self.combined_loss(loss,mu,std)

                loss_ += loss.item()

                loss.backward()
                optimizer.step()
        return model
            #if (epoch+1)%10 == 0:
            #    cv2_imshow(cv2.resize(batch[0].permute(1,2,0).cpu().detach().numpy()*255, (200,200)))
            #    cv2_imshow(cv2.resize(image[0].permute(1,2,0).cpu().detach().numpy()*255, (200,200)))