import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import cv2
import copy

import torch
torch.cuda.empty_cache()


seed = 100
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

from tqdm.notebook import tqdm
 
#h,w = 64,64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAEsmall_color(nn.Module):

    def __init__(self, emb_size = 30, sigmoid=False):
        super(VAEsmall_color, self).__init__()
        self.sigmoid = sigmoid

        #batch normalisierungs layer
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.batch_norm2 = nn.BatchNorm2d(24)
        self.batch_norm3 = nn.BatchNorm2d(32)

        #dropout layer
        self.dropout1 = nn.Dropout2d()
        self.dropout2 = nn.Dropout()

        #Encode-Convolutional-Layer
        self.cnn1 = nn.Conv2d(3, 8, kernel_size=4, stride=2)#31x31
        self.cnn2 = nn.Conv2d(8, 15, kernel_size = 3, stride = 1)#29x29
        self.cnn3 = nn.Conv2d(15, 24, kernel_size=3, stride=2)#14x14
        self.cnn4 = nn.Conv2d(24,32, kernel_size=3, stride=1)#12x12
        self.cnn5 = nn.Conv2d(32,42,kernel_size = 6, stride=3)#bilder in 3x3 dimension
        #elif emb_size == 2:

        #self.cnn3 = nn.Conv2d(5, 1, kernel_size=5, stride=2)

        #linear layer, nimmt convolution entgegen
        self.encode = nn.Linear(42*3*3, emb_size*2)#'''hier das geiche vll doch nochmal convolution so anassen dass größer?'''
        self.decode = nn.Linear(emb_size, 42*3*3)#von repräsentationsgröße wieder auf die, die in 2d darstellungen verarbeitet werden kann

        #decode convolutional layer
        self.cnn1_decode = nn.ConvTranspose2d(42,32, kernel_size=6, dilation=1, stride=2)#10x10
        self.cnn2_decode = nn.ConvTranspose2d(32,24, kernel_size=6, dilation=1, stride=2)#24x24
        self.cnn3_decode = nn.ConvTranspose2d(24,15, kernel_size=4, dilation=1, stride=2)#50x50
        self.cnn4_decode = nn.ConvTranspose2d(15,8, kernel_size=8, dilation=1, stride=1)#57x57
        self.cnn5_decode = nn.ConvTranspose2d(8,3, kernel_size=8, dilation=1, stride=1)#64x64

        #linear layer um log variance und mean zu erzeugen
        self.mean = nn.Linear(emb_size*2, emb_size)
        self.log_var = nn.Linear(emb_size*2, emb_size)

    def encode_convolutions(self,x):
        '''
        takes input and uses convolutional layer
        returns tensor
        '''
        x = F.relu(self.cnn1(x.float()))
        x = self.batch_norm1(x)
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = self.batch_norm2(x)
        x = F.relu(self.cnn4(x))
        x = self.batch_norm3(x)
        x = F.relu(self.cnn5(x))
        return x

    def sample(self, x):
        x = F.relu(self.encode(x.flatten(start_dim=1)))#flatten der bild matrix zu einem tensor
        mean_ = self.mean(x)#layer, da die dimension auf die embeddingsize reduziert
        #auch layer, das input auf die embeddingsize reduziert
        log_var = torch.exp(0.5*self.log_var(x))
        assert log_var.shape == mean_.shape#check für mich
        #normalverteiten rauschenvektor erstellen mit dimension von log var / mean
        n_ = torch.randn_like(log_var)

        #sample aus der so berechneten verteilung ziehen
        sample = mean_ + (n_*log_var)
        #sample, mean und std returnen (werden für loss benötigt)
        return sample, mean_, log_var

    def decode_convolutions(self, sample_):
        #sample mit linear decoder in so eine form bringen, dass es 
        #in einem nächsten schritt in benötigte form für conv layer 
        #gebracht werden kann und bild rekonstruiert werden kann
        #sample mit hier uafzunehmen ist wegen verwendbarkeit des decoders
        #im generationsschritt sinnvoll
        x = F.relu(self.decode(sample_))

        #reshapen der dimensionen, für richtiges convtranspose format
        x = x.view(self.batch_size,42,3,3)#1.5.5
        #hier die richtige dimension!!!
        
        #transposed convolutions nutzen, um ursprüngliche 
        #bildgröße wieder herzustellen
        x = F.relu(self.cnn1_decode(x))
        x = self.batch_norm3(x)
        x = F.relu(self.cnn2_decode(x))
        x = self.batch_norm2(x)
        x = F.relu(self.cnn3_decode(x))
        x = F.relu(self.cnn4_decode(x))
        x = self.batch_norm1(x)

        #vorher war hier relu und danach erst sigmoid
        if self.sigmoid:
            x = torch.sigmoid(self.cnn5_decode(x))
        else:
            x = F.relu(self.cnn5_decode(x))
        return x

    def forward(self,x):
        #encoden des inputs mit hilfe 
        self.batch_size = x.shape[0]
        encoded = self.encode_convolutions(x)
        sample_, mean, std = self.sample(encoded)

        #decoden des samples mit linear layer, um auf 5*5 bilddimension im
        #convolutional decoder zugreifen zu können
        #x = F.relu(self.decode(sample_))

        #aufrufen der convolutional transposed layer für den decoder
        #teil des VAE

        #sigmoid wegen bce cuda error: device side assert triggerd nn.Sigmoid(
        reconstructed = self.decode_convolutions(sample_)
        return reconstructed, mean, std


class VAE_Pipeline64(object):

    def __init__(self, path_images, lr=0.0005, num_epochs = 100, batch_size = 64, color = True, pretrained = False, model_path = None):

        #generelle informationen fürs training und zu den daten
        self.path = path_images
        self.epochs = num_epochs
        self.lr = lr
        self.color = color
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.model_path = model_path

        #modell, optimierer und lossfunction platzhalter
        self.model = None
        self.optimizer = None
        self.criterion = None

        #log fü den trainingsloss
        self.loss_log = pd.DataFrame(columns = ['epoche', 'loss'])
        self.reconstruction = dict()


        #listen, die bilder und batches mit bildern enthalten
        self.data = list()
        self.batches = list()

        #funktionen zum einlesen der bilder, erzeugen von batches und
        #trainieren des modells aufrufen, danach kann gernerate funktion der 
        #klasse aufgerufen und neue bilder erzeugt werden
        self.load_images()
        self.create_batches()
        self.train_model()

    def load_images(self):
        images = os.listdir(self.path)
        for i, image in tqdm(enumerate(images)):
            img_path = os.path.join(self.path, image)
            if self.color:
                img = cv2.imread(img_path, 1)
            else:
                img = cv2.imread(img_path, 0)
            #bilder auf einheitliche größe bringen
            img = cv2.resize(img, (64,64))
            self.data.append(img)

    def create_batches(self):
        batch = list()
        random.shuffle(self.data)
        for k, d_  in enumerate(self.data[:200], start = 1):
            if k % self.batch_size == 0:
                if self.color:
                    img = torch.from_numpy(d_/255).permute(2,0,1).to(device)#color diemension is last dimension, but needs to be first dimension --> permute
                else:
                    img = torch.from_numpy(cv2.equalizeHist(copy.deepcopy(d_))/255).unsqueeze(0).to(device)
                #img = torch.from_numpy(d_/255).unsqueeze(0).to(device)
                batch.append(img)
                batch = torch.stack(batch)
                self.batches.append(batch)
                batch = list()
            else:
                if self.color:
                    img = torch.from_numpy(d_/255).permute(2,0,1).to(device)
                else:
                    img = torch.from_numpy(cv2.equalizeHist(copy.deepcopy(d_))/255).unsqueeze(0).to(device)
                #img = torch.from_numpy(d_/255).unsqueeze(0).to(device)
                batch.append(img)

        if len(batch) > 0:
            batch = torch.stack(batch)
            self.batches.append(batch)
            batch = list()

    def final_loss(self, bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the 
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train(self):
        self.model.train()
        running_loss = 0
        key = list(self.reconstruction.keys())[-1]
        for batch_id, batch in enumerate(self.batches):
            self.optimizer.zero_grad()
            out, mu, std = self.model(batch.float())
            self.reconstruction[key].append((batch, out))
            bce_loss = self.criterion(out, batch.float())
            loss = self.final_loss(bce_loss, mu, std)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if batch_id % 10 == 0:
                #print(f"Fortschritt: {batch_id/len(self.batches)*100}% finished")
                pass
        avg_loss = running_loss/len(self.batches)
        return avg_loss

    def train_model(self):
        
        if self.color:
            self.model = VAEsmall_color(sigmoid = False).to(device)
            if self.pretrained:#laden einer bereits trainierten version des modells
                assert str(self.model_path) != 'None', 'model path needs to be specified!'
                print('load model from', self.model_path)
                self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        else:
            print('version without color not implemented yet!')
        #self.model = ImageVAE_grey400(sigmoid = False).to(device)
        
        lr = self.lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #wenn logits loss verwendet wird kein sigmoid layer, aber target sollt in [0,1] liegen
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        print(self.model)
        for epoch in range(self.epochs):
            print('epoche:', epoch+1)
            self.reconstruction[epoch] = list()
            train_loss = self.train()
            self.loss_log.loc[len(self.loss_log), :] = [epoch+1, train_loss]

    def generate_images(self, num_images, emb_size = 30, color = True, norm = False):
        self.model.eval()
        with torch.no_grad():
            for i in range(num_images):
                tensor = torch.zeros(emb_size)
                print(tensor.shape)
                prefix = 'normalized_'
                if not norm:
                    prefix = 'random_'
                    for k in range(tensor.shape[0]):
                        z = random.randint(-100,10)
                        tensor[k] = z
                    tensor = tensor.float().to(device)*torch.randn_like(torch.zeros(emb_size)).to(device)
                else:
                    tensor = torch.randn_like(torch.zeros(emb_size)).to(device)
                #tensor = torch.randn_like(torch.zeros(emb_size)).to(device)
                self.model.batch_size = 1
                img = self.model.decode_convolutions(tensor).cpu()
                if self.color:
                    img = img.permute(0,2,3,1).detach().numpy()[0]*255
                else:
                    img = img.detach().numpy()[0][0]*255
                #print(img.shape)
                img = cv2.resize(img, (300,300))
                cv2_imshow(img)
                #plt.figure()
                #plt.imshow(img/255)
                #img[:, :, 0] = 0
                #img[:, :, 2] = 0
                #cv2_imshow(img)
                #folder = '/content/drive/My Drive/ImageGeneration/generated_images'
                #file_name = prefix+str(i)+'_epoche_20_400400_ersteVersion_withlogitsloss_neueVersion.png'
                #if not os.path.exists(folder):
                #    os.mkdir(folder)
                #cv2.imwrite(os.path.join(folder, file_name), img)
                print('latent vector:', tensor)


###################################################
###################################################
######### ALTE UND GRÖßERE MODELLE#################
###################################################
###################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ImageVAE_color400_alt(nn.Module):

    def __init__(self, rep_dim = 5, sigmoid = True):
        #rep dim = 5
        super(ImageVAE_color400_alt, self).__init__()
        #speichern der embedding größe um diese nachher bie generation von bildenr nutzen zu können
        self.embedding_dim = rep_dim
        self.batch_size = 1
        self.sigmoid = sigmoid

        if sigmoid:
            print('please use BCEloss as criterion with sigmoid activation function')
        else:
            print('please use BCEWithLogitsLoss as criterion')
        #definieren von polling layern (max pooling layer in verschiedenen größen)
        self.pool_1 = nn.MaxPool2d(3,1)
        self.pool_2 = nn.MaxPool2d(3,2)
        self.pool_3 = nn.MaxPool2d(6,2)

        #Encode-Convolutional-Layer
        self.cnn1 = nn.Conv2d(3, 10, kernel_size=10, stride=2)
        self.cnn2 = nn.Conv2d(10, 5, kernel_size = 8, stride = 3)
        self.cnn3 = nn.Conv2d(5, 1, kernel_size=5, stride=2)

        #linear layer, nimmt convolution entgegen
        self.encode = nn.Linear(1*5*5, rep_dim*2)
        self.decode = nn.Linear(rep_dim, 1*5*5)

        #decode convolutional layer
        self.cnn1_decode = nn.ConvTranspose2d(1,5, kernel_size=6, dilation=3, stride=2)
        #die letzten zwei nochmal anpassen!!, dimensionen sind nicht so!
        self.cnn2_decode = nn.ConvTranspose2d(5,10, kernel_size=4, dilation=1, stride=3)
        self.cnn3_decode = nn.ConvTranspose2d(10,3, kernel_size=40, dilation=1, stride=5)

        #linear layer um log variance und mean zu erzeugen
        self.mean = nn.Linear(rep_dim*2, rep_dim)
        self.log_var = nn.Linear(rep_dim*2, rep_dim)

    def encode_convolutions(self,x):
        '''
        takes input and uses convolutional and pooling layer
        returns tensor
        '''
        x = self.pool_1(F.relu(self.cnn1(x.float())))
        #x.shape = (N, C, 194, 194)
        x = self.pool_2(F.relu(self.cnn2(x)))
        x = self.pool_3(F.relu(self.cnn3(x)))
        return x

    def sample(self, x):
        x = F.relu(self.encode(x.flatten(start_dim=1)))#flatten der matrix zu einem tensor
        mean_ = self.mean(x)#layer, da die dimension auf die embeddingsize reduziert
        #auch layer, das input auf die embeddingsize reduziert
        log_var = torch.exp(0.5*self.log_var(x))
        assert log_var.shape == mean_.shape#check für mich
        #normalverteiten rauschenvektor erstellen mit dimension von log var / mean
        n_ = torch.randn_like(log_var)

        #sample aus der so berechneten verteilung ziehen
        sample = mean_ + (n_*log_var)
        #sample, mean und std returnen (werden für loss benötigt)
        return sample, mean_, log_var

    def decode_convolutions(self, sample_):
        #sample mit linear decoder in so eine form bringen, dass es 
        #in einem nächsten schritt in benötigte form für conv layer 
        #gebracht werden kann und bild rekonstruiert werden kann
        #sample mit hier uafzunehmen ist wegen verwendbarkeit des decoders
        #im generationsschritt sinnvoll
        x = F.relu(self.decode(sample_))

        #reshapen der dimensionen, für richtiges convtranspose format
        x = x.view(self.batch_size,1,5,5)
        #hier die richtige dimension!!!
        
        #transposed convolutions nutzen, um ursprüngliche 
        #bildgröße wieder herzustellen
        x = F.relu(self.cnn1_decode(x))
        x = F.relu(self.cnn2_decode(x))
        #vorher war hier relu und danach erst sigmoid
        if self.sigmoid:
            x = torch.sigmoid(self.cnn3_decode(x))
        else:
            x = F.relu(self.cnn3_decode(x))
        return x

    def forward(self,x):
        #encoden des inputs mit hilfe 
        self.batch_size = x.shape[0]
        encoded = self.encode_convolutions(x)
        sample_, mean, std = self.sample(encoded)

        #decoden des samples mit linear layer, um auf 5*5 bilddimension im
        #convolutional decoder zugreifen zu können
        #x = F.relu(self.decode(sample_))

        #aufrufen der convolutional transposed layer für den decoder
        #teil des VAE

        #sigmoid wegen bce cuda error: device side assert triggerd nn.Sigmoid(
        reconstructed = self.decode_convolutions(sample_)
        return reconstructed, mean, std

###############################
###############################
###############################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ImageVAE_color400(nn.Module):

    def __init__(self, rep_dim = 5, sigmoid = True):
        #rep dim = 5
        super(ImageVAE_color400, self).__init__()
        #speichern der embedding größe um diese nachher bie generation von bildenr nutzen zu können
        self.embedding_dim = rep_dim
        self.batch_size = 1
        self.sigmoid = sigmoid

        if sigmoid:
            print('please use BCEloss as criterion with sigmoid activation function')
        else:
            print('please use BCEWithLogitsLoss as criterion')
        #definieren von polling layern (max pooling layer in verschiedenen größen)
        self.pool_1 = nn.MaxPool2d(3,1)
        self.pool_2 = nn.MaxPool2d(3,2)
        self.pool_3 = nn.MaxPool2d(6,2)

        #Encode-Convolutional-Layer
        self.cnn1 = nn.Conv2d(3, 10, kernel_size=10, stride=2)
        self.cnn2 = nn.Conv2d(10, 5, kernel_size = 8, stride = 3)
        self.cnn3 = nn.Conv2d(5, 1, kernel_size=5, stride=2)

        #linear layer, nimmt convolution entgegen
        self.encode = nn.Linear(1*5*5, rep_dim*2)
        self.decode = nn.Linear(rep_dim, 1*5*5)

        #decode convolutional layer
        self.cnn1_decode = nn.ConvTranspose2d(1,5, kernel_size=10, dilation=2, stride=3, output_padding=2)
        #die letzten zwei nochmal anpassen!!, dimensionen sind nicht so!
        self.cnn2_decode = nn.ConvTranspose2d(5,25, kernel_size=7, dilation=2, stride=1)
        self.cnn3_decode = nn.ConvTranspose2d(25,50, kernel_size=10, dilation=6, stride=2, output_padding=5, padding=1)
        #stride = 2, padding=0, dil = 3, kernel =8, output=1
        self.cnn4_decode = nn.ConvTranspose2d(50,10, kernel_size=8, dilation=3, stride=2,output_padding=1, padding=0)
        #stride = 1, padding=0, dil = 9, kernel =10, output=6)
        self.cnn5_decode = nn.ConvTranspose2d(10,3, kernel_size=10, dilation=9, stride=1, padding=0, output_padding=6)

        #linear layer um log variance und mean zu erzeugen
        self.mean = nn.Linear(rep_dim*2, rep_dim)
        self.log_var = nn.Linear(rep_dim*2, rep_dim)

    def encode_convolutions(self,x):
        '''
        takes input and uses convolutional and pooling layer
        returns tensor
        '''
        x = self.pool_1(F.relu(self.cnn1(x.float())))
        #x.shape = (N, C, 194, 194)
        x = self.pool_2(F.relu(self.cnn2(x)))
        x = self.pool_3(F.relu(self.cnn3(x)))
        return x

    def sample(self, x):
        x = F.relu(self.encode(x.flatten(start_dim=1)))#flatten der matrix zu einem tensor
        mean_ = self.mean(x)#layer, da die dimension auf die embeddingsize reduziert
        #auch layer, das input auf die embeddingsize reduziert
        log_var = torch.exp(0.5*self.log_var(x))
        assert log_var.shape == mean_.shape#check für mich
        #normalverteiten rauschenvektor erstellen mit dimension von log var / mean
        n_ = torch.randn_like(log_var)

        #sample aus der so berechneten verteilung ziehen
        sample = mean_ + (n_*log_var)
        #sample, mean und std returnen (werden für loss benötigt)
        return sample, mean_, log_var

    def decode_convolutions(self, sample_):
        #sample mit linear decoder in so eine form bringen, dass es 
        #in einem nächsten schritt in benötigte form für conv layer 
        #gebracht werden kann und bild rekonstruiert werden kann
        #sample mit hier uafzunehmen ist wegen verwendbarkeit des decoders
        #im generationsschritt sinnvoll
        x = F.relu(self.decode(sample_))

        #reshapen der dimensionen, für richtiges convtranspose format
        x = x.view(self.batch_size,1,5,5)
        #hier die richtige dimension!!!
        
        #transposed convolutions nutzen, um ursprüngliche 
        #bildgröße wieder herzustellen
        x = F.relu(self.cnn1_decode(x))
        x = F.relu(self.cnn2_decode(x))
        x = F.relu(self.cnn3_decode(x))
        x = F.relu(self.cnn4_decode(x))
        #vorher war hier relu und danach erst sigmoid
        if self.sigmoid:
            x = torch.sigmoid(self.cnn5_decode(x))
        else:
            x = F.relu(self.cnn5_decode(x))
        return x

    def forward(self,x):
        #encoden des inputs mit hilfe 
        self.batch_size = x.shape[0]
        encoded = self.encode_convolutions(x)
        sample_, mean, std = self.sample(encoded)

        #decoden des samples mit linear layer, um auf 5*5 bilddimension im
        #convolutional decoder zugreifen zu können
        #x = F.relu(self.decode(sample_))

        #aufrufen der convolutional transposed layer für den decoder
        #teil des VAE

        #sigmoid wegen bce cuda error: device side assert triggerd nn.Sigmoid(
        reconstructed = self.decode_convolutions(sample_)
        return reconstructed, mean, std

#############################
#############################
#############################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class ImageVAE_grey400(nn.Module):

    def __init__(self, rep_dim = 5, sigmoid = True):
        #rep dim = 5
        super(ImageVAE_grey400, self).__init__()
        #speichern der embedding größe um diese nachher bie generation von bildenr nutzen zu können
        self.embedding_dim = rep_dim
        self.batch_size = 1
        self.sigmoid = sigmoid

        if sigmoid:
            print('please use BCEloss as criterion with sigmoid activation function')
        else:
            print('please use BCEWithLogitsLoss as criterion')

        #definieren von polling layern (max pooling layer in verschiedenen größen)
        self.pool_1 = nn.MaxPool2d(3,1)
        self.pool_2 = nn.MaxPool2d(3,2)
        self.pool_3 = nn.MaxPool2d(6,2)

        #Encode-Convolutional-Layer
        self.cnn1 = nn.Conv2d(1, 10, kernel_size=10, stride=2)
        self.cnn2 = nn.Conv2d(10, 5, kernel_size = 8, stride = 3)
        self.cnn3 = nn.Conv2d(5, 1, kernel_size=5, stride=2)

        #linear layer, nimmt convolution entgegen
        self.encode = nn.Linear(1*5*5, rep_dim*2)
        self.decode = nn.Linear(rep_dim, 1*5*5)

        #decode convolutional layer
        self.cnn1_decode = nn.ConvTranspose2d(1,5, kernel_size=6, dilation=3, stride=2)
        #die letzten zwei nochmal anpassen!!, dimensionen sind nicht so!
        self.cnn2_decode = nn.ConvTranspose2d(5,10, kernel_size=4, dilation=1, stride=3)
        self.cnn3_decode = nn.ConvTranspose2d(10,1, kernel_size=40, dilation=1, stride=5)

        #linear layer um log variance und mean zu erzeugen
        self.mean = nn.Linear(rep_dim*2, rep_dim)
        self.log_var = nn.Linear(rep_dim*2, rep_dim)

    def encode_convolutions(self,x):
        '''
        takes input and uses convolutional and pooling layer
        returns tensor
        '''
        x = self.pool_1(F.relu(self.cnn1(x.float())))
        #x.shape = (N, C, 194, 194)
        x = self.pool_2(F.relu(self.cnn2(x)))
        x = self.pool_3(F.relu(self.cnn3(x)))
        return x

    def sample(self, x):
        x = F.relu(self.encode(x.flatten(start_dim=1)))#flatten der matrix zu einem tensor
        mean_ = self.mean(x)#layer, da die dimension auf die embeddingsize reduziert
        #auch layer, das input auf die embeddingsize reduziert
        log_var = torch.exp(0.5*self.log_var(x))
        assert log_var.shape == mean_.shape#check für mich
        #normalverteiten rauschenvektor erstellen mit dimension von log var / mean
        n_ = torch.randn_like(log_var)

        #sample aus der so berechneten verteilung ziehen
        sample = mean_ + (n_*log_var)
        #sample, mean und std returnen (werden für loss benötigt)
        return sample, mean_, log_var

    def decode_convolutions(self, sample_):
        #sample mit linear decoder in so eine form bringen, dass es 
        #in einem nächsten schritt in benötigte form für conv layer 
        #gebracht werden kann und bild rekonstruiert werden kann
        #sample mit hier uafzunehmen ist wegen verwendbarkeit des decoders
        #im generationsschritt sinnvoll
        x = F.relu(self.decode(sample_))

        #reshapen der dimensionen, für richtiges convtranspose format
        x = x.view(self.batch_size,1,5,5)
        #hier die richtige dimension!!!
        
        #transposed convolutions nutzen, um ursprüngliche 
        #bildgröße wieder herzustellen
        x = F.relu(self.cnn1_decode(x))
        x = F.relu(self.cnn2_decode(x))
        #vorher war hier relu und danach erst sigmoid
        if self.sigmoid:
            x = torch.sigmoid(self.cnn3_decode(x))
        else:
            x = F.relu(self.cnn3_decode(x))
        return x

    def forward(self,x):
        #encoden des inputs mit hilfe 
        self.batch_size = x.shape[0]
        encoded = self.encode_convolutions(x)
        sample_, mean, std = self.sample(encoded)

        #decoden des samples mit linear layer, um auf 5*5 bilddimension im
        #convolutional decoder zugreifen zu können
        #x = F.relu(self.decode(sample_))

        #aufrufen der convolutional transposed layer für den decoder
        #teil des VAE

        #sigmoid wegen bce cuda error: device side assert triggerd nn.Sigmoid(
        reconstructed = self.decode_convolutions(sample_)
        return reconstructed, mean, std




class VAE_Pipeline400(object):

    def __init__(self, path_images, lr=0.0005, num_epochs = 100, batch_size = 64, color = True):

        #generelle informationen fürs training und zu den daten
        self.path = path_images
        self.epochs = num_epochs
        self.lr = lr
        self.color = color
        self.batch_size = batch_size

        #modell, optimierer und lossfunction platzhalter
        self.model = None
        self.optimizer = None
        self.criterion = None

        #log fü den trainingsloss
        self.loss_log = pd.DataFrame(columns = ['epoche', 'loss'])
        self.reconstruction = dict()


        #listen, die bilder und batches mit bildern enthalten
        self.data = list()
        self.batches = list()

        #funktionen zum einlesen der bilder, erzeugen von batches und
        #trainieren des modells aufrufen, danach kann gernerate funktion der 
        #klasse aufgerufen und neue bilder erzeugt werden
        self.load_images()
        self.create_batches()
        self.train_model()

    def load_images(self):
        images = os.listdir(self.path)
        for i, image in tqdm(enumerate(images)):
            img_path = os.path.join(self.path, image)
            if self.color:
                img = cv2.imread(img_path, 1)
            else:
                img = cv2.imread(img_path, 0)
            #bilder auf einheitliche größe bringen
            img = cv2.resize(img, (400,400))
            self.data.append(img)

    def create_batches(self):
        batch = list()
        random.shuffle(self.data)
        for k, d_  in enumerate(self.data[:200], start = 1):
            if k % self.batch_size == 0:
                if self.color:
                    img = torch.from_numpy(d_/255).permute(2,0,1).to(device)#color diemension is last dimension, but needs to be first dimension --> permute
                else:
                    img = torch.from_numpy(cv2.equalizeHist(copy.deepcopy(d_))/255).unsqueeze(0).to(device)
                #img = torch.from_numpy(d_/255).unsqueeze(0).to(device)
                batch.append(img)
                batch = torch.stack(batch)
                self.batches.append(batch)
                batch = list()
            else:
                if self.color:
                    img = torch.from_numpy(d_/255).permute(2,0,1).to(device)
                else:
                    img = torch.from_numpy(cv2.equalizeHist(copy.deepcopy(d_))/255).unsqueeze(0).to(device)
                #img = torch.from_numpy(d_/255).unsqueeze(0).to(device)
                batch.append(img)

        if len(batch) > 0:
            batch = torch.stack(batch)
            self.batches.append(batch)
            batch = list()

    def final_loss(self, bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the 
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train(self):
        self.model.train()
        running_loss = 0
        key = list(self.reconstruction.keys())[-1]
        for batch_id, batch in enumerate(self.batches):
            self.optimizer.zero_grad()
            out, mu, std = self.model(batch.float())
            self.reconstruction[key].append((batch, out))
            bce_loss = self.criterion(out, batch.float())
            loss = self.final_loss(bce_loss, mu, std)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if batch_id % 10 == 0:
                print(f"Fortschritt: {batch_id/len(self.batches)*100}% finished")
        avg_loss = running_loss/len(self.batches)
        return avg_loss

    def train_model(self):
        if self.color:
            self.model = ImageVAE_color400(sigmoid = False).to(device)
        else:
            self.model = ImageVAE_grey400(sigmoid = False).to(device)
        lr = self.lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #wenn logits loss verwendet wird kein sigmoid layer, aber target sollt in [0,1] liegen
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        print(self.model)
        for epoch in range(self.epochs):
            print('epoche:', epoch+1)
            self.reconstruction[epoch] = list()
            train_loss = self.train()
            self.loss_log.loc[len(self.loss_log), :] = [epoch+1, train_loss]

    def generate_images(self, num_images, emb_size = 5, color = False, norm = False):
        for i in range(num_images):
            tensor = torch.zeros(emb_size)
            print(tensor.shape)
            prefix = 'normalized_'
            if not norm:
                prefix = 'random_'
                for k in range(tensor.shape[0]):
                    z = random.randint(-100,10)
                    tensor[k] = z
                tensor = tensor.float().to(device)*torch.randn_like(torch.zeros(emb_size)).to(device)
            else:
                tensor = torch.randn_like(torch.zeros(emb_size)).to(device)
            #tensor = torch.randn_like(torch.zeros(emb_size)).to(device)
            self.model.batch_size = 1
            img = self.model.decode_convolutions(tensor).cpu()
            if self.color:
                img = img.permute(0,2,3,1).detach().numpy()[0]*255
            else:
                img = img.detach().numpy()[0][0]*255
            #print(img.shape)
            cv2_imshow(img)
            #plt.figure()
            print('latent vector:', tensor)