import torch.optim as optim
import torch
import torch.nn as nn

import random 
import numpy as np

import cv2
import os
import copy
import pandas as pd

from tqdm import tqdm 

s = 0
random.seed(s)
torch.manual_seed(s)
np.random.seed(s)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAETrainer:

    def __init__(self, train_batches):

        #festlegen woher daten kommen --> neu einlesen oder schon in liste vorhanden?
        if isinstance(train_batches, list):# train batches ist liste --> wird als fertige batches interpretiert
            self.train_batches = train_batches
        elif isinstance(train_batches, str):#train batches ist str --> wird als pfad interpretiert
            self.path = train_batches
        else:
            raise ValueError('unknown type for variable train_batches')

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

class VAE_TrainPipeline(object):

    def __init__(self, path_images, lr=0.0005, num_epochs = 100, batch_size = 64, loss_func = 'BCE',color = True, pretrained = False, model_path = None, use_augmentation = False):
        '''
        Klasse, die alle relevanten Vorverarbeitung- und Trainingsschritte 
        für einen VAE vornimmt. Parameter:
        path_images: Pfad zu den einzulesenden Bildern
        lr: Lernrate
        num_epochs: Anzahl zu trainierender Epochen
        batch_size: Batch Größe fürs Training des Modells
        loss_func: Gibt an welche Lossfunktion verwendet wird (BCE = BCEWithLogits, MSE = MSE)
        color: True/False, ob Bilder mit 3 oder 1 Farbchannel eingelesen werden (vollständig implementiert nur für 3)
        pretrained: True/False ob ein Modell eingelesen und weiter trainiert werden soll
        model_path: Pfad zu dem einzulesenden und weiter zu trainierend Modell (wenn pretrained = True muss Pfad angegeben werden)
        use_augmentation: True / False ob Bilder bbeim Einlesne augmentiert werden sollen
        '''
        #generelle informationen fürs training und zu den daten
        self.path = path_images
        self.epochs = num_epochs
        self.loss_func = loss_func
        self.lr = lr
        self.color = color
        self.batch_size = batch_size
        self.pretrained = pretrained
        self.model_path = model_path
        self.use_augmentation = use_augmentation

        #modell, optimierer und lossfunction platzhalter
        self.model = None
        self.optimizer = None
        self.criterion = None

        #log für den trainingsloss
        self.loss_log = pd.DataFrame(columns = ['epoche', 'loss'])
        #dict enthält für jede epoche originalbild und das rekonstruierte bild
        self.reconstruction = dict()


        #listen, die bilder und batches mit bildern enthalten
        self.data = list()
        self.batches = list()
        #dicts die mittelwerte und var für jede epoche enthalten
        self.mus = dict()
        self.stds = dict()

        #funktionen zum einlesen der bilder, erzeugen von batches und
        #trainieren des modells aufrufen, danach kann generate funktion der 
        #klasse aufgerufen und neue bilder erzeugt werden
        self.load_images()
        self.create_batches()
        #self.train_model()

    def augment_image(self, img):
        flip_ = random.choice([-1,0,1,2])
        if flip_ == 2:#bild wird nur in der originalversion im datensatz verwendet
            return ''
        else:
            return cv2.flip(img,flip_)

    def load_images(self):
        
        if isinstance(self.path, list):#more than one path to read images from
            images = dict()
            for p in self.path:
                im_ = os.listdir(p)
                images[p] = im_
        else:
            images = os.listdir(self.path)#alle dateinamen aus angegebenem Pfad auslesen
        if isinstance(images, list):
            for i, image in tqdm(enumerate(images)):
                img_path = os.path.join(self.path, image)
                if self.color:
                    img = cv2.imread(img_path, 1)
                else:#einlesen in schwarz weiß möglich, weiteres verarbeiten der bilder jedoch nicht!
                    img = cv2.imread(img_path, 0)

                if self.use_augmentation:
                     #bilder augmentieren und auf einheitliche größe bringen
                    augmented = self.augment_image(img)
                    if augmented != '':
                        augmented = cv2.resize(augmented, (64,64))
                        self.data.append(augmented)
                #bilder auf einheitliche größe bringen
                img = cv2.resize(img, (64,64))
                self.data.append(img)
        elif isinstance(images, dict):
            for path_, images_ in images.items():
                for image in images_:
                    img_path = os.path.join(path_, image)
                    if self.color:
                        img = cv2.imread(img_path, 1)
                    else:#einlesen in schwarz weiß möglich, weiteres verarbeiten der bilder jedoch nicht!
                        img = cv2.imread(img_path, 0)

                    if self.use_augmentation:
                        #bilder augmentieren und auf einheitliche größe bringen
                        augmented = self.augment_image(img)#randomly checking if image should be augmented
                        if augmented != '':
                            augmented = cv2.resize(augmented, (64,64))
                            self.data.append(augmented)
                    #bilder auf einheitliche größe bringen
                    img = cv2.resize(img, (64,64))
                    self.data.append(img)

    def create_batches(self):
        batch = list()
        random.shuffle(self.data)#mischen der Bilder 
        for k, d_  in enumerate(self.data, start = 1):
            if k % self.batch_size == 0:
                if self.color:#normalisierte pixelwerte der bilder abspeichern
                    img = torch.from_numpy(d_/255).permute(2,0,1).to(device)#color diemension is last dimension, but needs to be first dimension --> permute
                else:
                    img = torch.from_numpy(cv2.equalizeHist(copy.deepcopy(d_))/255).unsqueeze(0).to(device)
                batch.append(img)
                batch = torch.stack(batch)#bilder in batch zu einem tensor zusammenmatschen
                self.batches.append(batch)
                batch = list()
            else:
                if self.color:
                    img = torch.from_numpy(d_/255).permute(2,0,1).to(device)
                else:
                    img = torch.from_numpy(cv2.equalizeHist(copy.deepcopy(d_))/255).unsqueeze(0).to(device)
                batch.append(img)

        if len(batch) > 0:#es gibt noch einen letzten kleineren batch mit einer batch size < definierte batch size
            batch = torch.stack(batch)
            self.batches.append(batch)
            batch = list()

    def combined_loss(self, loss, mu, logvar):
        '''
        Funktion kombiniert den Rekonstruktionsloss (BCE/MSE etc.)
        mit Nebenbedingung der KL Divergenz, um eine Normalverteilung
        im latenten Raum zu erzwingen
        '''
        LOSS = loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return LOSS+KLD

    def train(self):
        self.model.train()
        running_loss = 0
        #aktuelle stelle im dictionary auslesen
        key = list(self.reconstruction.keys())[-1]

        for batch_id, batch in enumerate(self.batches):
            self.optimizer.zero_grad()
            out, mu, std = self.model(batch.float())
            
            if batch_id == 1:#nur ausgewählte bilder abspeichern um cuda memory zu sparen
                self.reconstruction[key].append((batch, out))
                self.mus[key].append(mu)
                self.stds[key].append(std)

            loss = self.criterion(out, batch.float())
            loss = self.combined_loss(loss, mu, std)
            running_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            if batch_id % 10 == 0:
                #print(f"Fortschritt: {batch_id/len(self.batches)*100}% finished")
                pass
        avg_loss = running_loss/len(self.batches)
        return avg_loss

    def train_model(self, model):
        self.model = model
        lr = self.lr
        #wenn logits loss verwendet wird kein sigmoid layer, aber target sollt in [0,1] liegen
        if self.loss_func.lower() == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction='sum')#nn.MSELoss()
            if self.color:
                #self.model = VAEsmall_color(sigmoid = False).to(device)
                if self.pretrained:#laden einer bereits trainierten version des modells
                    assert str(self.model_path) != 'None', 'model path needs to be specified!'
                    print('load model from', self.model_path)
                    self.model.load_state_dict(torch.load(self.model_path, map_location=device))
            else:
                print('version without color not implemented yet!')
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        #MSE wird als Lossfunktion verwendet    
        elif self.loss_func.lower() == 'mse':
            self.criterion = nn.MSELoss()
            if self.color:
                #self.model = VAEsmall_color(sigmoid = True).to(device)
                if self.pretrained:#laden einer bereits trainierten version des modells
                    assert str(self.model_path) != 'None', 'model path needs to be specified!'
                    print('load model from', self.model_path)
                    self.model.load_state_dict(torch.load(self.model_path, map_location=device))
            else:
                print('version without color not implemented yet!')
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        #Als Loss wurde ein Parameter angegeben, der nicht implementiert ist
        else:
            raise ValueError ('unknown criterion, try MSE or BCE')
        print(self.model)
        for epoch in tqdm(range(self.epochs)):
            #print('epoche:', epoch+1)
            #storing reconstructed images, mus, stds in list 
            self.reconstruction[epoch] = list()
            self.mus[epoch] = list()
            self.stds[epoch] = list()
            #trainieren des modells
            train_loss = self.train()
            self.loss_log.loc[len(self.loss_log), :] = [epoch+1, train_loss]
        return model

    def generate_random_images(self, num_images, emb_size = 30, color = True, norm = False):
        self.model.eval()
        images = list()
        with torch.no_grad():
            for i in range(num_images):
                tensor = torch.zeros(emb_size)
                print(tensor.shape)
                if not norm:
                    for k in range(tensor.shape[0]):
                        z = random.randint(-100,10)
                        tensor[k] = z
                    tensor = tensor.float().to(device)*torch.randn_like(torch.zeros(emb_size)).to(device)
                else:
                    tensor = torch.randn_like(torch.zeros(emb_size)).to(device)
                #tensor = torch.randn_like(torch.zeros(emb_size)).to(device)
                self.model.batch_size = 1

                #checken wie die decoder funktionen in VAE heißen --> sind bekannte Funktionen in Modell enthalzen?
                if 'decode_convolutions' in dir(self.model):
                    img = self.model.decode_convolutions(tensor).cpu()
                elif 'decoder' in dir(self.model):
                    tensor = tensor.reshape(1, emb_size)
                    img = self.model.decoder(tensor).cpu()
                if self.color:
                    img = img.permute(0,2,3,1).detach().numpy()[0]*255
                else:
                    img = img.detach().numpy()[0][0]*255
                #print(img.shape)
                img = cv2.resize(img, (300,300))
                #cv2_imshow(img)
                print('latent vector:', tensor)
                images.append(img)
        return images