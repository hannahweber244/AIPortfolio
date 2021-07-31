"""genetischPole controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, PositionSensor, Motor, Supervisor
from robot_model import PoleRobot
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import operator
import random
import numpy as np
from tqdm import tqdm

from copy import deepcopy

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# create the Robot instance.
#robot = Robot()
robot = Supervisor()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
print(f"timestep: {timestep}")

#get all available devices for the robot
for i in range(robot.getNumberOfDevices()):
    print(str(robot.getDeviceByIndex(i)) + ": " + robot.getDeviceByIndex(i).getName())

#get access to pole position sensor
polePos:PositionSensor = robot.getDevice('polePosSensor')
polePos.enable(timestep)

#get access to motor
motor1: Motor = robot.getDevice('wheel1')
motor2: Motor = robot.getDevice('wheel2')
motor3: Motor = robot.getDevice('wheel3')
motor4: Motor = robot.getDevice('wheel4')

#set wheels to velocity control
motor1.setPosition(float('inf'))
motor2.setPosition(float('inf'))
motor3.setPosition(float('inf'))
motor4.setPosition(float('inf'))

class RobotIndividuum(object):

    def __init__(self):
        self.weights = PoleRobot()#weights parameter ist ein Modell

        #nograd model parameters
        for param in self.weights.parameters():
            param.requires_grad = False#requires_grad attribute auf False setzen --> kein .backwards() möglich/aber auch nicht beötigt
                                        # spart computation power

    def return_parameters(self):
        #returns current model weights for model
        return [p for p in self.weights.parameters()]

    def set_parameters(self, weights__):
        #function to set new weights to model
        for i, param in enumerate(self.weights.parameters()):#iterieren über die modell parameter
            new_param = weights__[i]#aus weights das entsprechende gewicht auslesen
            for p in range(param.shape[0]):#über jede stelle in parameter iterieren
                param[p] = new_param[p]#stelle in parameter durch stelle von neuem parameter auswechseln

    def mutate_individuum(self):
        #verändern / mutieren der modell parameter mit einem normalverteilten random vektor
        return [p+torch.randn(p.shape)/100 for p in self.weights.parameters()]

class Generation(object):

    def __init__(self, num_individuums = 100, pretrained = False):

        '''
        class to manage generation generation 
        num_individuums: amount of individuums in one generation (int)
        pretrained: Boolean if generation is randomly initialized (False) or weights from former
                    generations are used (True)
        weights: need to be assigned when pretrained = True, list containing objects of class RobotIndividuum
        '''

        self.num_ind = num_individuums#anzahl de rindividuen in einer generation
        self.pretrained = pretrained#werden die individuen neu initialisiert oder aus einer vorherigen generation zusammengebaut?
        self.individuums = list()#list containing all individuums of this generation

    def generate_individuums(self, weights = None):
        #check if weights from former generations are used
        if self.pretrained:
            assert str(weights) != 'None', 'if you wanna use already trained individuuals and not randomly generated weights, assign weights'
            
            while len(self.individuums) != self.num_ind:
                #self production enabled, da individuum auch mit sich selbst gekreuzt werden kann
                for trained_ind in weights:
                    indx_list = [_ for _ in range(len(weights))]
                    partner = weights[random.choice(indx_list)]#randomly take a partner from the pretrained individuums

                    #kopie der zu kreuzenden partner anlegen
                    cross_ind1 = deepcopy(trained_ind)
                    cross_ind2 = deepcopy(partner)

                    individuum = self.cross_individuums(cross_ind1, cross_ind2)
                    #create new "child" robot individuum
                    self.individuums.append(individuum)#add child as new individuum for Generation() object


        else:
            for i in range(self.num_ind):#solange neue individuen in liste hinzufügen, bis num_ind erfüllt ist
                self.individuums.append(RobotIndividuum())

    def cross_individuums(self, inda, indb):
        #zuföllig bestimmen, wie stark jedes der individuen mit in das neue "child" individuum einfließt
        a_weight = random.randint(0,10)
        b_weight = 10 - a_weight
        comb_ind = RobotIndividuum()

        #mutieren der beiden eltern individuen
        ## ist eine inplace funktion, da mutate_individuum eine self funktion ist
        ## returned eine liste der parameter, über die dann iteriert werden kann
        inda = inda.mutate_individuum()
        indb = indb.mutate_individuum()

        #über parameter des kindindividuums iterieren
        for i, param in enumerate(comb_ind.weights.parameters()):
            w_ = (a_weight*inda[i]+indb[i]*b_weight)/10#mitteln der parameter der eltern (sind ja schon mutiert)
            for p in range(param.shape[0]):#ersetzen der ursprünglichen parameter durch die neuen parameter
                param[p] = w_[p]

        return comb_ind

def fitness_function(model):
    fitness = 0
    count_steps = 0

    while robot.step(timestep) != -1:
        count_steps += 1

        poleVal = polePos.getValue()
        pole_abs = poleVal if poleVal >= 0 else (-1)*poleVal#betrag der poleposition

        f_ = abs(pole_abs*(10/count_steps))#je mehr steps gemacht werden, desto kleiner wird der wert --> fit wird minimiert
        fitness += f_ - (count_steps - pole_abs)/2#minimierungsproblem --> es wird mehr abgezogen, wenn PoleVal klein ist und count steps groß

        with torch.no_grad():
            out = model(torch.Tensor([poleVal])).cpu().numpy()
            w1= out[0]#durch tanh auf bereich [-1,1]
            w1 = w1*10#maximal mögliche geschwindigkeit ist 10, bzw -10 --> deswegen multiplikationsfaktor

            #alle reifen auf gleiche geschwindigkeit setzen 
            motor1.setVelocity(float(w1))
            motor2.setVelocity(float(w1))
            motor3.setVelocity(float(w1))
            motor4.setVelocity(float(w1))

        if pole_abs >= 1.2:
            break
    return fitness

num_best_ind = 2
num_generations = 1000
elitism = True #if elitism should be used --> take best model from generation in next generation
save_generations = [100, 250, 500, 750, 1000]
load_pretrained = True#laden eines bis zu einem gewissen punkt trainierten models?

start_epochs = 0
for k in range(start_epochs, num_generations):
    if k == 0:
        gen = Generation()#initialize one generation of robots (initial generation of robots)
        gen.generate_individuums()#randomly generate individuums

        fitnesses = {}#dict in form {robot_id: fitness}
        for id, individuum in tqdm(enumerate(gen.individuums)):
            robot.simulationReset()
            m = individuum.weights#auslesen des modells des roboters
            fit_ = fitness_function(m)
            fitnesses[id] = fit_
            #print(id+1, 'robots finished')

        print(f'===================== Gen {k+1} ===========================')
        print('best fit: ', min(fitnesses.values()))
        print('worst fit: ', max(fitnesses.values()))
        print('durchschnittliche fitness:', np.mean(list(fitnesses.values())))
    
    else:
        best_ind = dict()
        for _ in range(num_best_ind):#take as many "best" individuums as specified
            b_ = min(fitnesses, key=fitnesses.get)#find key for individuum with best (minimal) fitness
            if _ == 0:
                best_ind_idx = b_#abspeichern des indices des besten individuums aus dieser generation
            
            best_ind[b_] = gen.individuums[b_]#save best individuum in list (object of class RobotIndividuum)
            del fitnesses[b_]

        if elitism:
            best_indi = deepcopy(gen.individuums[best_ind_idx])
            #deepccopy ist hier wichtig, weil die mutation inplace auf dem objekt durchgeführt wird
            # bestes individuum soll unverändert / ohne mutation in neue generation hinzugefügt werden

        #erzeugen einer neuen generation, basierend auf den aus vorgeneration bekannten individuen
        gen = Generation(pretrained=True)
        gen.generate_individuums(weights=list(best_ind.values()))

        if elitism:
            gen.individuums.append(best_indi)#hinzufügen des besten individuums aus vorgeneration in neue generation
            #print(f"Anzahl Individuen in Generation: {len(gen.individuums)}")

        fitnesses = {}#dict in form {robot_id: fitness}
        for id, individuum in tqdm(enumerate(gen.individuums)):
            robot.simulationReset()
            m = individuum.weights#auslesen des modells des roboters
            fit_ = fitness_function(m)
            fitnesses[id] = fit_
            #print(id+1, 'robots finished')
        print(f'===================== Gen {k+1} ===========================')
        print('best fit: ', min(fitnesses.values()))
        print('worst fit: ', max(fitnesses.values()))
        print('durchschnittliche fitness:', np.mean(list(fitnesses.values())),'\n\n')

    if k+1 in save_generations:
        best_model = gen.individuums[min(fitnesses, key=fitnesses.get)]
        PATH = str(time.time())+'_'+str(k+1)+'.pt'
        torch.save(best_model.weights.state_dict(), PATH)

# Enter here exit cleanup code.


