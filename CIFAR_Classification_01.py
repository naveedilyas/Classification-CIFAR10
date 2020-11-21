import torch
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import Compose, ToTensor
from matplotlib import pyplot
import torch.nn as nn
from torch.optim import SGD
from torchvision.transforms import ToTensor,Normalize
from numpy import vstack
from numpy import argmax
#from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision


# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
load_model = False
# Data Preperation
def prepare_data():
    trans = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train = CIFAR10(root='./data',train= True, download=True,transform=trans)
    test = CIFAR10(root='./data',train= False,download=True,transform=trans)
    train_dl = DataLoader(train,batch_size=64,shuffle=True)
    test_dl = DataLoader(test,batch_size=4,shuffle=False)
    return train_dl,test_dl

# Saving Checkpoints
def save_checkpoint(state,filename = 'my_checkpoint.pth.tar'):
    print("Saving checkpoints")
    torch.save(state,filename)
    
# Loading Checkpoints
def load_checkpoint(checkpoint):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    
# Model
class Convnet(nn.Module):
    def __init__(self, in_channel):
        super(Convnet,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,X):
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = X.view(-1,16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        return X

model = Convnet(3).to(device)
num_epoch = 6

#Training Model
if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))
def train_model(train_dl,model):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    n_total_steps = len(train_dl)
    #print(n_total_steps)
    for epoch in range(num_epoch):
        if epoch % 2== 0:
            checkpoint = {'state_dict': model.state_dict(),'optimizer':optimizer.state_dict()}
            save_checkpoint(checkpoint)

        for i, (inputs,targets) in enumerate(train_dl):
            inputs = inputs.to(device)
            targets = targets.to(device)
            #print(targets)
            # Forward Pass
            yhat = model(inputs)
            loss = criterion(yhat,targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (i+1) % 2 == 0:
            print("Epoch:",epoch+1/num_epoch,"Step:", i+1/n_total_steps, "Loss:",loss.item())

    #print('Finished Training')
    #PATH = './cifar_net.pth'
    #torch.save(model.state_dict(),PATH)

#Evaluation Model    
def evaluate(test_dl, model):
    predictions, actuals = list(),list()
    n_samples = 0
    n_correct = 0
    for i, (inputs,targets) in enumerate(test_dl):
        inputs =inputs.to(device)
        targets= targets.to(device)

        yhat = model(inputs)

        #yhat = yhat.detach().numpy()
        #actual = targets.numpy()
        _, yhat = torch.max(yhat,axis=1)
        n_samples+= targets.size(0)
        n_correct+= (yhat == targets).sum().item()


    acc = 100.0*n_correct/n_samples
    return acc


#path = '~/.torch/datasets/mnist'
train_dl,test_dl = prepare_data()
#print ('==>>> total trainning batch number: {}'.format(len(train_dl)))
#print ('==>>> total testing batch number: {}'.format(len(test_dl)))
#print(len(train_dl.dataset), len(test_dl.dataset))
#cnn = CNN(1)
train_model(train_dl,model)
acc = evaluate(test_dl,model)
print('Accuracy',acc)
#print(len(train_dl.dataset), len(test_dl.dataset))


'''


'''
path = '~/.torch/dataset/mnist'
trans = Compose([ToTensor()])
train = MNIST(path, train=True, download=True,transform=trans)
test = MNIST(path,train=False,download=True,transform=trans)

train_dl = DataLoader(train,batch_size=32,shuffle=True)
test_dl = DataLoader(test,batch_size=32,shuffle=False)
i, (inputs, targets) = next(enumerate(train_dl))
print(type(targets))
for i in range(1):
    pyplot.plot(i)
    pyplot.imshow(inputs[i][0],cmap='gray')

pyplot.show()
'''
