import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
from skimage.transform import resize
from skimage.io import imshow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from test import run_test

from torch.autograd import Variable
from torchvision import datasets, transforms

from dataloaders import dataloaders
from cam.network.net import VGG, make_layers
from cam.network.utils import Flatten, accuracy, imshow_transform, SaveFeatures

def vgg19():
    model = VGG(make_layers())
    state_dict = torch.load('/storage/vgg19_pretrained_dictstate.pth')
    model.load_state_dict(state_dict, strict=False)
    return model

model = vgg19()
# model.cuda()


#freeze layers
for param in model.parameters():
    param.requires_grad = False

#modify the last two convolutions
model.features[-5] = nn.Conv2d(512,512,3, padding=1)
model.features[-3] = nn.Conv2d(512,2,3, padding=1)

#remove fully connected layer and replace it with AdaptiveAvePooling
model.classifier = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1),Flatten(),
                                nn.LogSoftmax()
                                )

"""
#Create datasets and dataloaders
input_size=224
batch_size = 16
train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

dataset_train = datasets.ImageFolder(root='hymenoptera_data/train/',
                                    transform=train_transforms)

dataset_valid = datasets.ImageFolder(root='hymenoptera_data/val/',
                                    transform=valid_transforms)



train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                                          shuffle=True,
                                          num_workers=1)

valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, 
                                          shuffle=True,
                                          num_workers=1)
"""

criterion = nn.CrossEntropyLoss()
lr =.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.cuda()

mean_train_losses = []
mean_val_losses = []
mean_train_acc = []

mean_val_acc = []
minLoss = 99999
maxValacc = -99999
num_epochs = 100
model_name = f"resnet19_transferred_v2_{num_epochs}e"

f = open("/storage/trainlogs/log_%s.txt" % model_name,"w+")

for epoch in range(num_epochs):
    f.write(f'EPOCH: {epoch+1}\n')
    
    train_acc = []
    val_acc = []
    
    running_loss = 0.0
    
    model.train()
    count = 0
    for images, labels in dataloaders['train']['loader']:        
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(images) 
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        train_acc.append(accuracy(outputs, labels))
        
        
        loss.backward()
        optimizer.step()        
        
        running_loss += loss.item()
        count +=1
    f.write('Training loss:  %d %s' % (running_loss/count, '\n'))
    mean_train_losses.append(running_loss/count)
        
    model.eval()
    count = 0
    val_running_loss = 0.0
    for images, labels in dataloaders['val']['loader']:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(images)
        loss = criterion(outputs, labels)

        val_acc.append(accuracy(outputs, labels))
        val_running_loss += loss.item()
        count +=1

    mean_val_loss = val_running_loss/count
    f.write('Validation loss:  %d %s' % (mean_val_loss, '\n'))

    f.write('Training accuracy:  %d %s' % (np.mean(train_acc), '\n'))
    f.write('Validation accuracy:  %d %s' % (np.mean(val_acc), '\n'))
    
    mean_val_losses.append(mean_val_loss)
    
    mean_train_acc.append(np.mean(train_acc))
    
    val_acc_ = np.mean(val_acc)
    mean_val_acc.append(val_acc_)
    
   
    if mean_val_loss < minLoss:
        torch.save(model, f'/storage/models/best_loss_vgg19_v2_{num_epochs}e.model' )
        f.write(f'NEW BEST Val Loss: {mean_val_loss} ........old best:{minLoss}\n')
        minLoss = mean_val_loss
        
    if val_acc_ > maxValacc:
        torch.save(model, f'/storage/models/best_acc_vgg19_v2_{num_epochs}e.model' )
        f.write(f'NEW BEST Val Acc: {val_acc_} ........old best:{maxValacc}\n')
        maxValacc = val_acc_

run_test(model, dataloaders, model_name)
f.write('training complete.')
f.close()