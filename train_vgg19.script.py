import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
from skimage.transform import resize
from skimage.io import imshow
"""
from test import compare_after_loading

import torch
import torch.nn as nn
import torch.nn.functional as F
from test import run_test

from torch.autograd import Variable
from torchvision import datasets, transforms

from dataloaders import dataloaders
from cam.network.net import VGG, make_layers
from cam.network.utils import Flatten, accuracy, imshow_transform, SaveFeatures

# Almog is in the house

# ----- Training Configuration ----- #

num_epochs = 150
lr =.0001
wd =.075
model_name = f"resnet19_v3_{num_epochs}e_{lr}lr_imbsam"
class_weights = torch.Tensor([0.5, 1.0])
# ---------------------------------- #

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

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
model.cuda()

mean_train_losses = []
mean_val_losses = []
mean_train_acc = []

mean_val_acc = []
minLoss = 99999
maxValacc = -99999
model_name = f"resnet19_{num_epochs}e_{lr}lr_{wd}wd"

f = open("/storage/trainlogs/log_%s.txt" % model_name,"w+")

train_accs = []
train_losses = []
val_accs = []
val_losses = []

for epoch in range(num_epochs):
    f.write(f'EPOCH: {epoch+1}\n')
    

    
    running_loss = 0.0
    running_corrects = 0
    model.train()
    count = 0

    for images, labels in dataloaders['train']['loader']:        
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(images) 
        
        optimizer.zero_grad()

        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

        
        loss.backward()
        optimizer.step()        
        
        running_loss += loss.item()
        count +=1

    f.write('Training loss:  %d %s' % (running_loss/count, '\n'))
    f.write('Training acc:  %d %s' % (running_corrects.double() / dataloaders['train']['length'], '\n'))
    train_losses.append(running_loss/count)
    train_accs.append(running_corrects.double() / dataloaders['train']['length'])

    model.eval()
    count = 0
    val_running_loss = 0.0
    val_running_corrects = 0

    for images, labels in dataloaders['val']['loader']:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        val_running_corrects += torch.sum(preds == labels.data)
        loss = criterion(outputs, labels)

        val_running_loss += loss.item()
        count +=1

    val_acc = val_running_corrects.double()/ dataloaders['val']['length']
    val_accs.append(val_acc)
    val_losses.append(val_running_loss/count)
    
    f.write('Validation loss:  %d %s' % (val_running_loss/count, '\n'))
    f.write('Validation accuracy:  %d %s' % (val_acc, '\n'))    
    
    mean_val_loss = val_running_loss/count
   
    if val_acc > maxValacc:
        torch.save(model, f'/storage/models/{model_name}.model' )
        f.write(f'NEW BEST Val Acc: {val_acc} old best:{maxValacc}\n')
        maxValacc = val_acc
        best_model = model

plt.figure()
plt.plot(train_accs, '-p')
plt.plot(val_accs, '-g')
plt.savefig(f'/storage/trainlogs/{model_name}_accfig.png')

plt.figure()
plt.plot(train_losses, '-b')
plt.plot(val_losses, '-g')
plt.savefig(f'/storage/trainlogs/{model_name}_lossfig.png')


f.write('training complete.')
f.close()
compare_after_loading(model, dataloaders, model_name)
