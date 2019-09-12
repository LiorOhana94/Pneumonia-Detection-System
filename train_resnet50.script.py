import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
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
import torchvision
from torchvision import datasets, transforms

from dataloaders import dataloaders
from cam.network.utils import Flatten, accuracy, imshow_transform, SaveFeatures

# Almog is in the house

# ----- Training Configuration ----- #
date = datetime.datetime.now()
time_str = date.strftime("%m%d%H%M")
num_epochs = 36
lr =.0001
wd =.000
loss='nll'
class_weights = [1.0, 1.0]
model_name = f"{time_str}_res50v2_{num_epochs}e_{loss}loss_{lr}lr_{wd}wd_cw{class_weights}"
class_weights = torch.Tensor(class_weights)
class_weights = class_weights.cuda()
# ---------------------------------- #



model = torchvision.models.resnet50(pretrained=False, num_classes=2)


#freeze layers
#for param in model.parameters():
#    param.requires_grad = False


criterion_first = nn.NLLLoss(weight=class_weights)
criterion_second = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
model.cuda()

mean_train_losses = []
mean_val_losses = []
mean_train_acc = []

mean_val_acc = []
minLoss = 99999
maxValacc = -99999

f = open("/storage/trainlogs/%s_trainlog.txt" % model_name,"w+")

train_accs = []
train_losses = []
val_accs = []
val_losses = []
best_model = None

for epoch in range(num_epochs):
    f.write(f'EPOCH: {epoch+1}\n')
    

    
    running_loss = 0.0
    running_corrects = 0
    model.train()
    count = 0
    train_len = 0

    for images, labels in dataloaders['train']['loader']:        
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(images) 
        
        optimizer.zero_grad()
        loss = 0
        if epoch < num_epochs/2:
            loss = criterion_first(outputs, labels)
        else:
            loss = criterion_second(outputs, labels)
            
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

        train_len += len(images)
        loss.backward()
        optimizer.step()        
        
        running_loss += loss.item()
        count +=1
    train_acc = running_corrects.double() / train_len
    f.write(f'Training loss: {running_loss/count}\n')
    f.write(f'Training accuracy: {train_acc}\n')    
    train_losses.append(running_loss/count)
    train_accs.append(train_acc)

    model.eval()
    count = 0
    val_len = 0
    val_running_loss = 0.0
    val_running_corrects = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    precision = 0
    recall = 0

    for images, labels in dataloaders['val']['loader']:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        val_running_corrects += torch.sum(preds == labels.data)
        loss = 0

        if epoch < num_epochs/2:
            loss = criterion_first(outputs, labels)
        else:
            loss = criterion_second(outputs, labels)


        val_running_loss += loss.item()
        count +=1
        val_len += len(images)
        TP += torch.sum(preds - labels.data*2 == -1)
        FP += torch.sum(preds - labels.data*2 == 1)
        TN += torch.sum(preds - labels.data*2 == 0)
        FN += torch.sum(preds - labels.data*2 == -2)

    val_acc = val_running_corrects.double()/ val_len
    val_accs.append(val_acc)
    val_losses.append(val_running_loss/count)
    
    recall = float(TP.tolist())/(TP.tolist() + FN.tolist() + 0.1)
    precision = float(TP.tolist())/(TP.tolist() + FP.tolist() + 0.1)
    f1_score = 2*(recall * precision) / (recall + precision  + 0.1)

    f.write(f'Validation loss: {val_running_loss/count}\n')
    f.write(f'Validation accuracy: {val_acc}\n')    
    f.write(f"Validation Recall : {recall :.2f}\n")
    f.write(f"Validation Precision : {precision :.2f}\n")
    f.write(f"Validation F1 Score : {f1_score :.2f}\n")
    mean_val_loss = val_running_loss/count
   
    if val_acc > maxValacc:
        torch.save(model, f'/storage/models/{model_name}.model' )
        f.write(f'NEW BEST Val Acc: {val_acc} old best:{maxValacc}\n')
        maxValacc = val_acc
        best_model = model
    f.write("###-----------------------------------------------------------------###\n")
    
torch.save(model, f'/storage/models/{model_name}_last.model' )

plt.figure(figsize=(8, 6), dpi=60)
plt.plot(train_accs, '-p')
plt.plot(val_accs, '-g')
plt.ylim(bottom=0)
plt.savefig(f'/storage/trainlogs/{model_name}_accfig.png')

plt.figure(figsize=(8, 6), dpi=60)
plt.plot(train_losses, '-b')
plt.plot(val_losses, '-g')
plt.savefig(f'/storage/trainlogs/{model_name}_lossfig.png')


f.write('training complete.')
f.close()
compare_after_loading(best_model, dataloaders, model_name)
