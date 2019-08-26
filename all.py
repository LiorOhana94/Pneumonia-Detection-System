import PIL
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn, optim
import time
import copy


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std*inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
inputs,classes = next(iter(dataloaders["train"]))
out = torchvision.utils.make_grid(inputs)
class_names = dset["train"].classes
imshow(out, title = [class_names[x] for x in classes])



def run_test(model, dataloaders, model_name):
    train_on_gpu = torch.cuda.is_available()
    criterion = nn.NLLLoss()
    since = time.time()
    model.eval()   
    running_loss = 0.0
    running_corrects = 0
    items_num = 0


    for inputs, labels in dataloaders['test']:
        if train_on_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
            model.cuda()

       
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
        
        items_num += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    f = open("/storage/test_res_%s.txt" % model_name,"w+")
    f.write("Test Results: we got {0} right out of {1}, ({2:.2f}%)".format(running_corrects, items_num, float(running_corrects)/items_num))
    f.close()
    return



