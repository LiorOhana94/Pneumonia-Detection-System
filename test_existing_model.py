from dataloaders import dataloaders
import torch
import torch.nn as nn
import torchvision
from cam.network.net import VGG, make_layers
from test import run_test

model = torch.load('/storage/models/best.model', map_location='cpu')

run_test(model, dataloaders, 'my_new_test_on_model')