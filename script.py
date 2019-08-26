import torch

from model import Model
from predict import predict
from layer_activation import LayerActivations

model = torch.load("./storage/vgg16_xray_transfered_new.model")
res = predict(model, './temp-images/0ba74bfa-7d5d-4147-a2f0-afe9e0375b04.jpeg')
acts = LayerActivations(model) 
out_features = acts.features[0].squeeze(0)
print('bye')
