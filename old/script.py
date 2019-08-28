import torch

from model import Model
from predict import predict
from layer_activation import LayerActivations
e_model = Model()
e_model.model = torch.load("./storage/vgg16_xray_transfered_new.model", map_location='cpu')
e_model.model.eval()
for param in e_model.model.parameters() : param.requires_grad = True
res = predict(e_model, './temp-images/2b48f1b3-34a9-4b6f-b835-29018e551f13.jpeg')
pred = res.argmax(dim=1)

acts = LayerActivations(e_model) 
out_features = acts.features[0].squeeze(0)
print('bye')
