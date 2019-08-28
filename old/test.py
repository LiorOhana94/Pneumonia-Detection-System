from all import Model, dataloaders, run_test
import torch

model = torch.load("/storage/vgg16_xray_transfered_new.model")
run_test(model, dataloaders, 'my_model')