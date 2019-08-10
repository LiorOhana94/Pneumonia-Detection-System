from all import Model, dataloaders, run_test
import torch

model = torch.load("/storage/resnet_xray_fitted_lr5.model")
run_test(model, dataloaders)
