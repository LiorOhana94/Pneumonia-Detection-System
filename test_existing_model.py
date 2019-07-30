from all import Model, dataloaders, run_test
import torch

model = torch.load("/storage/resnet_xray_fitted.model");
run_test(model, dataloaders)
