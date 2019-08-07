from all import Model, dataloaders, run_test
import torch
model = Model("/storage/resnet_pretrained.pth")
model_ft = model.fit(dataloaders, 50) 
run_test(model_ft, dataloaders)
torch.save(model_ft, "/storage/resnet_xray_fitted.model")