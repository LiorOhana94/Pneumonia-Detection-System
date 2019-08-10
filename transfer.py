from all import Model, dataloaders, run_test
import torch

model = Model('/storage/resnet_pretrained.pth', True)
fitted_model = model.fit(dataloaders, 50)

torch.save(fitted_model, "/storage/resnet_xray_transfered_new.model")

