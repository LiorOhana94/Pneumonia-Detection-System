from all import Model, dataloaders, run_test
import torch

model = Model(None, True)
fitted_model = model.fit(dataloaders, 50)

torch.save(fitted_model, "/storage/vgg16_xray_transfered_new.model")