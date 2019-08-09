from all import Model, dataloaders, run_test
import torch

for i in range(5):
    model = Model()
    model_ft = model.fit(dataloaders, 42, i + 1) 
    run_test(model_ft, dataloaders)
    torch.save(model_ft, "/storage/resnet_xray_fitted_lr%d.model" % (i + 1))
