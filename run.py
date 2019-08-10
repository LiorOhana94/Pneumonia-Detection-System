from all import Model, dataloaders, run_test
import torch


for i in range(4):
    model = Model()
    lr = i + 2
    fitted_model = model.fit(dataloaders, 42, lr) 
    run_test(fitted_model, dataloaders)
    torch.save(fitted_model, "/storage/resnet_xray_fitted_lr%d.model" % (lr))
