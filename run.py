from all import Model, dataloaders, run_test
import torch


for i in range(4):
    model = Model()
    lr = i + 2
    fitted_model = model.fit(dataloaders, 42, lr) 
    run_test(fitted_model, dataloaders, "vgg16_xray_fitted_lr%d.model" % (lr))
    torch.save(fitted_model, "/storage/vgg16_xray_fitted_lr%d.model" % (lr))
