from all import Model, dataloaders, run_test
model = Model("/storage/resnet_xray_fitted.pth");
run_test(model, dataloaders)
