from all import Model, dataloaders, run_test
model = Model("/storage/resnet_xray_fitted.pt");
run_test(model, dataloaders)
