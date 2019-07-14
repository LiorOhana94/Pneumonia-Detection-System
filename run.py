from all import Model, dataloaders, run_test
model = Model()
model_ft = model.fit(dataloaders, 1) 
run_test(model_ft, dataloaders)