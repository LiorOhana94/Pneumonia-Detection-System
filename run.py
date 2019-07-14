from all import Model, dataloaders 
model = Model()
model_ft = model.fit(dataloaders, 1) 
run_test(model_ft, dataloaders)