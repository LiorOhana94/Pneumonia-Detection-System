import Model, dataloaders from all
model = Model()
model_ft = model.fit(dataloaders, 1) 
run_test(model_ft, dataloaders)