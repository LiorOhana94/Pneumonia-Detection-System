from dataloaders import dataloaders
import torch
import torch.nn as nn
import torchvision
from cam.network.net import VGG, make_layers

epochs = 200
model_names = [f'best_loss_vgg19_v2_200e', f'best_acc_vgg19_v2_200e']

def run_test(model, dataloaders, model_name):
    train_on_gpu = torch.cuda.is_available()
    criterion = nn.NLLLoss()
    model.eval()   
    running_loss = 0.0
    running_corrects = 0
    items_num = 0


    for inputs, labels in dataloaders['test']['loader']:
        if train_on_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
            model.cuda()

       
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
        
        items_num += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    f = open("/storage/tests_results/test_res_%s.txt" % model_name,"w+")
    f.write("Test Results: we got {0} right out of {1}, ({2:.2f}%)".format(running_corrects, items_num, float(running_corrects)/items_num))
    f.close()
    return

# almog is in the house


for name in model_names:
    model = torch.load(f'/storage/models/{name}.model')
    run_test(model, dataloaders, f'{name}_after_load')
