from dataloaders import dataloaders
import torch
import torch.nn as nn

def run_test(model, dataloaders, model_name, testfile_prefix = ''):
    train_on_gpu = torch.cuda.is_available()
    criterion = nn.NLLLoss()
    model.eval()   
    running_loss = 0.0
    running_corrects = 0
    items_num = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for inputs, labels in dataloaders['test']['loader']:
        if train_on_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
            model.cuda()

       
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
        
        TP += torch.sum(preds == labels.data and preds == 1)
        FP += torch.sum(preds != labels.data and preds == 1)
        TN += torch.sum(preds == labels.data and preds == 0)
        FN += torch.sum(preds != labels.data and preds == 0)

        accuracy = float(running_corrects)/items_num
        recall = float(TP)/(TP + FN)
        precision = float(TP)/(TP + FP)
        items_num += inputs.size(0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    f = open(f"/storage/tests_results/{testfile_prefix}_test_res_{model_name}.txt" ,"w+")
    f.write(f"Test Results: we got {running_corrects} right out of {items_num}, ({accuracy :.2f}%)\n")
    f.write(f"Accuracy : {accuracy :.2f}%")
    f.write(f"recall : {recall :.2f}%")
    f.write(f"precision : {precision :.2f}%")
    f.close()
    return

def compare_after_loading(model, dataloaders, model_name):
    run_test(model, dataloaders, model_name, 'before_load')
    torch.save(model ,f'/storage/models/{model_name}.model')
    loaded_model = torch.load(f'/storage/models/{model_name}.model')
    run_test(loaded_model, dataloaders, model_name, 'after_load')
    return


run_test
