from dataloaders import dataloaders
import torch
import torch.nn as nn

def run_test(model, dataloaders, model_name, testfile_suffix = ''):
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
    precision = 0
    recall = 0
    f = open(f"/storage/tests_results/{model_name}_{testfile_suffix}_test_results.txt" ,"w+")

    for inputs, labels in dataloaders['test']['loader']:
        if train_on_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
            model.cuda()

       
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
        
        TP += torch.sum(preds - labels.data*2 == -1)
        FP += torch.sum(preds - labels.data*2 == 1)
        TN += torch.sum(preds - labels.data*2 == 0)
        FN += torch.sum(preds - labels.data*2 == -2)

        f.write(f"{TP}, {FP}, {TN}, {FN}\n")


        items_num += inputs.size(0)
        accuracy = float(running_corrects)/items_num
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    recall = float(TP.tolist())/(TP.tolist() + FN.tolist())
    precision = float(TP.tolist())/(TP.tolist() + FP.tolist())
    f1_score = 2*(recall * precision) / (recall + precision)
    f.write(f"Test Results: we got {running_corrects} right out of {items_num}\n")
    f.write(f"Accuracy : {accuracy :.2f}\n")
    f.write(f"Recall : {recall :.2f}\n")
    f.write(f"Precision : {precision :.2f}\n")
    f.write(f"F1 Score : {f1_score :.2f}\n")
    f.close()
    return

def compare_after_loading(model, dataloaders, model_name):
    torch.save(model ,f'/storage/models/{model_name}.model')
    loaded_model = torch.load(f'/storage/models/{model_name}.model')
    run_test(loaded_model, dataloaders, model_name, 'after_load')
    return


run_test
