import PIL
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn, optim
import time
import copy

class Model(nn.Module):
    def __init__(self, load_path = None, transfer = False, pretrained = False):
        super(Model, self).__init__()
        
        self.transfered = transfer
        self.gradients = None

        self.model = torchvision.models.vgg16(pretrained=pretrained)
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path), strict=False)
        
        if transfer:
            self.classifier = nn.Sequential(
                nn.Linear(self.model.classifier[6].in_features, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 2), nn.LogSoftmax(dim=1))

            for params in self.model.parameters():
                params.requires_grad = False

            self.model.classifier[6] = self.classifier
        else:
            for params in self.model.parameters():
                params.requires_grad = True


    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.model.features(x)
#        x = self.model.features_conv(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        print("hook!")
        # apply the remaining pooling
        x = self.model.max_pool(x)
        x = x.view((1, -1))
        x = self.model.classifier(x)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.model.features_conv(x)

    def fit(self, dataloaders, num_epochs, step_size=4):
        f= open("/storage/fit_run_lr%d_%s.txt" % (step_size, "transfered" if self.transfered else ""),"w+")

        train_on_gpu = torch.cuda.is_available()
        # Check:
        optimizer = optim.Adam(self.model.classifier[6].parameters()) 

        # Check:
        #Essentially what scheduler does is to reduce our learning by a certain factor when less progress is being made in our training.
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size)
        #criterion is the loss function of our model. we use Negative Log-Likelihood loss because we used  log-softmax as the last layer of our model. We can remove the log-softmax layer and replace the nn.NLLLoss() with nn.CrossEntropyLoss()
        criterion = nn.NLLLoss()
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        if train_on_gpu:
            self.model = self.model.cuda()
        for epoch in range(num_epochs):
            f.write('Epoch {}/{}'.format(epoch, num_epochs - 1))
            f.write('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train()  
                else:
                    self.model.eval()  

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase].loader:
                    if train_on_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataloaders[phase].length
                epoch_acc = running_corrects.double() / dataloaders[phase].length
                f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(
                    phase, epoch_loss, epoch_acc))
                    
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
        time_elapsed = time.time() - since
        f.write('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        f.write('Best val Acc: {:4f}'.format(best_acc))
        f.close()
        self.model.load_state_dict(best_model_wts)
        return self.model