import os
import PIL
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

model = torch.load(os.path.join("resnet_xray_fitted.model"), map_location='cpu')


test_loader = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(model, image_path):
    train_on_gpu = False # torch.cuda.is_available()
    model.eval()   
    
    image = image_loader(test_loader, image_path)
    with torch.no_grad():
        if train_on_gpu:
            image = image.cuda()
            model.cuda()
        
        outputs = model(image)
        return outputs


test_loader = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


    
def image_loader(loader, image_name):
    grey_image = Image.open(image_name)
    grey_image_array = np.array(grey_image)
    image = Image.fromarray(to_rgb(grey_image_array))
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def to_rgb(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret