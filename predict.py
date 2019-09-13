import os
import PIL
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np

from skimage.transform import resize
from skimage.io import imshow, imsave

import matplotlib.pyplot as plt

from cam.network.utils import SaveFeatures, imshow_transform

test_loader = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# VGG19:
def predict(model, image_path, scan_guid, generate_map=False):
    train_on_gpu = False # torch.cuda.is_available()
    model.eval()   
    model.cpu()
    image = image_loader(test_loader, image_path)
    with torch.no_grad():
        if train_on_gpu:
            image = image.cuda()
            model.cuda()
        
        if generate_map:
            sf = SaveFeatures(model.features[-3])

        outputs = model(image)

        if generate_map:
            create_heatmap_file(sf, outputs, image, scan_guid)
        
        res = torch.argmax(outputs.data).cpu().detach().numpy()
        prob = torch.softmax(outputs, 1)

        return res, torch.max(prob)


def create_heatmap_file(sf, outputs, image, scan_guid):
    sf.remove()
    arr = sf.features.cpu().detach().numpy()
    features_data = arr[0]
    res = torch.argmax(outputs.data).cpu().detach().numpy()
    ans = np.dot(np.rollaxis(features_data,0,3), [res, 1 - res])
    ans = resize(ans, (224,224))
    plt.figure()
    plt.subplots(figsize=(4,4))
    plt.imshow(imshow_transform(image))
    plt.imshow(ans, alpha=.4, cmap='jet')
    cam_path = f'./class-activation-maps/{scan_guid}.cam.png'
    plt.axis('off')
    plt.savefig(cam_path)
    plt.close()
    return



# Resnet50:
def predictResnet(model, image_path, scan_guid, generate_map=False):
    train_on_gpu = False # torch.cuda.is_available()
    model.eval()   
    model.cpu()
    image = image_loader(test_loader, image_path)
    with torch.no_grad():
        if train_on_gpu:
            image = image.cuda()
            model.cuda()
        
        if generate_map:
            sf = SaveFeatures(model.layer4[-1])

        outputs = model(image)

        if generate_map:
            create_heatmap_file_resnet(model, sf, outputs, image, scan_guid)
        
        res = torch.argmax(outputs.data).cpu().detach().numpy()
        prob = torch.softmax(outputs, 1)

        return res, torch.max(prob)





def create_heatmap_file_resnet(model, sf, outputs, image, scan_guid):
    sf.remove()
    res = torch.argmax(outputs.data).cpu().detach().numpy()
    features = sf.features[0].cpu().detach().numpy()

    W = model.fc.weight 
    w = W[res,:]

    ans = np.dot(np.rollaxis(features,0,3),w.detach().cpu())

    ans = resize(ans, (224,224))
    plt.figure()
    plt.subplots(figsize=(4,4))
    plt.imshow(imshow_transform(image))
    plt.imshow(ans, alpha=.4, cmap='jet')
    cam_path = f'./class-activation-maps/{scan_guid}.cam.png'
    plt.axis('off')
    plt.savefig(cam_path)    
    plt.close()
    return


def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


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
    ret = im

    if(len(im.shape) < 3):
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = im
        ret[:, :, 1] = im
        ret[:, :, 2] = im

    return ret