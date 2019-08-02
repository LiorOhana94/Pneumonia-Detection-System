import os
import PIL
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torch

model = torch.load(os.path.join("storage", "resnet_xray_fitted.model"))


test_loader = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(model, image_path):
    train_on_gpu = torch.cuda.is_available()
    model.eval()   
    
    image = image_loader(test_loader, image_path)

    print(image)
    if train_on_gpu:
        image = image.cuda()
        model.cuda()
    
    outputs = model(image)
    return outputs

    
def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image
