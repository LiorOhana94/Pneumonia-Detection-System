import torch
import torchvision
from predict import predictResnet, predict


m_resnet50 = torchvision.models.resnet50(pretrained=False, num_classes=2)
m_vgg19 = torchvision.models.vgg19(pretrained=False, num_classes=2)

#predict(m_vgg19, './temp-images/person1_bacteria_1.jpeg', 'HI', generate_map=True)
predictResnet(m_resnet50, './temp-images/person1_bacteria_1.jpeg', 'HI', generate_map=True)