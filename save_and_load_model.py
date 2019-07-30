import torch
from torch import nn
import torchvision

def save():
	m = torchvision.models.resnet152(pretrained=True)
	torch.save(m.state_dict(), "resnet_pretrained.pth")
	return

def load():
	model = torchvision.models.resnet50(pretrained=False)
	model.load_state_dict(torch.load("resnet_pretrained.pth"), strict=False)
	return model

