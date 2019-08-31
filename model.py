import torch

model = torch.load('./storage/models/best_loss_vgg19_v2_100e.model', map_location='cpu')
