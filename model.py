import torch

model = torch.load('./current_model/best.model', map_location='cpu')
