import torch
import torch.nn as nn
from torchvision import models

def get_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    return model
