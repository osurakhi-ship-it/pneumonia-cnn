import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

model.load_state_dict(torch.load("weights/resnet18_weighted_best.pth", map_location=DEVICE))

model.eval()
print("Model loaded successfully.")
