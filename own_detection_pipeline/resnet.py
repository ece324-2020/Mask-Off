import torch.nn as nn
from torchvision import *


def get_resnet():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(pretrained=True)
    model = model.cuda() if device else model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model.fc = model.fc.cuda() if device else model.fc
    ct = 0
    for name, child in model.named_children():
        if ct < 4:
            for name2, params in child.named_parameters():
                params.requires_grad = False
        ct += 1

    return model
