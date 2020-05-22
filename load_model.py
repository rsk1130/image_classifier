import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as py



def load_model(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == "vgg16":
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == "resnet18":
        model = models.resnet18(pretrained=True)
    elif checkpoint['arch'] == "alexnet":
        model = models.alexnet(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model