import torch
from torch import nn
from torch import optim
from PIL import Image
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import ast


vgg16 = models.vgg16(pretrained=True)
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
models = {'vgg16': vgg16, 'resnet': resnet18, 'alexnet': alexnet}


def build_model(model_name, learning_rate, hidden_units, gpu):
    if gpu == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif gpu == 'cpu':
        device = torch.device("cpu")
    else:
        print("Invalid Input")

        
    model = models[model_name]
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))

    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    
    return model, device, optimizer