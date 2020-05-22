import torch
from torch import nn
from torch import optim
from PIL import Image
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import numpy as py
import time
import ast
from workspace_utils import active_session




def classify(model, device, optimizer, image_dir, epochs):
    
    
    train_dir = image_dir + '/train'
    valid_dir = image_dir + '/valid'
    test_dir = image_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                           transforms.RandomResizedCrop(224), 
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(), 
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    vt_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validset = datasets.ImageFolder(valid_dir, transform=vt_transforms)
    testset = datasets.ImageFolder(test_dir, transform=vt_transforms)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64)

    criterion = nn.NLLLoss()
    
    
    with active_session():
        
        print("\nTraining and Validating...")
        
        epochs = epochs
        train_losses, test_losses = [], []

        for epoch in range(epochs):
            running_loss = 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
        
                optimizer.zero_grad()
        
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
    
            else:
                test_loss = 0
                accuracy = 0
        
                with torch.no_grad():
                    model.eval()
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        test_loss += criterion(logps, labels)
                
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(validloader))
            
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/len(trainloader):.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
    print("Finished!")