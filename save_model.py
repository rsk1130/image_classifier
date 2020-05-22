import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def save_model(model, pretrained, epochs, optimizer, save_directory, image_dir):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                           transforms.RandomResizedCrop(224), 
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(), 
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    train_dir = image_dir + '/train'
    trainset = datasets.ImageFolder(train_dir, transform=train_transforms)
    model.class_to_idx = trainset.class_to_idx
    
    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': pretrained,
              'classifier': model.classifier,
              'epochs': epochs,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, (save_directory + '/checkpoint.pth'))
    
    print("\nModel Saved!")