import torch
from resnet import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from visdom import Visdom

class  options:
    def __init__(self):
        self.input_nc = 1 # num of input channel
        self.ngpu = 1 # num of gpus to train on
        self.batch_size = 128 # size of batch train
        self.epoch = 10 # number of training epochs
        self.load_path = "MNIST_resnet18/epoch10" # save path to model
        self.save_frequency = 1 # save every 2 epochs
        # self.visdom_port = 8097
        # self.visdom_hostname= "http://localhost"

# define options
opt  = options()
epoch =1
# defining device

if opt.ngpu>=1:
    if torch.cuda.is_available():
        print("GPU found: device set to cuda:0")
        device = torch.device("cuda:0")
    else:
        print("No GPU found: device set to cpu")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")


# Load inputs
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=opt.batch_size, shuffle=True)

num_images = len(test_loader.dataset)
# Classifier  definition
Classifier = resnet18(opt.input_nc)
Classifier.load_state_dict(torch.load(opt.load_path))
Classifier.to(device)


# Transform input to -1 1
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




running_acc = 0
for i, data in enumerate(test_loader, 0):

    # get the inputs
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    # zero the parameter gradients

    # forward + backward + optimize
    outputs = Classifier(inputs)

    _, predicted = torch.max(outputs.data, 1)

    with torch.no_grad():
        running_acc += (predicted==labels).sum().item()

print("Accuracy on test data:",running_acc/num_images)
