import torch
from resnet import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from visdom import Visdom
from attacks.FGSM import *


class  options:
    def __init__(self):
        self.dataset = "CIFAR10"
        if self.dataset == "MNIST":
            self.input_nc = 1 # num of input channels
        else:
            self.input_nc = 3 # num of input channels

        self.ngpu = 1 # num of gpus to train on
        self.batch_size = 64 # size of batch test
        self.epoch = 1 # number of training epochs
        self.load_path = "model/epoch50" # save path to model
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
transform = transforms.ToTensor()
if opt.dataset == "CIFAR10":
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform),
            batch_size=opt.batch_size, shuffle=False)
    print("Loaded CIFAR 10 dataset")
elif opt.dataset == "MNIST":
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform),
            batch_size=opt.batch_size, shuffle=False)
    print("Loaded MNIST dataset")

num_images = len(test_loader.dataset)
# Classifier  definition
Classifier = resnet34(opt.input_nc)
Classifier.load_state_dict(torch.load(opt.load_path)['state_dict'])
Classifier.to(device)
print("Classifier intialized")
print(Classifier)
Classifier.eval()



# Testing
running_acc = 0
running_acc_adv = 0
for i, data in enumerate(test_loader, 0):

    # get the inputs
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    # zero the parameter gradients

    # forward + backward + optimize
    outputs = Classifier(inputs)

    _, predicted = torch.max(outputs.data, 1)
    _, predicted_adv, _ = FGSM(Classifier, inputs ,labels, eps=0.1, x_val_min=0, x_val_max=1)
    _, predicted_adv=torch.max( predicted_adv.data,1)

    with torch.no_grad():
        running_acc += (predicted==labels).double().sum().item()
        running_acc_adv +=(predicted_adv==labels).double().sum().item()

print("Accuracy on test data for natural images:",running_acc/num_images)
print("Accuracy on test data for adversarial images:",running_acc_adv/num_images)
