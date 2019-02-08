import torch
from resnet import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from visdom import Visdom
from options_classifier import *
from argparse import ArgumentParser


# TODO: add options on:
# -architecture
# -optimizer
# -multigpu
# -resume training

class  options:
    def __init__(self):
        self.dataset = "CIFAR10"
        if self.dataset == "MNIST":
            self.input_nc = 1 # num of input channel
        else:
            self.input_nc = 3 # num of input channel

        self.ngpu = 1 # num of gpus to train on
        self.batch_size = 128 # size of batch train
        self.epoch = 20 # number of training epochs
        self.save_path = "CIFAR_resnet18" # save path to model
        self.save_frequency = 1 # save every 2 epochs
        # self.visdom_port = 8097
        # self.visdom_hostname= "http://localhost"

# define options
opt  = get_args()
opt.save_path
# defining device

if torch.cuda.is_available():
    print("GPU found: device set to cuda:0")
    device = torch.device("cuda:{}".format(opt.gpu))
else:
    print("No GPU found: device set to cpu")
    device = torch.device("cpu")

transform = transforms.ToTensor()
# Load inputs
if opt.dataset == "CIFAR10":
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform),
            batch_size=opt.batch_size, shuffle=True)
    print("Loaded CIFAR 10 dataset")
elif opt.dataset == "MNIST":
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform),
            batch_size=opt.batch_size, shuffle=True)
    print("Loaded MNIST dataset")

num_images = len(train_loader.dataset)
# Transform input to [-1,1]



# Classifier  definition
Classifier = resnet34(opt.input_nc)

Classifier.to(device)
print("Classifier intialized")
print(Classifier)
Classifier.train()


# optimizer and criterion
criterion = torch.nn.CrossEntropyLoss().cuda(opt.gpu)
optimizer = torch.optim.SGD(Classifier.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)


if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        Classifier.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

# for param_group in optimizer.param_groups:
#     param_group['lr']=opt.lr
#     param_group['momentum']=opt.momentum
#     param_group['weight_decay']=opt.weight_decay

# initialize Visdom
# viz = Visdom(port=opt.visdom_port, server=opt.visdom_hostname)

for epoch in range(opt.start_epoch,opt.epochs):  # loop over the dataset multiple times
    current_num_input = 0
    running_loss = 0.0
    running_acc= 0
    for i, data in enumerate(train_loader, 0):

        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = Classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)

        # print statistics
        running_loss += loss.item()
        current_num_input += len(labels)
        with torch.no_grad():
            running_acc += (predicted==labels).double().sum().item()/len(labels)
        if i % 20 == 19:
            # print every 20 mini-batches
            print("Epoch :[", epoch+1,"/",opt.epochs,
                    "] [",current_num_input,"/",num_images,
                    "] Running loss:",running_loss/20,
                    ", Running accuracy:",running_acc/20)
            running_loss = 0.0
            running_acc = 0
    if (epoch +1) % opt.save_frequency == 0:
        path_to_save = os.path.join(opt.save_path,"epoch"+str(epoch+1))

        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        save_dict={'epoch': epoch + 1,
                'state_dict': Classifier.state_dict(),
                'optimizer' : optimizer.state_dict()}
        torch.save(save_dict,path_to_save)
