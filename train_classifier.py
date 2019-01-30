import torch
from resnet import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
class  options:
    def __init__(self):
        self.input_nc = 1 # num of input channel
        self.ngpu = 1 # num of gpus to train on
        self.batch_size = 128 # size of batch train
        self.epoch = 10 # number of training epochs
        self.save_path = "MNIST_resnet18" # save path to model
        self.save_frequency = 2 # save every 2 epochs
# define generators
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
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=opt.batch_size, shuffle=True)

num_images = len(train_loader.dataset)
# Classifier  definition
Classifier = resnet18(opt.input_nc)

Classifier.to(device)


# Transform input to -1 1
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




# optimizer and criterion
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(Classifier.parameters(), lr=0.001, momentum=0.9)


for epoch in range(opt.epoch):  # loop over the dataset multiple times
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
            running_acc += (predicted==labels).sum().item()/len(labels)
        if i % 20 == 19:
            # print every 20 mini-batches
            print("Epoch :[", epoch+1,"/",opt.epoch,
                    "] [",current_num_input,"/",num_images,
                    "] Running loss:",running_loss/20,
                    ", Running accuracy:",running_acc/20)
            running_loss = 0.0
            running_acc = 0
    if epoch % 2 == 1:
        path_to_save = os.path.join(opt.save_path,"epoch"+str(epoch))

        if os.path.exists(path_to_save):
            os.remove(path_to_save)
            os.makedirs(path_to_save)
        else:
            os.makedirs(path_to_save)
        torch.save(Classifier.state_dict(),path_to_save)
os.exists(path_to_save)
