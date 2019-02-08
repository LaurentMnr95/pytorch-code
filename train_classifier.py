import torch
from resnet import *
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from visdom import Visdom
from options_classifier import *
from argparse import ArgumentParser
from utils import *

# TODO:
# -architecture: code utils.get_model
# -optimizer: code utils.get_optimizer
# -multigpu
# -scheduler lr: linear and others
# -visdom (when at FB)

# define options
opt  = get_args()


# defining device
# TODO change device gestion
if torch.cuda.is_available():
    print("GPU found: device set to cuda:0")
    device = torch.device("cuda:{}".format(opt.gpu))
else:
    print("No GPU found: device set to cpu")
    device = torch.device("cpu")



# Load inputs
train_loader = load_data(opt)
num_images=len(train_loader.dataset)
# Classifier  definition
Classifier = resnet34(opt.input_nc)
Classifier.to(device)
print("Classifier intialized")
print(Classifier)
Classifier.train()


# optimizer and criterion
criterion = torch.nn.CrossEntropyLoss().cuda(opt.gpu)
optimizer = torch.optim.SGD(Classifier.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
scheduler = get_scheduler(optimizer,opt)


# resume learning
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        Classifier.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


# initialize Visdom
# viz = Visdom(port=opt.visdom_port, server=opt.visdom_hostname)

 # loop over the dataset multiple times
for epoch in range(opt.start_epoch,opt.epochs):
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

    # save model
    if (epoch +1) % opt.save_frequency == 0:
        path_to_save = os.path.join(opt.save_path,"epoch"+str(epoch+1))

        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)
        save_dict={'epoch': epoch + 1,
                'state_dict': Classifier.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
        torch.save(save_dict,path_to_save)

    scheduler.step()
