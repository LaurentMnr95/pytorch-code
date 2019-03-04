import torch
from resnet import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from visdom import Visdom
from attacks.FGSM import *
#from attacks.DFOattacks import *
from options_classifier import *
from utils import *
import torch.backends.cudnn as cudnn



# define options
opt  = options_test()
# defining device
# TODO change device gestion
# if torch.cuda.is_available():
#     print("GPU found: device set to cuda:0")
#     device = torch.device("cuda:{}".format(opt.gpu))
# else:
#     print("No GPU found: device set to cpu")
#     device = torch.device("cpu")


# Load inputs
test_loader = load_data(opt, train_mode=False)
num_images = len(test_loader.dataset)

# Classifier  definition
#Classifier,filename = getNetwork(opt)
checkpoint = torch.load(opt.path_to_model)
Classifier = checkpoint['net']

if opt.gpu:
    Classifier.cuda()
    Classifier = torch.nn.DataParallel(Classifier,device_ids=range(torch.cuda.device_count()))#,device_ids)
    cudnn.benchmark =True
print("Classifier intialized")
print(Classifier)

Classifier.eval()

# Testing
running_acc = 0
running_acc_adv = 0
for i, data in enumerate(test_loader, 0):


    # get the inputs
    inputs, labels = data
    inputs, labels = inputs.cuda(), labels.cuda()

    # predict
    outputs = Classifier(inputs)

    _, predicted = torch.max(outputs.data, 1)
    #_, outputs_adv = DFOattack(Classifier, inputs ,labels, eps=0.1, x_val_min=0, x_val_max=1)
    #_, predicted_adv=torch.max( outputs_adv.data,1)
    #print()
    #print("predicted adv label",outputs_adv.view(-1)[labels])
    #print("predicted label",outputs.view(-1)[labels])
    #print("true label",labels)
    with torch.no_grad():
        running_acc += (predicted==labels).double().sum().item()
        #running_acc_adv +=(predicted_adv==labels).double().sum().item()

    #print(i)
print("Accuracy on test data for natural images:{}".format(running_acc/num_images))
print("Accuracy on test data for adversarial images:{}".format(running_acc_adv/num_images))
