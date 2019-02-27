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
opt.val_test = True
DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "localhost"
viz = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)

# defining device
# TODO 
# change device gestion to multiple GPUs 
# use visdom
# Look at test accuracy at -
if torch.cuda.is_available():
    print("GPU found: device set to cuda:0")
    device = torch.device("cuda:{}".format(opt.gpu))
else:
    print("No GPU found: device set to cpu")
    device = torch.device("cpu")



# Load inputs
train_loader = load_data(opt)
if opt.val_test == True:
    optn=opt
    optn.batch_size = 10000
    test_loader = load_data(opt,train_mode=False)
    dataiter = iter(test_loader)
    images_test, labels_test = dataiter.next()
    images_test, labels_test = images_test.to(device), labels_test.to(device)

num_images=len(train_loader.dataset)
# Classifier  definition
Classifier = resnet34(opt.input_nc)
Classifier.to(device)
print("Classifier intialized")
print(Classifier)
Classifier.train()


# optimizer and criterion
criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda(opt.gpu)
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
X=[]
Y_loss = [[],[]]
Y_acc = [[],[]]

data_loss=[None,None]
data_acc=[None,None]

win_loss= 'loss'
win_acc='acc'

layout_loss= {
                'title':"Loss vs epoch",
                'xaxis':{'title':'Epoch'},
                'yaxis':{'title':'Loss'}
}
layout_acc= {
                'title':"Accuracy vs Epoch",
                'xaxis':{'title':'Epoch'},
                'yaxis':{'title':'Acc'}
}
env='main'
opts={}
 # loop over the dataset multiple times

for epoch in range(opt.start_epoch,opt.epochs):
    current_num_input = 0

    running_loss = 0.0
    running_acc= 0

    training_loss = 0
    training_acc = 0
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
        current_num_input = len(labels)
        training_loss += (loss.item()*current_num_input)

        with torch.no_grad():
            running_acc += (predicted==labels).double().sum().item()/len(labels)
            training_acc += (predicted==labels).double().sum().item()

        if i % 20 == 19:
            # print every 20 mini-batches
            print("Epoch :[", epoch+1,"/",opt.epochs,
                    "] [",i*current_num_input,"/",num_images,
                    "] Running loss:",running_loss/20,
                    ", Running accuracy:",running_acc/20)
            running_loss = 0.0
            running_acc = 0


    # save model
    if (epoch +1) % opt.save_frequency == 0:
        path_to_save = os.path.join(opt.save_path,opt.model_name)

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        save_dict={'epoch': epoch + 1,
                'state_dict': Classifier.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
        torch.save(save_dict,os.path.join(path_to_save,"epoch"+str(i+1)))


    X.append(epoch)
    Y_loss[0].append(training_loss/num_images)
    Y_acc[0].append(training_acc/num_images)
    data_loss[0] = {'x': X,'y':Y_loss[0],'name':'Train','type':'line'}
    data_acc[0] = {'x': X,'y': Y_acc[0],'name':'Train','type': 'line'}

    if opt.val_test == True:
        outputs_test = Classifier(images_test)
        loss_test = criterion(outputs_test, labels_test)

        _, predicted = torch.max(outputs_test.data, 1)
        test_acc = (predicted==labels_test).double().mean().item()
        test_loss = loss_test.item()

        Y_loss[1].append(test_loss)
        Y_acc[1].append(test_acc)

        data_loss[1] = {'x': X,'y':Y_loss[1],'name':'Test','type':'line'}
        data_acc[1] = {'x': X,'y': Y_acc[1],'name':'Test','type': 'line'}

        viz._send({'data': data_loss, 'win': win_loss, 'eid': env, 'layout': layout_loss,'opts':opts})
        viz._send({'data': data_acc, 'win': win_acc, 'eid': env, 'layout': layout_acc,'opts':opts})
    else:
        viz._send({'data': [data_loss[0]], 'win': win_loss, 'eid': env, 'layout': layout_loss,'opts':opts})
        viz._send({'data': [data_acc[0]], 'win': win_acc, 'eid': env, 'layout': layout_acc,'opts':opts})
    scheduler.step()
