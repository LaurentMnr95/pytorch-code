import torch
#from resnet import *
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import numpy as np
import os
from visdom import Visdom
from options_classifier import *
from argparse import ArgumentParser
from utils import *
from networks import *
import sys
import time
import copy

# TODO:
# -architecture: code utils.get_model
# -optimizer: code utils.get_optimizer
# -multigpu
# -scheduler lr: linear and others
# add option on parallel, add option on testshows

# define options
opt  = options_train()

opt.val_test = True
DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "localhost"
viz = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)


# if opt.gpu is not None and torch.cuda.is_available():
#     print("GPU found: device set to cuda:0")
#     device = torch.device("cuda:{}".format(opt.gpu))
# else:
#     print("No GPU found: device set to cpu")
#     device = torch.device("cpu")



# Load inputs
train_loader = load_data(opt)
if opt.val_test == True:
    optn=copy.copy(opt)
    optn.batch_size= 100
    test_loader = load_data(optn,train_mode=False)

num_images = len(train_loader.dataset)
num_images_test = len(test_loader.dataset)
num_classes = opt.num_classes





# Classifier  definition
Classifier,filename = getNetwork(opt)
Classifier.apply(conv_init)
#.to(device)
if opt.gpu:
    Classifier.cuda()
    Classifier = torch.nn.DataParallel(Classifier,device_ids=range(torch.cuda.device_count()))#,device_ids)
    cudnn.benchmark =True
print("Classifier intialized")
print(Classifier)


# optimizer and criterion
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(Classifier.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
scheduler = get_scheduler(optimizer,opt)

# resume learning
if opt.resume == True:
    path_to_load = os.path.join(opt.save_path,filename+'_'+opt.dataset,opt.resume_name)

    if os.path.isfile(path_to_load):
        print("=> loading checkpoint '{}'".format(path_to_load))
        checkpoint = torch.load(path_to_load)
        opt.start_epoch = checkpoint['epoch']
        Classifier.load(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path_to_load))



sys.stdout.flush()
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
opts_visdom={}
 # loop over the dataset multiple times

for epoch in range(opt.start_epoch,opt.epochs):
    current_num_input = 0

    running_loss = 0.0
    running_acc= 0

    training_loss = 0
    training_acc = 0
    best_acc = -1

    Classifier.train()
    start_time_epoch = time.time()
    for i, data in enumerate(train_loader, 0):

        # get the inputs
        inputs, labels = data
        if opt.gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

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
        training_loss += loss.item()
        running_acc += predicted.eq(labels.data).cpu().sum().numpy()
        training_acc += predicted.eq(labels.data).cpu().sum().numpy()
        curr_batch_size=inputs.size(0)
        if i % 20 == 19:
            # print every 20 mini-batches
            print("Epoch :[", epoch+1,"/",opt.epochs,
                    "] [",i*opt.batch_size,"/",num_images,
                    "] Running loss:",running_loss/20,
                    ", Running accuracy:",running_acc/(20*curr_batch_size)," time:",time.time()-start_time_epoch)
            running_loss = 0.0
            running_acc = 0
            sys.stdout.flush()



    X.append(epoch)
    Y_loss[0].append(training_loss/num_images)
    Y_acc[0].append(training_acc/num_images)
    data_loss[0] = {'x': X,'y':Y_loss[0],'name':'Train','type':'line'}
    data_acc[0] = {'x': X,'y': Y_acc[0],'name':'Train','type': 'line'}

    if opt.val_test:
        test_loss = 0
        test_acc = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if opt.gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = Classifier(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_acc += predicted.eq(targets.data).cpu().sum().numpy()
        
        Y_loss[1].append(test_loss/num_images_test)
        Y_acc[1].append(test_acc/num_images_test)

        data_loss[1] = {'x': X,'y':Y_loss[1],'name':'Test','type':'line'}
        data_acc[1] = {'x': X,'y': Y_acc[1],'name':'Test','type': 'line'}

        viz._send({'data': data_loss, 'win': win_loss, 'eid': env, 'layout': layout_loss,'opts':opts_visdom})
        viz._send({'data': data_acc, 'win': win_acc, 'eid': env, 'layout': layout_acc,'opts':opts_visdom})
    else:
        viz._send({'data': [data_loss[0]], 'win': win_loss, 'eid': env, 'layout': layout_loss,'opts':opts_visdom})
        viz._send({'data': [data_acc[0]], 'win': win_acc, 'eid': env, 'layout': layout_acc,'opts':opts_visdom})
    
    # save model
    if (epoch +1) % opt.save_frequency == 0:
        path_to_save = os.path.join(opt.save_path,filename+'_'+opt.dataset)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        state={
                'epoch': epoch + 1,
                'net': Classifier.module if opt.gpu else Classifier,
                }
        torch.save(state,os.path.join(path_to_save,"epoch"+str(epoch+1)+'.t7'))

    if opt.val_test and test_acc > best_acc:
        state = {
                'net':Classifier.module if opt.gpu else Classifier,
                'epoch':epoch+1,
                }
        path_to_save = os.path.join(opt.save_path,filename+'_'+opt.dataset)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        torch.save(state,os.path.join(path_to_save,"BEST.t7"))
        test_acc = best_acc

        

    scheduler.step()
