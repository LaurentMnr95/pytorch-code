
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import nevergrad
from nevergrad.optimization import optimizerlib
import numpy as np




def DFOattack(net, x,y, criterion=F.cross_entropy, targeted=False, eps=0.1, x_val_min=0, x_val_max=1, optimizer="DE"):

    x=x.to("cuda:0")

    def convert_individual_to_image(individual):
        perturbation = torch.from_numpy((eps*np.tanh(individual)).astype(np.float32))
        perturbation = perturbation.view(3,32,32)
        perturbation = perturbation.to("cuda:0")

        x_adv = x + perturbation
        return x_adv

    def loss(individual):
        netx_adv = net(convert_individual_to_image(individual))
        return float(-criterion(netx_adv,y))

    optimizer = optimizerlib.registry[optimizer](dimension=32*32*3,budget=1000, num_workers=1)
    x_adv = convert_individual_to_image(optimizer.optimize(loss))


    x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
    h_adv = net(x_adv)

    return x_adv, h_adv
