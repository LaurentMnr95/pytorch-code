import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image


def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

def FGSM(net, x,y, criterion=F.cross_entropy, targeted=False, eps=0.03, x_val_min=0, x_val_max=1):
        """Return FGSM attack for pytorch. This attack can be applied to a batch of input data.
        Parameters:
            net: network to be attacked (nn.Module)
            x: input to be fooled
            y: true label or target
            criterion: criterion to minimize/maximize
            targeted: Boolean targeted attack or not
            eps: size of attack
            x_val_min: minimal value of input
            x_val_max: maximal value of input
        Returns:
            x_adv: adversarial image
            h_adv: prediction for adversarial example  x_adv
            h:  prediction for original input x
        """
        x_adv = Variable(x.data, requires_grad=True)
        h_adv = net(x_adv)
        if targeted:
            cost = criterion(h_adv, y)
        else:
            cost = -criterion(h_adv, y)

        net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)


        h = net(x)
        h_adv = net(x_adv)

        return x_adv, h_adv, h

def iterFGSM(net, x, y, criterion=F.cross_entropy, targeted=False, eps=0.03, alpha=0.01, iteration=1, x_val_min=0, x_val_max=1):
    """Return iterFGSM attack for pytorch. This attack can be applied to a batch of input data.
    Parameters:
        net: network to be attacked (nn.Module)
        x: input to be fooled
        y: true label or target
        criterion: criterion to minimize/maximize
        targeted: Boolean targeted attack or not
        eps: limit size of attack
        alpha: iteration step
        iteration: number of iterations
        x_val_min: minimal value of input
        x_val_max: maximal value of input
    Returns:
        x_adv: adversarial image
        h_adv: prediction for adversarial example  x_adv
        h:  prediction for original input x
    """
    x_adv = Variable(x.data, requires_grad=True)
    for i in range(iteration):
        h_adv = net(x_adv)
        if targeted:
            cost = criterion(h_adv, y)
        else:
            cost = -criterion(h_adv, y)

        net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - alpha*x_adv.grad
        x_adv = where(x_adv > x+eps, x+eps, x_adv)
        x_adv = where(x_adv < x-eps, x-eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    h = net(x)
    h_adv = net(x_adv)

    return x_adv, h_adv, h
