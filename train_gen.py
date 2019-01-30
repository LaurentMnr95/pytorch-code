import torch
from Generators import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class  options:
    def __init__(self):
        self.input_nc = 1
        self.ngf = 16
        self.netG = 'resnet_1blocks'
        self.ngpu = 1
        opt.batch_size = 128
# define generators
opt  = options()


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





# Generator  definition
Generator = define_G(opt.input_nc, opt.ngf, opt.netG,
                        norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[])

Generator.to(device)


# Transform input to -1 1
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=opt.batch_size, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=4, shuffle=True)

# optimizer and criterion
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(Generator.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = Generator(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()


        # print statistics
        running_loss += loss.item()

        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(test_loader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
with torch.no_grad():
    imshow(torchvision.utils.make_grid(Generator(images)))
