import argparse
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


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        help='Dataset to train on')
    parser.add_argument('--input_nc', default=3, type=int, metavar='N',
                        help='number of input channels')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
    #                     choices=model_names,
    #                     help='model architecture: ' +
    #                         ' | '.join(model_names) +
    #                         ' (default: resnet18)')
    # parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
    #                     help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    # optimizer parameters for SGD
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    #lr parameters
    parser.add_argument('--lr_policy', default='step', type=str,
                        metavar='L', help='lr_policy')
    parser.add_argument('--stepsize_lr', default=30, type=int, metavar='N',
                        help='step size for step decayed lr policy')


    # parser.add_argument('-p', '--print-freq', default=10, type=int,
    #                     metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save_frequency', default=2, type=int, metavar='N',
                            help='Frequency of saving in number of period')
    parser.add_argument('--save_path', default='model', type=str, metavar='PATH',
                            help='Path to save models')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
    #                     help='evaluate model on validation set')
    # parser.add_argument('--pretrained', dest='pretrained', action='store_true',
    #                     help='use pre-trained model')
    # parser.add_argument('--world-size', default=-1, type=int,
    #                     help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist-backend', default='nccl', type=str,
    #                     help='distributed backend')
    # parser.add_argument('--seed', default=None, type=int,
    #                     help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    # parser.add_argument('--multiprocessing-distributed', action='store_true',
    #                     help='Use multi-processing distributed training to launch '
    #                          'N processes per node, which has N GPUs. This is the '
    #                          'fastest way to use PyTorch for either single node or '
    #                          'multi node data parallel training')

    args = parser.parse_args()
    return args
