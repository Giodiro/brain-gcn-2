
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from vade_pytorch import VaDE
import argparse


class FlattenTransform(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample.view(-1)


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.002, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
args = parser.parse_args()


transform = transforms.Compose([transforms.ToTensor(), FlattenTransform()])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('....//data/mnist', train=True, download=True, transform=transform),
    batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('....//data/mnist', train=False, transform=transform),
    batch_size=args.batch_size, shuffle=True, num_workers=2)

vade = VaDE(input_dim=784, z_dim=10, n_centroids=10, encoder_dims=[500,500,2000], decoder_dims=[2000,500,500])

#vade.pretrain(num_epochs=10, tr_loader=train_loader)
vade.initialize_gmm(train_loader)

vade.fit(args.epochs, train_loader, test_loader)
