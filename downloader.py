import os
from torchvision.datasets import MNIST, FashionMNIST

dir = os.path.join(os.getcwd(), 'data')

MNIST(dir, train=True, download=True)
MNIST(dir, train=False, download=True)
FashionMNIST(dir, train=True, download=True)
FashionMNIST(dir, train=False, download=True)