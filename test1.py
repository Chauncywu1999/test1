import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from IPython.display import display


data_path = './datasets/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
cifar10_val = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

img, label = cifar10[99]
plt.imshow(img.permute(1, 2, 0))
plt.show()
