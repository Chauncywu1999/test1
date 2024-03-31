import os
import torch
import datetime
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from IPython.display import display


# create the custom Net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


# create the training loop function
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 100 == 0:
            print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch,
                                                         loss_train / len(train_loader)))


# create a validate function to measuring accuracy
def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)

                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name, correct / total))


# this is a question
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# data loading
data_path = './datasets/'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                              (0.2470, 0.2435, 0.2616))]))
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             transforms.Normalize((0.4942, 0.4851, 0.4504),
                                                                                  (0.2467, 0.2429, 0.2616))]))


# datasets extract
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]

# set minibatch to train set
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

# set minibatch to validation set
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

# check the device on computer
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

# instantiates the Net
model = Net().to(device=device)

# define the optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# define the loss function
loss_fn = nn.CrossEntropyLoss()

# set the training loop parameters
training_loop(n_epochs=1000, optimizer=optimizer, model=model, loss_fn=loss_fn, train_loader=train_loader)

# save the model as a file
model_path = './models/'
torch.save(model.state_dict(), model_path + 'birds_vs_airplanes.pt')

# set the measuring accuracy parameters
validate(model, train_loader, val_loader)
