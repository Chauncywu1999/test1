import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from IPython.display import display


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

# # calculate the mean and std
# imgs = torch.stack([img_t for img_t, _ in cifar10], dim=3)
# display(imgs.view(3, -1).mean(dim=1))
# display(imgs.view(3, -1).std(dim=1))
# imgs_val = torch.stack([img_t for img_t, _ in cifar10_val], dim=3)
# display(imgs_val.view(3, -1).mean(dim=1))
# display(imgs_val.view(3, -1).std(dim=1))


# datasets extract
label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]


# # img test
# img, label = cifar2[3]
# display(img)

# n_out = 2

# create model
model = nn.Sequential(nn.Linear(3072, 512),
                      nn.Tanh(),
                      nn.Linear(512, 2),
                      nn.LogSoftmax(dim=1))

# set learning rate
learning_rate = 1e-2

# set SGD as optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# set loss function
loss_fn = nn.NLLLoss()

# set epochs
n_epochs = 100

loss = None
for epoch in range(n_epochs):
    for img, label in cifar2:
        out = model(img.view(-1).unsqueeze(0))
        loss = loss_fn(out, torch.tensor([label]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
