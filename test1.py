import os
import torch
import numpy as np
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from IPython.display import display

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

# set minibatch to train set
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

# set minibatch to validation set
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

# create model
model = nn.Sequential(nn.Linear(3072, 1024),
                      nn.Tanh(),
                      nn.Linear(1024, 512),
                      nn.Tanh(),
                      nn.Linear(512, 128),
                      nn.Tanh(),
                      nn.Linear(128, 2))

# set learning rate
learning_rate = 1e-2

# set SGD as optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# set loss function
loss_fn = nn.CrossEntropyLoss()

# set epochs
n_epochs = 156

# specify the device as "cuda"
device = torch.device("cuda")

# move your model to the device
model.to(device)

loss = None
loss_values = []
for epoch in range(n_epochs):
    for imgs, labels in train_loader:
        # move data to the device
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs.view(imgs.shape[0], -1))
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_values.append(float(loss))
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

plt.plot(np.arange(1, len(loss_values) + 1), loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# compute the Accuracy
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        # move data to the device
        imgs = imgs.to(device)
        labels = labels.to(device)

        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

print("Accuracy: %f" % (correct / total))


