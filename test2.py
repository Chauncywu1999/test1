import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from IPython.display import display
from collections import OrderedDict

# data reading
wine_path = "./winequality-white.csv"
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=";", skiprows=1)
col_list = next(csv.reader(open(wine_path), delimiter=";"))
wine_properties = wineq_numpy[:, :-1]
wine_quality = wineq_numpy[:, -1:]


# data processing creating the train set and the validation set
n_samples = wine_properties.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_wine_properties = torch.tensor(wine_properties[train_indices])
val_wine_properties = torch.tensor(wine_properties[val_indices])

train_wine_quality = torch.tensor(wine_quality[train_indices]).long()
val_wine_quality = torch.tensor(wine_quality[val_indices]).long()

train_wine_quality_encode = torch.zeros((train_wine_quality.shape[0], 10))
val_wine_quality_encode = torch.zeros((val_wine_quality.shape[0], 10))

train_wine_quality_encode.scatter_(1, train_wine_quality - 1, 1.0)
val_wine_quality_encode.scatter_(1, val_wine_quality - 1, 1.0)

# create the model by nn
seq_model = nn.Sequential(OrderedDict([('hidden_linear', nn.Linear(11, 20)),
                                       ('hidden_activation', nn.Tanh()), ('output_linear', nn.Linear(20, 10))]))


# create the optimizer
optimizer = optim.SGD(seq_model.parameters(), lr=1e-4)


# define the training loop function
def training_loop(n_epochs, this_optimizer, model, loss_fn, this_train_wine_properties,
                  this_val_wine_properties, this_train_wine_quality, this_val_wine_quality):
    for epoch in range(1, n_epochs + 1):
        train_wine_quality_p = model(this_train_wine_properties)
        loss_train = loss_fn(train_wine_quality_p, this_train_wine_quality)

        val_wine_quality_p = model(this_val_wine_properties)
        loss_val = loss_fn(val_wine_quality_p, this_val_wine_quality)

        this_optimizer.zero_grad()
        loss_train.backward()
        this_optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
                  f"  Validation loss {loss_val.item():.4f}")


training_loop(n_epochs=5000, this_optimizer=optimizer, model=seq_model, loss_fn=nn.MSELoss(),
              this_train_wine_properties=train_wine_properties, this_val_wine_properties=val_wine_properties,
              this_train_wine_quality=train_wine_quality_encode, this_val_wine_quality=val_wine_quality_encode)


