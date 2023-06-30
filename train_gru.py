import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LocationDataset
from random import shuffle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

class Control(nn.Module):
    def __init__(self, input_size, hidden_size, bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = hidden_size // 2
        self.model = nn.Sequential(nn.Linear(self.input_size,
                                             self.hidden_size,
                                             bias),
                                   nn.Tanh(),
                                   nn.GRUCell(self.hidden_size,
                                              self.hidden_size,
                                              bias),
                                   nn.Linear(self.hidden_size,
                                             self.out_size,
                                             bias),
                                  )

    def step(self, input_):
        return model(input_)

    def forward(self, input_):
        with torch.autograd.set_detect_anomaly(True):
            out = self.model(input_)
        return out


def train(model, lossfn, optimizer):
    size = 500
    picks = list(range(size))
    shuffle(picks)
    while picks:
        idx = picks.pop()
        in_ = torch.load(f'./data/training/inputs/vector_{idx}.pt')
        out = torch.load(f'./data/training/labels/label_{idx}.pt')
        tend = out.shape[0]
        history = torch.zeros_like(out)
        for tt in range(tend):
            prediction = model(in_[:, tt])
            history[tt, :, :] = prediction.reshape((9, 9))
        loss1 = lossfn(history[out == 0], out[out == 0])
        loss2 = lossfn(history[out != 0], out[out != 0])
        loss = loss1 + loss2
        print(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.h = torch.zeros_like(model.h)
    return loss.item()


print('Define Model')
model = Control(9, 81 * 2, False)

E = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr = 0.01)

epochs = 100
loss_epoch = np.zeros(epochs)
for e in range(epochs):
    print(f'Epoch: {e}')
    fe = train(model, E, optim)
    print(fe)
    loss_epoch[e] = fe

torch.save(model, f'GRU_training_{epochs}_epochs.pt')

fig, ax = plt.subplots()
ax.plot(range(1, epochs + 1), loss_epoch)
ax.set_xlabel('Epoch')
ax.set_ylabel('Error')
ax.set_ylim((0, max(loss_epoch) * 1.1))
fig.suptitle('Error per epoch')

print('Look at me!')
plt.show()

