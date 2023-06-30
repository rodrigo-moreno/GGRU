import torch
import torch.nn as nn
import geometry as g
import pickle

import numpy as np
from random import shuffle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def progress(iterator, size):
    current = (size - len(iterator) / size) * 20
    current = round(current)
    desired = '+' * current
    complement = '-' * (20 - current)
    print(f'|{desired}{complement}|', end = '\r')
    pass


def train(model, errorfn, optimizer, epoch, loss_container):
    '''
    Trains model with errorfn using optimizer. The rest is filler for checking
    the status or saving progress files..
    '''
    # Determine amount and order of sampling
    size = 500
    picks = list(range(size))
    shuffle(picks)

    while picks:
        # This can be replaced with geometry.input_vector_generator()
        idx = picks.pop()
        in_ = torch.load(f'./data/training/inputs/vector_{idx}.pt')
        out = torch.load(f'./data/training/labels/label_{idx}.pt')
        tend = out.shape[0]
        history = torch.zeros_like(out)
        #progress(picks, size)

        # Simulate until final timepoint
        for tt in range(tend):
            prediction = model(in_[:, tt])
            history[tt, :, :] = prediction.reshape((9, 9))
        
        # Calculate error
        E1 = errorfn(history[out == 0], out[out == 0])
        E2 = errorfn(history[out != 0], out[out != 0])
        E = E1 + E2
        loss_container[epoch, size - len(picks) - 1] = E

        # Backward step and reset values
        E.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.synaptic.h = torch.rand_like(model.synaptic.h) * 0.05
        model.hp = torch.rand_like(model.hp) * 0.05
        model.hg = torch.rand_like(model.hg) * 0.05
        #print(f'\t{E.item()}')
    return E.item()

# Set un initial conditions
Wg = g.make_planar_weightmatrix(9, 9, True) # This is just an interaction matrix
model = g.SGGRU(9, 81, Wg, final = 'linear')
model.synaptic.h = torch.rand_like(model.synaptic.h) * 0.05
model.hp = torch.rand_like(model.hp) * 0.05
model.hg = torch.rand_like(model.hg) * 0.05
torch.save(model.state_dict(), 'GGRU/base.pt')

E = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr = 0.01)

# Train for all epochs
epochs = 100
loss_epoch = np.zeros((epochs, 500))
for e in range(epochs):
    print(f'Epoch: {e}')
    fe = train(model, E, optim, e, loss_epoch)
    print(fe)
    loss_epoch[e] = fe
    if epochs % 10 == 0:
        torch.save(model.state_dict(), f'GGRU/step_{e}.pt')

with open('GGRU/loss.pckl', 'wb') as f:
    pickle.dump(loss_epoch, f)
torch.save(model.state_dict(), f'GGRU/GGRU_training_{epochs}_epochs.pt')

