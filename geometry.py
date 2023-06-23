import numpy as np
from itertools import product
from scipy.spatial import distance_matrix

import torch
import torch.nn as nn


################################################################################
################################################################################
################################################################################
### GEOMETRIC RNN

class GRNN(nn.Module):
    def __init__(self, celltype, Wg):
        pass

    def forward(self):
        pass


class GRNNCell(nn.Module):
    """
    A RNN that works with an implicit geometric interaction matrix underlying
    the interactions between neurons. Very basic, works considering the Elman
    scheme.
    """
    def __init__(self, input_size, hidden_size, nl, Wg):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nl = nl
        # See parameter modifications
        self.Wp = nn.Parameter(torch.rand(hidden_size, input_size,
                                          requires_grad = True))
        self.bp = nn.Parameter(torch.rand(hidden_size), requires_grad = True)
        self.Wg = nn.Parameter(torch.Tensor(Wg), requires_grad = False)
        self.bg = nn.Parameter(torch.rand(hidden_size), requires_grad = False)

    def forward(self, x, h = None):
        if not h:
            hp = torch.zeros(self.hidden_size)
            hg = torch.zeros(self.hidden_size)
        else:
            hp, hg = h
        last_hp = hp
        hp = self.nl(self.Wp@x + self.bp + self.Wg@hg + self.bg)
        hg = torch.tanh(torch.abs(last_hp + 0.5)) + 1
        return (hp, hg)


class GGRUCell(nn.Module):
    """
    A cell of a Geometric RNN which uses the GRU method. Basically, botht the
    geometric and synaptic parts are different GRU cells. The synaptic works
    (almost) as a standard GRU, while the geometric works as a standard GRU
    but with the output of the synaptic as its input. The output of the
    geometric is then used to finish computing the output of the synaptic.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 Wg,
                 h0 = None,
                 bias = False,
                 alpha = 0.5,
                 optim = None):
        super().__init__()
        # Store size of things
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Create each individual cell
        self.synaptic = nn.GRUCell(input_size, hidden_size, bias)
        self.geometric = nn.GRUCell(hidden_size, hidden_size, bias)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_size,
                                           2,
                                           bias)
                                )

        # Last things
        self.Wg = Wg
        if h0 is None:
            self.hp = torch.zeros((hidden_size * 1, ))
            self.hg = torch.zeros((hidden_size * 1, ))
        self.init_params()
        self.alpha = alpha
        if optim is None:
            self.optim = torch.optim.SGD

    def init_params(self):
        i = torch.eye(self.hidden_size)
        self.geometric.weight_ih = nn.Parameter(torch.cat((i,
                                                           i,
                                                           i)),
                                                requires_grad = False)
        self.geometric.weight_hh = nn.Parameter(torch.cat((self.Wg,
                                                           self.Wg,
                                                           self.Wg)),
                                                requires_grad = False)

    def forward(self, input_):
        h_hat = self.synaptic.forward(input_, self.hp)
        self.hg = self.geometric.forward(h_hat, self.hg)
        self.hp = (1 - self.alpha)*h_hat + self.alpha*self.hg
        self.o = self.mlp(self.hp)
        return (self.hp, self.hg, self.o)


    def backwards(self):
        optim1 = self.optim(self.synaptic.params(), lr = 0.01)
        optim2 = self.optim(self.mlp.params(), lr = 0.01)





################################################################################
################################################################################
################################################################################
### GEOMETRIC INTERACTION MATRICES
def make_planar_weightmatrix(len_x:int, len_y:int,
                              long_range_inhibition = False):
    '''
    A function that makes a planar weight matrix of the interactions of cells
    distributed in a plane. All cells have the same separation between them.
    Input:
    - len_x: length along the x axis
    - len_y: length along the y axis
    - long_range_inhibition: boolean denoting if the function for calculating
      the effect of the field will consider long range inhibition. True by
      default.

    Output:
    - matrix with interaction weights.

    Note:
    - A diffusion constant should probably be implemented to account for the
      difference in values between the two functions for long range inhibition.
      Will do later.
    '''
    combinations = list(product(range(len_x), range(len_y)))
    weights = distance_matrix(combinations, combinations)
    weights = LFP(weights)
    for ii in range(weights.shape[0]):
        weights[ii, ii] = 0
    return torch.Tensor(weights)


def input_vector_generator(t, grid, label = False, long = False):
    """
    Creates a random input vector with t entries. The first two elements of the
    input vector are the target (only shown at a random time between start and
    stop) and the start/stop signals (shown at random moments throughout the
    test). The rest is a stretched out version of input grid, which will only
    be shown when the stop signal appears.

    Input:
    - t: duration of the task. For the timing of the signal to be realistic, t
      needs to be at least 240.
    - grid: n-dimensional representation of search space, encoding for the
      coordinates of the elements to be found.
    - label: boolean. Determines if the desired output needs to be shown.
    - long: boolean. Determines if the output signal should show the expected
      decision until the end or not. Defaults to False.

    Output:
    - tuple containing the input vector, and the label. None if label = False.
    """
    # Randomly determine start, end, and signal times
    start = torch.randint(low = 20, high = t // 3 - 20, size = (1, )).item()
    end = torch.randint(low = 2*(t // 3) + 10, high = t - 30, size = (1, ))
    end = end.item()
    signal = torch.randint(low = start + 50, high = end - 39, size = (1, 1))
    signal = signal.item()

    # Determine the value to be shown and its output (if needed)
    max_ = int(torch.max(grid))
    min_ = int(torch.min(grid))
    target = torch.randint(low = min_ + 1, high = max_ + 1, size = (1, ))
    if label:
        coordinates = (grid == target).nonzero()
        out = torch.zeros(2, t)
        if long:
            for tt in range(t - end):
                out[:, end + tt] = coordinates
        else:
            out[:, end] = coordinates
    else:
        coordinates = None

    grid = grid.flatten()
    input_vector = torch.zeros(len(grid) + 2, t)
    input_vector[0, signal:signal+10] = target
    input_vector[1, start:start+10] = 1
    input_vector[1, end:end+10] = 1
    input_vector[2:, end] = grid
    return (input_vector, out)









if __name__ == '__main__':
    w = _make_planar_weightmatrix(2, 3)
    print(w)

