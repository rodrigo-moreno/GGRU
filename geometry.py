import numpy as np
from itertools import product
from scipy.spatial import distance_matrix

import torch
import torch.nn as nn


################################################################################
################################################################################
################################################################################
### GEOMETRIC RNN

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


class SGGRU(nn.Module):
    '''
    An implementation of the Geometric GRU suggested after discussion with
    Alessio.
    '''
    def __init__(self, input_size, hidden_size, Wg, h0 = None, bias = False,
                 optim = None, final = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wg = Wg
        if h0 is None:
            self.hp = torch.zeros(hidden_size)
            self.hg = torch.zeros(hidden_size)
        
        # Declare layers
        self.mlp = nn.Sequential(nn.Linear(self.input_size,
                                           self.hidden_size,
                                           bias),
                                 nn.Tanh(),
                                )
        self.synaptic = SGGRUCell(self.hidden_size,
                                  self.hidden_size,
                                  Wg,
                                  h0)
        self.geometric = nn.GRUCell(self.hidden_size,
                                    self.hidden_size,
                                    bias)
        new_hh = torch.cat((Wg, Wg, Wg), 0)
        self.geometric.weight_hh = nn.Parameter(new_hh, requires_grad = False)
        self.final = final
        if final is not None:
            if final == 'linear':
                self.end_layer = nn.Linear(hidden_size, hidden_size, bias)
            elif final == 'GRU':
                self.end_layer = nn.GRUCell(hidden_size, hidden_size, bias)
            else:
                raise TypeError(f'Layer of type {final} has not been'
                                f'implemented.')

    def forward(self, input_):
        with torch.autograd.set_detect_anomaly(True):
            filter_ = self.mlp(input_)
            self.hp = self.synaptic(filter_, self.hg)
            self.hg = self.geometric(self.hp)
            if self.final:
                out = self.end_layer(self.hp)
                return out
            return (self.hp)


class SGGRUCell(nn.Module):
    '''
    Implementation of a single time instance of the recurrent cell with spatial
    memory.
    '''
    def __init__(self, input_size, hidden_size, Wg, h0 = None, bias = False,
                 optim = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize Parameters
        self.weight_xz = nn.Parameter(torch.rand((hidden_size, input_size)),
                                      requires_grad = True)
        self.weight_hz = nn.Parameter(torch.rand((hidden_size, hidden_size)),
                                      requires_grad = True)
        #self.weight_gpz = nn.Parameter(Wg, requires_grad = False)
        self.weight_gpz = nn.Parameter(torch.rand((hidden_size, hidden_size)),
                                       requires_grad = True)
        self.weight_xr = nn.Parameter(torch.rand((hidden_size, input_size)),
                                      requires_grad = True)
        self.weight_hr = nn.Parameter(torch.rand((hidden_size, hidden_size)),
                                      requires_grad = True)
        #self.weight_gpr = nn.Parameter(Wg, requires_grad = False)
        self.weight_gpr = nn.Parameter(torch.rand((hidden_size, hidden_size)),
                                       requires_grad = True)
        self.weight_xn = nn.Parameter(torch.rand((hidden_size, input_size)),
                                      requires_grad = True)
        #self.weight_gpn = nn.Parameter(Wg, requires_grad = False)
        self.weight_gpn = nn.Parameter(torch.rand((hidden_size, hidden_size)),
                                       requires_grad = True)

        # Initialize Memory
        if h0 is None:
            self.h = torch.zeros(self.hidden_size)

    def forward(self, input_, hg):
        z = torch.sigmoid(self.weight_xz @ input_ +
                          self.weight_hz @ self.h +
                          self.weight_gpz @ hg)
        r = torch.sigmoid(self.weight_xr @ input_ +
                          self.weight_hr @ self.h +
                          self.weight_gpr @ hg)
        n = torch.tanh(self.weight_xn @ input_ +
                       self.weight_gpn @ hg +
                       r * self.h)
        self.h = (1 - z)*self.h + z*n
        return self.h
        # Not used because it is inconvenient as a matrix
        len_ = int(np.sqrt(len(self.h)))
        return self.h.reshape((len_, len_))





################################################################################
################################################################################
################################################################################
### GEOMETRIC INTERACTION MATRICES
def base_dmatrix(n):
    '''
    Creates the basic distance matrix in an nxn grid. Currently implemented
    only for odd values of n. Each element of this matrix tells what the
    Euclidean distance to the center is.

    Input:
    - n: amount of elements in each row/column.

    Output:
    - distances: and nxn array of distances to the center.
    '''
    if n % 2 == 0:
        raise ValueError('Currently not implemented for even n.')
    bound = n // 2
    disp = [[(ii, jj) for jj in range(-bound, bound + 1)]
            for ii in range(-bound, bound + 1)]
    distances = [[np.sqrt(disp[ii][jj][0]**2 + disp[ii][jj][1]**2)
                  for jj in range(n)]
                 for ii in range(n)]
    return np.array(distances)


def exchange_matrix(n, drow, dcol):
    '''
    Generates the matrices needed to exchange rows and columns. All changes
    are shifts: all rows/columns are moved in the same direction, by the same
    amount. If any reaches the beginning/end, they move to the other side.

    Input:
    - n: number denoting the amount of rows/columns in the matrix to be shifted.
    - drow: displacement in rows. Possitive number means moving the rows down.
      A negative number means moving the rows up.
    - dcol: displacement in columns. Possitive number means shifting the columns
      to the right. A negative number means moving them left.

    Output:
    - tuple containing F, the pre-multiplying matrix to shift rows, and C, the
      post-multiplying matrix to shift columns. Both matrices are nxn.
    '''
    F = np.zeros((n, n))
    C = np.zeros((n, n))
    for ii in range(n):
        new_row = (ii + drow) % n
        F[new_row, ii] = 1
        new_col = (ii + dcol) % n
        C[ii, new_col] = 1
    return (F, C)


def general_dmatrix(n):
    '''
    Creates the matrix of distances between all elements in an nxn, evenly-
    spaced grid.

    Currently implemented only for odd values of n.
    
    Input:
    - n: the size of the square grid of 'interactions'.

    Output:
    - overall: an (n**2)x(n**2) matrix of distances between ALL elements in the
      grid.
    '''
    if n % 2 == 0:
        raise ValueError('Currently not implemented for even n.')
    basal = base_dmatrix(n)
    overall = np.zeros((n**2, n**2))
    bound = n // 2
    disp = [[(ii, jj) for jj in range(-bound, bound + 1)]
            for ii in range(-bound, bound + 1)]
    flat_disp = []
    for pos, val in enumerate(disp):
        flat_disp += val
    for pos, val in enumerate(flat_disp):
        drow, dcol = val
        F, C = exchange_matrix(n, drow, dcol)
        dmatrix = F @ basal @ C
        overall[pos, :] = dmatrix.flatten().copy()
    return overall


def normalize(matrix):
    '''
    Normalizes the matrix so that the sum over the columns equals 0, and the
    sum of the squares over the column equals 1.

    Input:
    - matrix: the matrix to be normalized.

    Output:
    - a normalized version of the same matrix.
    '''
    c = sum(matrix)[0]
    matrix = matrix - (c / matrix.shape[0])
    a = sum(matrix ** 2)[0]
    return matrix / np.sqrt(a)


def make_planar_weightmatrix(len_x:int, len_y:int,
                              long_range_inhibition = True):
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
    if not long_range_inhibition:
        def LFP(x):
            return np.exp(-x)
    else:
        def LFP(x):
            x = np.exp(-0.9*x) - 0.4*np.exp(-0.2*x)
            return x
    gm = general_dmatrix(len_x)
    gm = LFP(gm)
    gm = normalize(gm)
    #for ii in range(weights.shape[0]):
        #weights[ii, ii] = 0
    return torch.Tensor(gm)


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

