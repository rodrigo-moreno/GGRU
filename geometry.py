import numpy as np
from itertools import product
from scipy.spatial import distance_matrix

def _make_planar_weightmatrix(len_x:int, len_y:int):
    '''
    A function that makes a planar weight matrix of the interactions of cells
    distributed in a plane. All cells have the same separation between them.
    Input:
    - len_x = length along the x axis
    - len_y = length along the y axis

    Output:
    - matrix with interaction weights.

    Note:
    The current solution for the problem of the diagonal of the weight matrix
    depends strictly on len_x == len_y.
    '''
    combinations = list(product(range(len_x), range(len_y)))
    weights = distance_matrix(combinations, combinations)
    weights = np.exp(-weights)
    return weights


if __name__ == '__main__':
    w = _make_planar_weightmatrix(2, 3)
    print(w)

