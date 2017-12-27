import numpy as np
from helpers import shift_x

def euclidian_dist(x0, y0, x1, y1):
    """Find euclidian distances between gaze points.
    - x, y: vectors, coordinates of a point,
    - x_prev, y_prev: vectors, coordinates of a previous point."""
    return np.sqrt(np.square(x0 - x1) + np.square(y0 - y1))

def move_length(x, y):
    """Find euclidian distances between two points."""
    if len(x) < 2:
        return np.array(0)
    x_prev = shift_x(x, fill_value=x[0])
    y_prev = shift_x(y, fill_value=y[0])
    return euclidian_dist(x, y, x_prev, y_prev)[1:]

def tiny_moves_mask(x, y, delta=0.001):
    
    move_len = move_length(x, y)
    return move_len < delta
