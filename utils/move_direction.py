import numpy as np
from utils.helpers import shift_x


def vec_length(x, y):
    return np.sqrt(np.square(x) + np.square(y))

def dot_product(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2

def determinant(x1, y1, x2, y2):
    return x1 * y2 - y1 * x2

def inner_angle(x1, y1, x2, y2, radians):
    cosx = np.dot(np.array([x1, y1]), np.array([x2, y2])) / ((vec_length(x1, y1) * vec_length(x2, y2)) + 1e-9)
    rad = np.arccos(cosx) # in radians        
    return rad if radians else rad * 180 / np.pi # returns degrees

def angle_clockwise(x1, y1, x2, y2, radians=False):
    """Coordinates to angle """
    coef = 2 * np.pi if radians else 360
    inner = inner_angle(x1, y1, x2, y2, radians)
    det = determinant(x1, y1, x2, y2)
    # If the det < 0 then B is clockwise of A, else A is clockwise of B
    res = np.where(det < 0, inner, coef - inner)
#     res = np.round(res, 0)
    # replace 360 with 0 angle
    res = np.where(res == coef, 0, res)
    
    return res

def move_abs_direction(x, y, invert_x=False, invert_y=True, radians=False):
    """Find an angle between zero vector (0, 1) and vector between prev and current point."""

    x_prev = shift_x(x, fill_value=x[0])
    y_prev = shift_x(y, fill_value=y[0])
        
    # new relative position, invert scales if needed
    x_pos = x - x_prev if not invert_x else -(x - x_prev)
    y_pos = y - y_prev if not invert_y else -(y - y_prev)

    # compute angles
    abs_angle = angle_clockwise(x1 = 0, y1 = 1, x2 = x_pos.T, y2 = y_pos.T, radians=radians)

    return abs_angle[1:]


def move_rel_direction(x, y, invert_x=False, invert_y=True):
    """ Find an angle between zero vector (0, 1) and vector between prev and current point."""
    
    move_abs_angle = move_abs_direction(x, y, invert_x, invert_y)
    rel_angles = move_abs_angle - shift_x(move_abs_angle, fill_value=move_abs_angle[0])

    return rel_angles