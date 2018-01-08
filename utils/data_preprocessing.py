import numpy as np

from move_length import tiny_moves_mask
from move_direction import move_abs_direction
from data_to_segments import data_to_segments
from helpers import repetitive_seq_mask


def remove_tiny_moves(df, delta=0.001):
    """Remove a row if distance between two points is less than delta"""
    
    tiny_moves = np.concatenate([[False], tiny_moves_mask(df.x.values, df.y.values, delta)])
    while np.any(tiny_moves):
        df = df[~tiny_moves]
        tiny_moves = np.concatenate([[False], tiny_moves_mask(df.x.values, df.y.values, delta)])  
    return df

def merge_same_direction(df, n_directions):
    """Merge rows with movements towards the same direction"""
    
    move_dir = move_abs_direction(df.x.values, df.y.values, invert_y=True)
    move_dir_chars = data_to_segments(move_dir, n_segments=n_directions, segment_ranges=False)
    repet_moves = repetitive_seq_mask(move_dir_chars, include_boundary='last')
    repet_moves = np.concatenate([[False], repet_moves])
    df = df[~repet_moves]
    return df

def norm_values(X, x_min=None, x_max=None):
    x_min = x_min if x_min is not None else np.min(X)
    x_max = x_max if x_max is not None else np.max(X)
    return (X - x_min) / (x_max - x_min + 1e-18)

def denorm(X, x_max, x_min=0):
    return (X + x_min) * (x_max - x_min + 1e-18) 