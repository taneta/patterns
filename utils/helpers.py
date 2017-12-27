import numpy as np


def segment_arr(segm_arr, s_len): 
    """ Mask segments of string with ones. 
    Input:
    
    - segm_arr: np.array or segments ([start, end]) to mask. 
    - s_len: string len. 
    
    Output:
    - bool np.array with ones where inside segment ranges.
    """
    s = np.zeros(s_len, dtype=int)
    stop = segm_arr[:,1]
    np.add.at(s, segm_arr[:,0], 1)
    np.add.at(s, stop[stop < len(s)], -1)
    return (s.cumsum() > 0).astype(int)


def shift_x(x, shift=1, fill_value=0):
    """Shift a vector. . 
        First value can be set to some int.
        - x: vector,
        - shift: 1 = one to the right (x can be matched then with previous x)
                -1 = one to the left (x can be matched then with next x)
        - first_value: int (TO DO Fill value from a source)"""

    x_prev = np.roll(x, shift, axis=0) 
    if shift > 0:
        x_prev[:shift] = fill_value  # set a value of the first item
    else:
        x_prev[shift:] = fill_value

    return x_prev

def chunks_from_origin(x, step, origin_value=0):
    """Cut an array into overlaid chunks of size step, add zero point (origin) to every chunk, 
    compute changes in relation to zero point"""
    
    arrs = []

    for i in range(step):
        x_s = np.roll(x, -i)
        arrs.append(x_s)

    arrs_shifted = np.array(arrs)[:,:-i]
    
    x_d = np.diff(arrs_shifted, axis=0)
    x_d = np.vstack([np.repeat(origin_value, x_d.shape[1]), x_d])
    
    res = np.cumsum(x_d, axis=0)
    
    return res


def average_over_window(x, window_size):
    """Compute running mean: average data over window size"""
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size 


def repetitive_seq_mask(x, include_boundary='last'):
    """Get a mask of repetitive elements in an array. 
    The mask leave untouched.
    - include_boundary: (last, first, both, None) - which boundary of repetitive sequence to leave."""

    prev_x = shift_x(x, shift=1, fill_value=0)
    next_x = shift_x(x, shift=-1, fill_value=0)

    prev_in_same_dir = np.array(x == prev_x).astype(int)
    next_in_same_dir = np.array(x == next_x).astype(int)

    repetitive_mask = np.any([next_in_same_dir, prev_in_same_dir], axis=0)

    if include_boundary is not None:
        boundaries = (prev_in_same_dir - next_in_same_dir) # -1 for beginning, 1 for ending

        if include_boundary == 'last':
            leave_boundary = boundaries == 1
        elif include_boundary == 'first':
            leave_boundary = boundaries == -1
        elif include_boundary == 'both':
            leave_boundary = boundaries != 0
        else:
            raise AssertionError("Boundary can be (last, first, both, None)")

        duplicates = (repetitive_mask & ~leave_boundary)
    else: 
        duplicates = repetitive_mask

    return duplicates


def vec_to_coord(a, length):
    """Convert vector coded with dicrection (in radians) and length into coordinates (x, y) from origin."""
    
    angle = np.pi / 2 - a
    angle[angle < 0] += np.pi * 2
    y = length * np.sin(angle)
    x = np.sqrt(length**2 - y**2)
    x[(angle > np.pi / 2) & (angle < 3 * np.pi / 2)] *= -1
    return (x, -y)

def traject_to_coord(angle, length, x_init=None, y_init=None):
    """Convert sequence of vectors (trajectory|path) into (x, y) coordinates."""
    
    x, y = vec_to_coord(angle, length)

    xy = np.vstack([x, y]).T
    xy = np.insert(xy, 0, [0, 0], axis=0)
    xy = np.cumsum(xy, axis=0)
    
    if (x_init is not None) and (y_init is not None):
        xy -= xy[0]
        xy += np.array([x_init, y_init])
    
    return  xy

def slice_gen(data, timesteps, step=None, only_full=False):
    """Generates infinite loop of time slices for the given data.
    Set step to 1 for overlaid sequences. Full to stop when slice is less than timesteps"""
    step = timesteps if step is None else step 
    contin = True 
    while contin:
        for idx in np.arange(0, len(data), step):
            sl = data[idx : idx + timesteps]
            if only_full:
                if len(sl) == timesteps: 
                    contin = False
                    yield sl
            else:
                yield sl


def slice_gen_batch(data, batch_size, timesteps, step=None, only_full=False):
    """Generates infinite loop of time slice batches for the given data"""
    gen = slice_gen(data, timesteps, step=step, only_full=only_full)
    while True:
        x_batch = np.empty((batch_size, timesteps, *data.shape[1:]))
        for i in range(batch_size):
            x_batch[i] = next(gen)
        yield x_batch