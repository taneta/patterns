import six
import itertools
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection

from move_direction import angle_clockwise
from data_to_segments import angle_to_segments
from patterns import find_substr_idx
from helpers import shift_x, chunks_from_origin
from data_to_char import code_factors_in_chars

COLORS = [color for color in list(six.iteritems(mcolors.cnames)) if not ':' in color]
np.random.seed(42)

def xy_to_segments(xy):
    return list(zip(xy[:-1], xy[1:]))

def plot_path(xy_arr, segments_weights=None, clip=True, figsize=(6, 4), title='', 
              change_width=True, change_color=True, screen_lims=False, colorbar=True,
              weight_threshold=None, show_joints=False, 
              feed_lines=False, **kwargs):
    """Plot trajectories on a screen and highlight segments with color and linewidth"""
   
    # Reshape xy_arr to a sequence of segments [[(x0,y0),(x1,y1)],[(x1,y1),(x2,y2)],...]
    if clip:
        xy_arr = np.clip(xy_arr, 0, 1)
    if feed_lines:
        segments = xy_arr
    else:
        segments = np.array(xy_to_segments(xy_arr))
    
    sw = np.array(segments_weights) if  segments_weights is not None else None
    cmap = None
    linewidths = None
    
    # Show weights with color and linewidth
    if (sw is not None) and change_color:
            cmap = plt.get_cmap('plasma')
    if (sw is not None) and change_width:
            linewidths = (1 + (sw-min(sw)/(max(sw)-min(sw) + 1e-16)) * 3)
            
    # Plot only segments where weight is higher than zero:
    if (sw is not None) and (weight_threshold is not None):
        segments = segments[sw > weight_threshold]
        sw = sw[sw > weight_threshold]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Convert data to LineCollection
    line_segments = LineCollection(segments, linewidths=linewidths, linestyles='solid', 
                                   cmap=cmap,  **kwargs)#, norm=plt.Normalize(min(fw), max(fw)))
    line_segments.set_array(sw)
    ax.add_collection(line_segments)
    
    if show_joints:
        # x, y start segment points + last x, y point from segments ends
        x = np.concatenate([segments[:, 0][:, 0], [segments[:, 1][-1, 0]]])
        y = np.concatenate([segments[:, 0][:, 1], [segments[:, 1][-1, 1]]])
        ax.scatter(x, y, s=10)
    
    # Add colorbar for weights
    if (sw is not None) and colorbar:
        axcb = fig.colorbar(line_segments)
        axcb.set_label('Activation')
        plt.sci(line_segments)  # this allows interactive changing of the colormap.
    
    # Set plot limits
    if screen_lims:
        ax.set_ylim([-0.05, 1.05])   
        ax.set_xlim([-0.05, 1.05])  
    else:
        ax.set_xlim((np.amin(xy_arr[:, 0] - 0.05), np.amax(xy_arr[:, 0]) + 0.05))
        ax.set_ylim((np.amin(xy_arr[:, 1] - 0.05), np.amax(xy_arr[:, 1]) + 0.05))

    ax.set_title(title, fontsize=12, y=1.1)
    ax.invert_yaxis() # invert y axis according to the eye-tracking data
    ax.xaxis.tick_top() 
        
    return fig


def plot_dir_segments(n_segments=12, n=1000):
    """Plot example of N segments"""
    
    np.random.seed(48)
    p = 2 * np.random.rand(3, n*2) - 1
    p = p[:n, np.power(np.sum(p * p, 0), 0.5) <= 1]
    x, y = p[0], p[1]

    move_dir = angle_clockwise(x1 = 0, y1 = 1, x2 = x.T, y2 = y.T)  
    dir_segm = angle_to_segments(move_dir, n_segments, segment_ranges=False)

    fig = plt.figure(figsize=(7, 7))
    _ = plt.scatter(x, y, c=dir_segm, cmap=plt.cm.tab20_r, s=15)
    return fig

def plot_len_segments(move_len, factor_len):
    """Plot distribution lengths into N segments"""
    
    labels = list(set(factor_len))
    colors = plt.cm.tab20_r
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    p = plt.hist(move_len, bins=50)

    for i, (l_s, l_e) in enumerate(labels):
        ax.axvspan(l_s, l_e, facecolor=COLORS[i][1],
                   label='{:.3f}-{:.3f}'.format(l_s, l_e), alpha=0.2)

    ax.legend(loc='upper right')
    return fig

def plot_chunks_example(n_segments=12, n=10):
    
    p = 2 * np.random.rand(3, n) - 1
    p = p[:, np.power(np.sum(p * p, 0), 0.5) <= 1]
    x, y = p[0], p[1]

    move_dir = angle_clockwise(x1 = 0, y1 = 1, x2 = x.T, y2 = y.T)  
    dir_segm = angle_to_segments(move_dir, n_segments)
    
    fig = plt.figure(figsize=(7, 7))
    
    seq, codes = code_factors_in_chars([dir_segm])

    for i, char in enumerate(sorted(codes.keys())):
        mask = [c == char for c in seq]
        mask = shift_x(mask, shift=-1) | mask # to include previous x
        x_chunks = chunks_from_origin(x[mask], step=2, origin_value=0.)
        y_chunks = chunks_from_origin(y[mask], step=2, origin_value=0.)
        plt.plot(x_chunks, y_chunks, color=COLORS[i][1], alpha=0.5)
    
    plt.scatter(x, y)
    
    for i in range(len(x)):
        plt.annotate(str(i), (x[i], y[i]+0.02), fontsize=12)
    return fig

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, figsize=(6, 6)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04) # magic to equlize height of colorbar with the image height
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig