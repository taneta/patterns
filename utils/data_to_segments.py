import numpy as np


def digitize(x, bins, segment_ranges=True):
    segments = np.digitize(x, bins) - 1
    
    if segment_ranges:
        def names(i):
            if i < len(bins) - 1:
                name = (bins[i], bins[i+1])
            else:
                name = (bins[i], np.max(x))
            return name
        segments = [names(i) for i in segments]
    return segments

def split_to_segments(x, x_min, x_max, n_segments=12, segment_ranges=True):
    """Split data into n segments"""
    
    x_size = x_max - x_min
    bins = np.linspace(x_min, x_max, n_segments+1)
    return digitize(x, bins[:-1], segment_ranges)


def round_nearest_quantile(x, levels, p_min=0.1, p_max=99.9, segment_ranges=True):
    bins = np.linspace(np.percentile(x, p_min), np.percentile(x, p_max), num=levels)
    centers = (bins[1:] + bins[:-1]) / 2.
    centers = np.insert(centers, 0, np.min(x))
    return digitize(x, centers, segment_ranges)


def split_equal_bins(x, levels, segment_ranges=True):
    sep = (x.size/float(levels)) * np.arange(1, levels + 1)
    bins = np.insert(sep, 0, 0)
    return digitize(x, bins, segment_ranges)

def angle_to_segments(angles, n_segments=12, segment_ranges=True):
    """ Find a letter-coded segment of n_segments for every angle data point.
        Segments are shifted for 15 degrees to the left, to keep all vertical lines in one segment."""

    shift = 360./(n_segments*2)
    angles = (angles + shift) % 360
    angle_segments = split_to_segments(angles, x_min=0, x_max=360, n_segments=n_segments, segment_ranges=segment_ranges)
    return angle_segments


def data_to_segments(x, n_segments, segment_ranges=True):
    """ Find a letter-coded segment of n_segments for every data point."""
    return split_to_segments(x, x_min=np.min(x), x_max=np.max(x), n_segments=n_segments, segment_ranges=segment_ranges)


def data_to_segments_uniform(x, n_segments, segment_ranges=True):
    """ Split data into segments of equal size (number of observations)."""
    return split_equal_bins(x, n_segments)

def data_to_segments_quantile(x, n_segments, p_min=0.1, p_max=99.9, segment_ranges=True):
    """ Split data into segments by closiness to quantiles (works good for 3 levels)"""
    return round_nearest_quantile(x, n_segments, p_min, p_max, segment_ranges)
