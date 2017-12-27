import numpy as np
import re
import operator
from collections import defaultdict

from helpers import segment_arr


def find_substr_idx(seq, substring):
    """Find all occurrences in a string. Returns list of tuples (start, end)."""
    
    return np.array([i.span() for i in re.finditer(substring, seq)])


def freq_substrings(seq, n_chars, min_occ=5):
    """Find frequent patterns in the sequence"""

    pattern_dic = defaultdict(int)
    for i in xrange(len(seq) - n_chars + 1):
        chunk = seq[i:i + n_chars]
        pattern_dic[chunk] += 1
    
    return pattern_dic


def pattern_mask(seq, pattern):
    """Get mask of pattern presence in a string."""

    idxs = find_substr_idx(seq, pattern)
    p_mask = segment_arr(idxs, len(seq))
    
    return p_mask


def pattern_coverage(seq, pattern):
    """Find a fraction of sequence, covered by pattern"""
    
    pattern_m = pattern_mask(seq, pattern)
    
    return np.sum(pattern_m)/float(len(pattern_m))


def find_patterns(seq, n_chars):
    """Find frequent patterns in the sequence.
    Output: {pattern:[[idx_start, idx_stop]]}"""

    pattern_dict = defaultdict(list)
    
    for i in range(len(seq) - n_chars + 1):
        chunk = seq[i:i + n_chars]
        i_end = i + n_chars - 1
        # check if pattern start is within last seen segment
        if len(pattern_dict[chunk]) > 0:
            idx_prev = pattern_dict[chunk][-1]
            # add new end idx for the last segment
            if i < idx_prev[1]:
                pattern_dict[chunk][-1][1] = i_end
            else:
                pattern_dict[chunk].append([i, i_end])
        else:
            pattern_dict[chunk].append([i, i_end])
         
    return pattern_dict


def patterns_coverage(pattern_dict, seq_len, sort=True):
    """Find coverage (percent of string which is masked by pattern) for every pattern.
    Input: {pattern:[[idx_start, idx_stop]]}
    Output: {pattern: coverage}"""
    
    def segm_len(segm_lst):
        l = np.sum([i[1] - i[0] + 1 for i in segm_lst])
        return l / float(seq_len)
    
    cov = {k: segm_len(v) for k, v in pattern_dict.items()}
    if sort:
        cov = sorted(cov.items(), key=operator.itemgetter(1), reverse=True)
    return cov