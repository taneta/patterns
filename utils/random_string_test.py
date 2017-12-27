import numpy as np
import matplotlib.pyplot as plt

from helpers import repetitive_seq_mask

def merge_repetitions_str(s):
    """Merge repetitive chars in a string"""
    rep_mask = repetitive_seq_mask(s, include_boundary='last')
    return s[~rep_mask]


def gen_random_str(s_len=None, from_seq=None, n_letters=None):
    """Generate random string of length 's_len' from string or from N letters"""
    
    s_len = s_len if s_len is not None else len(from_seq)
    
    if from_seq is not None:
        seq = np.asarray(from_seq)
        np.random.shuffle(seq)
        return seq[:s_len]

    letters = [chr(ord('A') + x) for x in range(60) if chr(ord('A') + x).isalpha()][:n_letters]
    return np.random.choice(letters, s_len)
    
    
def random_seq(s_len, n_letters=None, merge_repetitions=True, from_seq=None): 
    """ Generate random string and find patterns there. 
    Check explained variance."""
    
    random_s = gen_random_str(s_len, from_seq, n_letters)
    
    if merge_repetitions:
        random_s = merge_repetitions_str(random_s)
        while len(random_s) < s_len:
            len_diff = s_len - len(random_s)
            add_rand_s = gen_random_str(len_diff, from_seq, n_letters)
            add_rand_s = merge_repetitions_str(add_rand_s)
            random_s = np.concatenate([random_s, add_rand_s]) 
    
    return ''.join(random_s[:s_len])


def plot_coverage_comparison(pattern_coverage, rand_pattern_coverage):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    plt.plot(np.array(pattern_coverage)*100, color='orange', label='gaze data')
    plt.plot(np.array(rand_pattern_coverage)*100, color='gray', label='permuted gaze data')
    plt.xlabel("Pattern item")
    plt.ylabel("% covered")
    plt.title("Sequence coverage by {} most common patterns".format(len(pattern_coverage)), y=1.05, fontsize=18)
    plt.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig