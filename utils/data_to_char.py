import argparse
import os
import json
import numpy as np
from collections import Counter

def create_char_dict(items):
    """Create dictionary of characters item list"""
    n = len(items)
    if n > 90:
        raise AssertionError("Too many factors to code. Limit is 90.")
    s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,-.:;<=>?@[]^_`{|}~'
    return {tuple(items[i]):s[i] for i in range(n)}   
    
def code_factors_in_chars(factors):
    """Convert list of factor arrays of size m into characters by their unique combinations.
    Input: list of np arrays"""
    arr = np.hstack(factors)
    unique_combs = np.unique(arr, axis=0)
    codes = create_char_dict(unique_combs)
    seq = [codes[tuple(i)] for i in arr]
    return ''.join(seq), {v:k for k,v in codes.items()}

def data_to_factors(data, factor_funcs):
    """Apply functions to data and code results in characters"""
    factors = [f(data) for f in factor_funcs]
    return code_factors_in_chars(factors)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="""Code data into characters by appying factor functions: 
        functions, which convert data into levels""")
    
    parser.add_argument('-d', type=str, required=True,
                        help='Name of the CSV file where XY coordinate data is sstored.')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. Sequence and a file with {comb:key} will be saved at the same folder.')
    parser.add_argument('--prefix', type=str, default='', 
                        help='Output name if specificed.')
    parser.add_argument('--move_dir', default=None, dest='move_dir', type=json.loads,
                        help='Dictionary "{\'key\': \'value\'}" of arguments to compute the direction of a movement.')
    parser.add_argument('--move_len', default=None, dest='move_len', type=json.loads,
                        help='Dictionary "{\'key\': \'value\'}" of arguments to compute the length of a movement')

    args = parser.parse_args()
    data_file = args['d']
    out_fld = args['o']
    prefix = args['prefix']
    move_dir_args = args['move_dir']
    move_len_args = args['move_len']
    factors = []
    
    if not os.path.exists(data_file):
        parser.error('The file %s does not exist!' % data_file)
    else:
        data = np.genfromtxt(data_file, delimiter=',')
        
    if move_dir_args is not None:
        from move_direction import move_direction
        data_dir = move_direction(data, **move_dir_args)
        factors.append(data_dir)
        
    if move_len_args is not None:
        from move_length import move_length
        data_len = move_length(data, **move_len_args)
        factors.append(data_len)
        
    if len(factors) == 0:
        raise AssertionError('No functions to process the data.')
    
    sequence, codes = code_factors_in_chars(factors)
    
    if not os.path.exists(out_fld):
        os.makedirs(out_fld)
        seq_file = os.path.join(out_fld, '{}seq.txt'.format(prefix))
        code_file = os.path.join(out_fld, '{}seq_codes.txt'.format(prefix))
        with open(seq_file, 'w') as f:
            f.write(sequence)
        with open(code_file, 'w') as f:
            json.dump(codes, f)              
    print ('Done! Find your sequence in {} and codes in {}'.format(seq_file, code_file))
    print ('Codes statistics:')
    print (Counter(codes.most_common()))