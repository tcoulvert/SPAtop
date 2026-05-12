# stdlib packages
import sys

# common python packages
import h5py
import numpy as np

############################


orig_filepath = sys.argv[1]
new_filepath = orig_filepath[:orig_filepath.rfind('.')] + '_reformat' + orig_filepath[orig_filepath.rfind('.'):]

with h5py.File(orig_filepath, 'r') as orig_file:
    with h5py.File(new_filepath, 'w') as new_file:
        for orig_key in orig_file['source'].keys():
            if orig_key == 'mask': new_key = 'MASK'
            else: new_key = orig_key
            new_file.create_dataset(f'INPUTS/Jets/{new_key}', data=np.array(orig_file['source'][orig_key][:]))
        for orig_key in orig_file['t1'].keys():
            if orig_key == 'mask': new_key = 'MASK'
            else: new_key = orig_key
            new_file.create_dataset(f'TARGETS/FRt1/{new_key}', data=np.array(orig_file['t1'][orig_key][:]))
            new_file.create_dataset(f'TARGETS/FRt2/{new_key}', data=np.array(orig_file['t2'][orig_key][:]))
