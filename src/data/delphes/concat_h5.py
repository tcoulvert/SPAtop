import sys

import h5py
import numpy as np
#################################
output_h5 = sys.argv[1]
input_h5s = sys.argv[2:]

# Checks the h5 files to merge have identical structure
h5input_structure = {}
def check_structure(name, obj):
    # print(f"name: {name}, type: {type(obj)}")
    if isinstance(obj, h5py.Dataset):
        if name not in h5input_structure: h5input_structure[name] = obj.dtype 
        if obj.dtype != h5input_structure[name]: return False

# Appends the datasets to the merged container
merged_dataset = {}
def append_dataset(name, obj):
    if isinstance(obj, h5py.Dataset):
        # print(f"name: {name}, type: {obj.dtype}")
        if name not in merged_dataset: merged_dataset[name] = []
        merged_dataset[name].append(obj[:])

#################################
# Builds the merged container with good-structure, input h5 files 
for h5file in input_h5s:
    h5 = h5py.File(h5file, 'r')
    if h5.visititems(check_structure) is not None:
        print(f'h5 file at {h5file} doesn\'t match first files structure, conintuing with other files'); continue

    h5.visititems(append_dataset)

# Creates the output h5 file
with h5py.File(output_h5, "w") as output:
    for dataset_name, all_data in merged_dataset.items():
        concat_data = np.concatenate(all_data, axis=0)
        output.create_dataset(dataset_name, data=concat_data)