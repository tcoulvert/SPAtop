#!/usr/bin/env python3
import h5py as h5
import numpy as np
from pathlib import Path

def merge_h5_files(file1, file2, output_file, shuffle=False, seed=42):
    """
    Merge two h5 files with the same structure.
    
    Args:
        file1: Path to first h5 file
        file2: Path to second h5 file
        output_file: Path for merged output file
        shuffle: Whether to shuffle the merged data (default: False)
        seed: Random seed for shuffling
    """
    if shuffle:
        np.random.seed(seed)
    
    with h5.File(file1, 'r') as f1, h5.File(file2, 'r') as f2:
        # Get number of events from each file
        n_events1 = len(f1['INPUTS']['Jets']['btag'])
        n_events2 = len(f2['INPUTS']['Jets']['btag'])
        n_total = n_events1 + n_events2
        
        print(f"File 1 events: {n_events1}")
        print(f"File 2 events: {n_events2}")
        print(f"Total events: {n_total}")
        
        # Create indices for potential shuffling
        if shuffle:
            indices = np.arange(n_total)
            np.random.shuffle(indices)
        
        with h5.File(output_file, 'w') as out_f:
            # Merge structure and data
            for main_key in f1.keys():  # INPUTS, TARGETS
                main_group = out_f.create_group(main_key)
                
                if isinstance(f1[main_key], h5.Group):
                    for sub_key in f1[main_key].keys():  # Jets, BoostedJets, etc.
                        sub_group = main_group.create_group(sub_key)
                        
                        if isinstance(f1[main_key][sub_key], h5.Group):
                            for dataset_key in f1[main_key][sub_key].keys():
                                # Concatenate data from both files
                                data1 = f1[main_key][sub_key][dataset_key][:]
                                data2 = f2[main_key][sub_key][dataset_key][:]
                                merged_data = np.concatenate([data1, data2], axis=0)
                                
                                # Shuffle if requested
                                if shuffle:
                                    merged_data = merged_data[indices]
                                
                                sub_group.create_dataset(dataset_key, data=merged_data)
                        else:
                            # Handle case where it's a dataset directly
                            data1 = f1[main_key][sub_key][:]
                            data2 = f2[main_key][sub_key][:]
                            merged_data = np.concatenate([data1, data2], axis=0)
                            
                            if shuffle:
                                merged_data = merged_data[indices]
                            
                            main_group.create_dataset(sub_key, data=merged_data)
                else:
                    # Handle case where main_key points to dataset directly
                    data1 = f1[main_key][:]
                    data2 = f2[main_key][:]
                    merged_data = np.concatenate([data1, data2], axis=0)
                    
                    if shuffle:
                        merged_data = merged_data[indices]
                    
                    out_f.create_dataset(main_key, data=merged_data)
        
        print(f"\nMerged file saved: {output_file}")
        print(f"Shuffled: {shuffle}")
    
    # Verify the merge
    with h5.File(output_file, 'r') as f:
        merged_events = len(f['INPUTS']['Jets']['btag'])
        print(f"\nVerification - Merged file events: {merged_events}")
        assert merged_events == n_total, f"Event count mismatch! Expected {n_total}, got {merged_events}"
    
    return output_file


def merge_multiple_h5_files(file_list, output_file, shuffle=False, seed=42):
    """
    Merge multiple h5 files with the same structure.
    
    Args:
        file_list: List of paths to h5 files
        output_file: Path for merged output file
        shuffle: Whether to shuffle the merged data
        seed: Random seed for shuffling
    """
    if len(file_list) < 2:
        raise ValueError("Need at least 2 files to merge")
    
    if shuffle:
        np.random.seed(seed)
    
    # Get total number of events
    total_events = 0
    event_counts = []
    
    for filepath in file_list:
        with h5.File(filepath, 'r') as f:
            n_events = len(f['INPUTS']['Jets']['btag'])
            event_counts.append(n_events)
            total_events += n_events
            print(f"{Path(filepath).name}: {n_events} events")
    
    print(f"\nTotal events to merge: {total_events}")
    
    # Create indices for potential shuffling
    if shuffle:
        indices = np.arange(total_events)
        np.random.shuffle(indices)
    
    # Open first file to get structure
    with h5.File(file_list[0], 'r') as f_ref:
        with h5.File(output_file, 'w') as out_f:
            # Create structure and merge data
            for main_key in f_ref.keys():
                main_group = out_f.create_group(main_key)
                
                if isinstance(f_ref[main_key], h5.Group):
                    for sub_key in f_ref[main_key].keys():
                        sub_group = main_group.create_group(sub_key)
                        
                        if isinstance(f_ref[main_key][sub_key], h5.Group):
                            for dataset_key in f_ref[main_key][sub_key].keys():
                                # Collect data from all files
                                all_data = []
                                for filepath in file_list:
                                    with h5.File(filepath, 'r') as f:
                                        data = f[main_key][sub_key][dataset_key][:]
                                        all_data.append(data)
                                
                                # Concatenate all data
                                merged_data = np.concatenate(all_data, axis=0)
                                
                                # Shuffle if requested
                                if shuffle:
                                    merged_data = merged_data[indices]
                                
                                sub_group.create_dataset(dataset_key, data=merged_data)
    
    print(f"\nMerged file saved: {output_file}")
    print(f"Shuffled: {shuffle}")
    
    # Verify
    with h5.File(output_file, 'r') as f:
        merged_events = len(f['INPUTS']['Jets']['btag'])
        print(f"\nVerification - Merged file events: {merged_events}")
        assert merged_events == total_events, f"Event count mismatch!"
    
    return output_file


# Example usage
if __name__ == "__main__":
    # Example 1: Merge two files
    file1 = "/ceph/cms/store/user/dprimosc/spatop/temp2/tt_hadronic_testing_train.h5"
    file2 = "/ceph/cms/store/user/dprimosc/spatop/tt_hadronic_training.h5"
    output = "/ceph/cms/store/user/dprimosc/spatop/tmp3/tt_hadronic_training.h5"
    
    merge_h5_files(file1, file2, output, shuffle=True)
    
    # Example 2: Merge multiple files
    # files = [
    #     "file1.h5",
    #     "file2.h5", 
    #     "file3.h5"
    # ]
    # merge_multiple_h5_files(files, "merged_all.h5", shuffle=True)