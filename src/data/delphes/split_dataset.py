#!/usr/bin/env python3
import h5py as h5
import numpy as np
from pathlib import Path

def split_h5_file(input_file, train_ratio=0.6, output_dir='.', seed=42):
    """
    Split an h5 file into train and validation sets.
    
    Args:
        input_file: Path to input h5 file
        train_ratio: Fraction of data for training (default: 0.6)
        output_dir: Directory to save output files
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Open input file
    with h5.File(input_file, 'r') as f:
        # Get total number of events from first dataset
        n_events = len(f['INPUTS']['Jets']['btag'])
        print(f"Total events: {n_events}")
        
        # Create shuffled indices
        indices = np.arange(n_events)
        np.random.shuffle(indices)
        
        # Split indices
        n_train = int(n_events * train_ratio)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        print(f"Train events: {len(train_indices)} ({len(train_indices)/n_events*100:.1f}%)")
        print(f"Validation events: {len(val_indices)} ({len(val_indices)/n_events*100:.1f}%)")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create output file names
        input_stem = Path(input_file).stem
        train_file = output_dir / f"{input_stem}_train.h5"
        val_file = output_dir / f"{input_stem}_val.h5"
        
        # Function to copy data for given indices
        def copy_split_data(output_file, split_indices, split_name):
            with h5.File(output_file, 'w') as out_f:
                # Copy structure and data
                for main_key in f.keys():  # INPUTS, TARGETS
                    main_group = out_f.create_group(main_key)
                    
                    if isinstance(f[main_key], h5.Group):
                        for sub_key in f[main_key].keys():  # Jets, BoostedJets, etc.
                            sub_group = main_group.create_group(sub_key)
                            
                            if isinstance(f[main_key][sub_key], h5.Group):
                                for dataset_key in f[main_key][sub_key].keys():
                                    # Copy selected indices
                                    data = f[main_key][sub_key][dataset_key][:]
                                    split_data = data[split_indices]
                                    sub_group.create_dataset(dataset_key, data=split_data)
                            else:
                                # Handle case where it's a dataset directly
                                data = f[main_key][sub_key][:]
                                split_data = data[split_indices]
                                main_group.create_dataset(sub_key, data=split_data)
                    else:
                        # Handle case where main_key points to dataset directly
                        data = f[main_key][:]
                        split_data = data[split_indices]
                        out_f.create_dataset(main_key, data=split_data)
                        
            print(f"Saved {split_name} file: {output_file}")
        
        # Create train and validation files
        copy_split_data(train_file, train_indices, "train")
        copy_split_data(val_file, val_indices, "validation")
        
    print("\nSplit complete!")
    return train_file, val_file

# Example usage
if __name__ == "__main__":
    input_file = "/ceph/cms/store/user/dprimosc/spatop/temp/tt_hadronic_testing.h5"
    
    # Split the file (60% train, 40% validation)
    train_file, val_file = split_h5_file(
        input_file,
        train_ratio=0.6,
        output_dir="/ceph/cms/store/user/dprimosc/spatop/temp2/",
        seed=42
    )
    
    # Verify the split
    print("\nVerifying split:")
    with h5.File(train_file, 'r') as f:
        print(f"Train file events: {len(f['INPUTS']['Jets']['btag'])}")
    
    with h5.File(val_file, 'r') as f:
        print(f"Val file events: {len(f['INPUTS']['Jets']['btag'])}")