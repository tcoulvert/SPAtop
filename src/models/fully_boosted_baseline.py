#import packages!
import h5py
import os
import vector
import awkward as ak
import matplotlib.pyplot as plt 
import numpy as np

TOP_MASS = 172.52  # GeV

#make the file path
# file_path = "/Users/laurencadle/Downloads/tt_hadronic_testing.h5"
file_path = "/storage/af/user/tsievert/topNet/SPAtop/data/delphes/v4/tt_hadronic_testing_SLIMMED.h5"
with h5py.File(file_path, 'r') as f: 
    very_fat_jets = f['INPUTS/VeryBoostedJets']

#vfj_mass
with h5py.File(file_path, 'r') as f:
    pt = f['INPUTS/VeryBoostedJets/vfj_pt'][:]
    eta = f['INPUTS/VeryBoostedJets/vfj_eta'][:]
    phi = f['INPUTS/VeryBoostedJets/vfj_phi'][:]
    mass = f['INPUTS/VeryBoostedJets/vfj_mass'][:]
    
    very_fat_jets = ak.zip({
    "pt": pt,
    "eta": eta,
    "phi": phi,
    "mass": mass 
    }, with_name="Momentum4D")

#chi2 calculations only rely on the mass of the fat jets vs the top
mass_diff = ak.where(
    very_fat_jets.mass > TOP_MASS,
    very_fat_jets.mass - TOP_MASS,
    TOP_MASS - very_fat_jets.mass
)

#save the indices and jets!
# print("Num events")
# print(ak.to_list(best2_jets))
# print("Indices of the best 2 jets: ", best2_indices)

#time to make plots!
vfj_mass = ak.to_numpy(ak.flatten(very_fat_jets.mass))

plt.hist(vfj_mass, bins = 100, label = 'VFJ masses', color = 'turquoise', alpha = 0.5)
plt.axvline(TOP_MASS, color = 'pink', linestyle = 'solid', linewidth =  2, label = 'Mass of top (172.52 GeV)')
plt.title("Very Fat Jet Mass Distributions")
plt.xlabel('Mass (GeV)')
plt.ylabel('Number of Jets')
plt.legend()
plt.show()