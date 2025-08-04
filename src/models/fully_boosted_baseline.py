import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import awkward as ak
import vector as vec
vec.register_awkward()

FILEPATH = os.path.abspath(__file__)
DIRPATH = '/'.join(FILEPATH.split('/')[:-1])

TOP_MASS = 172.52  # GeV

PLOT_CHI2_HISTS = True
PLOT_ROCS = True
PLOT_MASSES = True

#make the file path
file_path = os.path.join(DIRPATH, "../../data/delphes/v4/tt_hadronic_testing_SLIMMED.h5")
with h5py.File(file_path, 'r') as f:
    pt = f['INPUTS/VeryBoostedJets/vfj_pt'][:]
    eta = f['INPUTS/VeryBoostedJets/vfj_eta'][:]
    phi = f['INPUTS/VeryBoostedJets/vfj_phi'][:]
    mass = f['INPUTS/VeryBoostedJets/vfj_mass'][:]

    tgt_t1_bqq = f['TARGETS/FBt1/bqq'][:]
    tgt_t1_mask = f['TARGETS/FBt1/mask'][:]

    tgt_t2_bqq = f['TARGETS/FBt1/bqq'][:]
    tgt_t2_mask = f['TARGETS/FBt1/mask'][:]
    
vf_jets = ak.zip({
    "pt": pt,
    "eta": eta,
    "phi": phi,
    "mass": mass 
}, with_name="Momentum4D")
vf_jets = ak.with_field(vf_jets, ak.local_index(vf_jets, axis=1), "index")

# Fully Boosted χ² calculations only rely on the mass of the very-fat-jets vs the top
mass_diff = ak.where(
    vf_jets.mass > TOP_MASS,
    vf_jets.mass - TOP_MASS,
    TOP_MASS - vf_jets.mass
)

chi2_t1 = mass_diff[ak.local_index(mass_diff) == 0]
t1_bqq_idx = vf_jets.index[ak.local_index(vf_jets.index) == 0]
chi2_t2 = mass_diff[ak.local_index(mass_diff) == 1]
t2_bqq_idx = vf_jets.index[ak.local_index(vf_jets.index) == 1]

# Plot the "χ²" histograms
if PLOT_CHI2_HISTS:
    # Plot Top1 χ² histogram
    chi2_t1_vals = ak.ravel(chi2_t1[~ak.is_none(chi2_t1)])
    plt.figure()
    plt.hist(chi2_t1_vals, bins=50)
    plt.xlabel("χ² (Top1)")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.title("Chi-Squared Distribution for Top1 Candidates")
    plt.grid(True)
    plt.savefig(os.path.join(DIRPATH, "fully_boosted_chisq_top1.pdf"))
    # Plot Top2 χ² histogram 
    chi2_t2_vals = ak.ravel(chi2_t2[~ak.is_none(chi2_t2)])
    plt.figure()
    plt.hist(chi2_t2_vals, bins=50)
    plt.xlabel("χ² (Top2)")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.title("Chi-Squared Distribution for Top2 Candidates")
    plt.grid(True)
    plt.savefig(os.path.join(DIRPATH, "fully_boosted_chisq_top2.pdf"))

# Plot the ROCs for the fully-boosted baseline
if PLOT_ROCS:
    def correct_mask(pred_bqq):
        return (
            tgt_t1_mask
            & (pred_bqq == tgt_t1_bqq)
        ) | (
            tgt_t2_mask
            & (pred_bqq == tgt_t2_bqq)
        )
    
    correct_t1 = correct_mask(t1_bqq_idx)
    correct_t2 = correct_mask(t2_bqq_idx)

    valid_t1 = ~ak.is_none(correct_t1)
    valid_t2 = ~ak.is_none(correct_t2)

    print(f"num valid t1 = {ak.sum(valid_t1)} out of {ak.num(valid_t1, axis=0)}")
    print(f"num valid t2 = {ak.sum(valid_t2)} out of {ak.num(valid_t2, axis=0)}")
    print(f"num correct and valid t1 = {ak.sum(correct_t1[valid_t1])} out of {ak.num(correct_t1[valid_t1], axis=0)}")
    print(f"num correct and valid t2 = {ak.sum(correct_t2[valid_t2])} out of {ak.num(correct_t2[valid_t2], axis=0)}")

    chi2_t1 = ak.to_numpy(chi2_t1[valid_t1])
    chi2_t2 = ak.to_numpy(chi2_t2[valid_t2])
    label_t1 = ak.to_numpy(correct_t1[valid_t1])
    label_t2 = ak.to_numpy(correct_t2[valid_t2])

    # === Plot ROC ===
    def plot_roc(chi2_vals, label, plotlabel):
        fpr, tpr, _ = roc_curve(label, 1/chi2_vals)  # -chi2_vals
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{plotlabel} (AUC = {roc_auc:.3f})")

    plt.figure(figsize=(7, 6))
    plot_roc(chi2_t1, label_t1, "Top1")
    plot_roc(chi2_t2, label_t2, "Top2")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Chi² Discriminator")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(DIRPATH, "fully_boosted_chisq_ROC.pdf"))

# if PLOT_MASSES:
#     plt.hist(vfj_mass, bins = 100, label = 'VFJ masses', color = 'turquoise', alpha = 0.5)
#     plt.axvline(TOP_MASS, color = 'pink', linestyle = 'solid', linewidth =  2, label = 'Mass of top (172.52 GeV)')
#     plt.title("Very Fat Jet Mass Distributions")
#     plt.xlabel('Mass (GeV)')
#     plt.ylabel('Number of Jets')
#     plt.legend()
#     plt.savefig(os.path.join(DIRPATH, "fully_boosted_chisq_"))