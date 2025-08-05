import os
import time

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import awkward as ak
import vector as vec
vec.register_awkward()

# start = time.time()
FILEPATH = os.path.abspath(__file__)
DIRPATH = '/'.join(FILEPATH.split('/')[:-1])

N_TOPS = 2
TOP_MASS = 172.52  # GeV
W_MASS = 80.37  # GeV

PLOT_CHI2_HISTS = False
PLOT_ROCS = False
SAVE_H5 = True

file_path = os.path.join(DIRPATH, "../../data/delphes/v4/tt_hadronic_testing_SLIMMED.h5")
# 1) Load arrays
with h5py.File(file_path, "r") as f:
    pt   = f['INPUTS/Jets/pt'][:]
    eta  = f['INPUTS/Jets/eta'][:]
    phi  = f['INPUTS/Jets/phi'][:]
    mass = f['INPUTS/Jets/mass'][:]
    btag = f['INPUTS/Jets/btag'][:]

    tgt_t1_b    = f['TARGETS/FRt1/b'][:]
    tgt_t1_q1   = f['TARGETS/FRt1/q1'][:]
    tgt_t1_q2   = f['TARGETS/FRt1/q2'][:]
    tgt_t1_mask = f["TARGETS/FRt1/mask"][:]

    tgt_t2_b    = f['TARGETS/FRt2/b'][:]
    tgt_t2_q1   = f['TARGETS/FRt2/q1'][:]
    tgt_t2_q2   = f['TARGETS/FRt2/q2'][:]
    tgt_t2_mask = f["TARGETS/FRt2/mask"][:]

# 2) Build jagged [events][jets] array
jets = ak.zip({
    "pt": pt,
    "eta": eta,
    "phi": phi,
    "mass": mass,
    "btag": btag
}, with_name="Momentum4D")
jets = ak.with_field(jets, ak.local_index(jets, axis=1), "index")


# 3) Split jets
bjets_mask = (jets.btag == 1)
bjets = ak.drop_none(ak.mask(jets, bjets_mask))
ljets = ak.drop_none(ak.mask(jets, ~bjets_mask))


top_dict = {}
for i in range(N_TOPS):
    # 4) W jj combinations
    w = ak.combinations(ljets, 2, axis=1, fields=["j1", "j2"])
    w = ak.with_field(w, (w.j1 + w.j2).mass, "w_mass")

    # 5) Top combinations
    t = ak.cartesian({"w": w, "b": bjets}, axis=1)
    t = ak.with_field(t, (t.w.j1 + t.w.j2 + t.b).mass, "top_mass")
    t = ak.with_field(t, (t.w.j1 + t.w.j2 + t.b).pt, "top_pt")

    # 6) Top χ²
    chi2_all1 = ( (t.w.w_mass - W_MASS) / (0.1 * W_MASS) )**2 + ( (t.top_mass - TOP_MASS) / (0.1 * TOP_MASS) )**2
    idx1 = ak.argmin(chi2_all1, axis=1)
    best_t = ak.firsts(t[ak.local_index(t) == idx1])

    top_dict[f'FRt{i+1}_mask'] = ~ak.is_none(best_t)
    top_dict[f'FRt{i+1}_b'] = best_t.b.index
    top_dict[f'FRt{i+1}_q1'] = best_t.w.j1.index
    top_dict[f'FRt{i+1}_q2'] = best_t.w.j2.index
    top_dict[f'FRt{i+1}_pt'] = best_t.pt
    top_dict[f'FRt{i+1}_chi2'] = ak.firsts(chi2_all1[ak.local_index(chi2_all1) == idx1])

    # 7) Build ak arrays of unused jets
    bjets = bjets[bjets.index != best_t.b.index]
    ljets = ljets[(ljets.index != best_t.w.j1.index) & (ljets.index != best_t.w.j2.index)]

    # 9) Repeat


# Save out new h5 file
if SAVE_H5:
    out_filepath = os.path.join(DIRPATH, "../../data/delphes/v4/tt_hadronic_baseline.h5")
    with h5py.File(out_filepath, 'a') as f:
        with h5py.File(file_path, 'r') as test_f:
            f['INPUTS'] = test_f['INPUTS']

        for i in range(N_TOPS):
            f[f'TARGETS/FRt{i+1}/mask'] = ak.to_numpy(top_dict[f't{i+1}_mask'])
            f[f'TARGETS/FRt{i+1}/b'] = ak.to_numpy(top_dict[f't{i+1}_b'])
            f[f'TARGETS/FRt{i+1}/q1'] = ak.to_numpy(top_dict[f't{i+1}_q1'])
            f[f'TARGETS/FRt{i+1}/q2'] = ak.to_numpy(top_dict[f't{i+1}_q2'])
            f[f'TARGETS/FRt{i+1}/pt'] = ak.to_numpy(top_dict[f't{i+1}_pt'])
            f[f'TARGETS/FRt{i+1}/chi2'] = ak.to_numpy(top_dict[f't{i+1}_chi2'])


## Outputs ##
# Plot resolved baseline χ² distributions
if PLOT_CHI2_HISTS:
    # Plot Top χ² histograms
    for i in range(N_TOPS):
        chi2_vals_1 = ak.ravel(top_dict[f'FRt{i+1}_chi2'][~ak.is_none(top_dict[f'FRt{i+1}_chi2'])])
        plt.figure()
        plt.hist(chi2_vals_1, bins=50)
        plt.xlabel(f"χ² (Top{i+1})")
        plt.ylabel("Frequency")
        plt.yscale('log')
        plt.title(f"Chi-Squared Distribution for Top{i+1} Candidates")
        plt.grid(True)
        plt.savefig(os.path.join(DIRPATH, f"fully_resolved_chisq_top{i+1}.pdf"))

# Plot resolved baseline ROC curve
if PLOT_ROCS:
    # Computes if χ² method found correct tops
    def correct_mask(pred_b, pred_q1, pred_q2):
        return (
            tgt_t1_mask
            & (pred_b == tgt_t1_b)
            & (
                ( (pred_q1 == tgt_t1_q1) & (pred_q2 == tgt_t1_q2) ) 
                | ( (pred_q1 == tgt_t1_q2) & (pred_q2 == tgt_t1_q1) )
            )
        ) | (
            tgt_t2_mask
            & (pred_b == tgt_t2_b)
            & (
                ( (pred_q1 == tgt_t2_q1) & (pred_q2 == tgt_t2_q2) ) 
                | ( (pred_q1 == tgt_t2_q2) & (pred_q2 == tgt_t2_q1) )
            )
        )

    correct_t1 = correct_mask(top_dict[f't{1}_b'], top_dict[f't{1}_q1'], top_dict[f't{1}_q2'])
    correct_t2 = correct_mask(top_dict[f't{2}_b'], top_dict[f't{2}_q1'], top_dict[f't{2}_q2'])
    
    valid_t1 = ~ak.is_none(correct_t1)
    valid_t2 = ~ak.is_none(correct_t2)

    print(f"num valid t1 = {ak.sum(valid_t1)} out of {ak.num(valid_t1, axis=0)}")
    print(f"num valid t2 = {ak.sum(valid_t2)} out of {ak.num(valid_t2, axis=0)}")
    print(f"num correct and valid t1 = {ak.sum(correct_t1[valid_t1])} out of {ak.num(correct_t1[valid_t1], axis=0)}")
    print(f"num correct and valid t2 = {ak.sum(correct_t2[valid_t2])} out of {ak.num(correct_t2[valid_t2], axis=0)}")

    chi2_t1 = ak.to_numpy(top_dict[f'FRt{1}_chi2'][valid_t1])
    chi2_t2 = ak.to_numpy(top_dict[f'FRt{2}_chi2'][valid_t2])
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
    plt.savefig(os.path.join(DIRPATH, "fully_resolved_chisq_ROC.pdf"))
