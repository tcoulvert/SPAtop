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

TOP_MASS = 172.52  # GeV
W_MASS = 80.37  # GeV

PLOT_CHI2_HISTS = False
PLOT_ROCS = True

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

## Top 1 ##
# 3) Split jets
bjets_mask = (jets.btag == 1)
bjets = ak.drop_none(ak.mask(jets, bjets_mask))
ljets = ak.drop_none(ak.mask(jets, ~bjets_mask))

# 4) W1 jj combinations
w1 = ak.combinations(ljets, 2, axis=1, fields=["j1", "j2"])
w1 = ak.with_field(w1, (w1.j1 + w1.j2).mass, "w_mass")

# 5) Top1 combinations
t1 = ak.cartesian({"w": w1, "b": bjets}, axis=1)
t1 = ak.with_field(t1, (t1.w.j1 + t1.w.j2 + t1.b).mass, "top_mass")

# 6) Top1 χ²
chi2_all1 = ( (t1.w.w_mass - W_MASS) / (0.1 * W_MASS) )**2 + ( (t1.top_mass - TOP_MASS) / (0.1 * TOP_MASS) )**2
idx1 = ak.argmin(chi2_all1, axis=1)
best1 = ak.firsts(t1[ak.local_index(t1) == idx1])
best_chi2_1 = ak.firsts(chi2_all1[ak.local_index(t1) == idx1])

## Top 2 ##
# 7) Build ak arrays of unused jets
t1_bjet_idx = ak.firsts(t1.b.index[ak.local_index(t1) == idx1])
t1_qjet1_idx = ak.firsts(t1.w.j1.index[ak.local_index(t1) == idx1])
t1_qjet2_idx = ak.firsts(t1.w.j2.index[ak.local_index(t1) == idx1])

top2_bjets = bjets[bjets.index != t1_bjet_idx]
top2_ljets = ljets[(ljets.index != t1_qjet1_idx) & (ljets.index != t1_qjet2_idx)]

# 8) W2 jj combinations
w2 = ak.combinations(top2_ljets, 2, axis=1, fields=["j1", "j2"])
w2 = ak.with_field(w2, (w2.j1 + w2.j2).mass, "w_mass")

# 9) Top2 combinations
t2 = ak.cartesian({"w": w2, "b": top2_bjets}, axis=1)
t2 = ak.with_field(t2, (t2.w.j1 + t2.w.j2 + t2.b).mass, "top_mass")

# 10) Top2 χ²
chi2_all2 = ( (t2.w.w_mass - W_MASS) / (0.1 * W_MASS) )**2 + ( (t2.top_mass - TOP_MASS) / (0.1 * TOP_MASS) )**2
idx2 = ak.argmin(chi2_all2, axis=1)
best2 = ak.firsts(t2[ak.local_index(t2) == idx2])
best_chi2_2 = ak.firsts(chi2_all2[ak.local_index(t2) == idx2])

t2_bjet_idx = ak.firsts(t2.b.index[ak.local_index(t2) == idx2])
t2_qjet1_idx = ak.firsts(t2.w.j1.index[ak.local_index(t2) == idx2])
t2_qjet2_idx = ak.firsts(t2.w.j2.index[ak.local_index(t2) == idx2])


## Outputs ##
# Plot resolved baseline χ² distributions
if PLOT_CHI2_HISTS:
    # Plot Top1 χ² histogram
    chi2_vals_1 = ak.ravel(best_chi2_1[~ak.is_none(best_chi2_1)])
    plt.figure()
    plt.hist(chi2_vals_1, bins=50)
    plt.xlabel("χ² (Top1)")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.title("Chi-Squared Distribution for Top1 Candidates")
    plt.grid(True)
    plt.savefig(os.path.join(DIRPATH, "fully_resolved_chisq_top1.pdf"))
    # Plot Top2 χ² histogram 
    chi2_vals_2 = ak.ravel(best_chi2_2[~ak.is_none(best_chi2_2)])
    plt.figure()
    plt.hist(chi2_vals_2, bins=50)
    plt.xlabel("χ² (Top2)")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.title("Chi-Squared Distribution for Top2 Candidates")
    plt.grid(True)
    plt.savefig(os.path.join(DIRPATH, "fully_resolved_chisq_top2.pdf"))

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

    correct_t1 = correct_mask(t1_bjet_idx, t1_qjet1_idx, t1_qjet2_idx)
    correct_t2 = correct_mask(t2_bjet_idx, t2_qjet1_idx, t2_qjet2_idx)
    
    valid_t1 = ~ak.is_none(correct_t1)
    valid_t2 = ~ak.is_none(correct_t2)

    print(f"num valid t1 = {ak.sum(valid_t1)} out of {ak.num(valid_t1, axis=0)}")
    print(f"num valid t2 = {ak.sum(valid_t2)} out of {ak.num(valid_t2, axis=0)}")
    print(f"num correct and valid t1 = {ak.sum(correct_t1[valid_t1])} out of {ak.num(correct_t1[valid_t1], axis=0)}")
    print(f"num correct and valid t2 = {ak.sum(correct_t2[valid_t2])} out of {ak.num(correct_t2[valid_t2], axis=0)}")

    chi2_t1 = ak.to_numpy(best_chi2_1[valid_t1])
    chi2_t2 = ak.to_numpy(best_chi2_2[valid_t2])
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
