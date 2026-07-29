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
PLOT_DIRPATH = os.path.join(DIRPATH, 'v10/Chi2_FR')
if not os.path.exists(PLOT_DIRPATH): os.makedirs(PLOT_DIRPATH)

N_TOPS = 2
TOP_MASS = 172.52  # GeV
TOP_SIGMA = 20.
W_MASS = 80.37  # GeV
W_SIGMA = 14.

FILL_VALUE = 1e5

# PLOT_CHI2_HISTS = True
# PLOT_ROCS = True
# SAVE_H5 = True
PLOT_CHI2_HISTS = False
PLOT_ROCS = False
SAVE_H5 = False

file_path = "/storage/af/user/tsievert/topNet/tt_hadronic_fixed_test.h5"
################################################
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
    tgt_t1_mask = f["TARGETS/FRt1/MASK"][:]

    tgt_t2_b    = f['TARGETS/FRt2/b'][:]
    tgt_t2_q1   = f['TARGETS/FRt2/q1'][:]
    tgt_t2_q2   = f['TARGETS/FRt2/q2'][:]
    tgt_t2_mask = f["TARGETS/FRt2/MASK"][:]

################################################
# 2) Build jagged [events][jets] array
jets = ak.zip({
    "pt": pt,
    "eta": eta,
    "phi": phi,
    "mass": mass,
    "btag": btag
}, with_name="Momentum4D")
jets = ak.with_field(jets, ak.local_index(jets, axis=1), "index")
jets = jets[ak.argsort(jets.btag, axis=1, ascending=False)]


################################################
# 3) Split jets
# 3a) events with exactly 0 btagged jets
ex0_bjet_candidates = (ak.sum(jets["btag"], axis=1) == 0)
ex0_jets = jets[ex0_bjet_candidates][ak.argsort(jets[ex0_bjet_candidates].pt, axis=1, ascending=False)]
ex0_bjets = ex0_jets[:, :2]
ex0_bjets = ex0_bjets[ak.argsort(ex0_bjets.pt, axis=1, ascending=False)]
ex0_ljets = ex0_jets[:, 2:6]
ex0_ljets = ex0_ljets[ak.argsort(ex0_ljets.pt, axis=1, ascending=False)]

# 3b) events with exactly 1 btagged jets
ex1_bjet_candidates = (ak.sum(jets["btag"], axis=1) == 1)
ex1_jets = jets[ex1_bjet_candidates]
ex1_bjets = ak.concatenate([
    ak.singletons(ex1_jets[:, 0]), ak.singletons(ex1_jets[:, 1:][ak.argsort(ex1_jets[:, 1:].pt, axis=1, ascending=False)][:, 0])
], axis=1)
ex1_bjets = ex1_bjets[ak.argsort(ex1_bjets.pt, axis=1, ascending=False)]
ex1_ljets = ex1_jets[:, 1:][ak.argsort(ex1_jets[:, 1:].pt, axis=1, ascending=False)][:, 1:5]
ex1_ljets = ex1_ljets[ak.argsort(ex1_ljets.pt, axis=1, ascending=False)]

# 3c) events with exactly 2 btagged jets
ex2_bjet_candidates = (ak.sum(jets["btag"], axis=1) == 2)
ex2_jets = jets[ex2_bjet_candidates]
ex2_bjets = ex2_jets[:, :2]
ex2_bjets = ex2_bjets[ak.argsort(ex2_bjets.pt, axis=1, ascending=False)]
ex2_ljets = ex2_jets[:, 2:6]
ex2_ljets = ex2_ljets[ak.argsort(ex2_ljets.pt, axis=1, ascending=False)]

# 3d) events with more than 2 btagged jets
gt2_bjet_candidates = (ak.sum(jets["btag"], axis=1) > 2)
gt2_jets = jets[gt2_bjet_candidates]
gt2_bjets = gt2_jets[gt2_jets["btag"] == 1][ak.argsort(gt2_jets[gt2_jets["btag"] == 1].pt, axis=1, ascending=False)][:, :2]
gt2_bjets = gt2_bjets[ak.argsort(gt2_bjets.pt, axis=1, ascending=False)]
gt2_ljets = ak.concatenate([
    gt2_jets[gt2_jets["btag"] == 1][ak.argsort(gt2_jets[gt2_jets["btag"] == 1].pt, axis=1, ascending=False)][:, 2:],
    gt2_jets[gt2_jets["btag"] == 0]
], axis=1)[:, :4]
gt2_ljets = gt2_ljets[ak.argsort(gt2_ljets.pt, axis=1, ascending=False)]

# 3e) merge different n bTag categories and require regular arrays for chi2
bjets = ak.concatenate([ex0_bjets, ex1_bjets, ex2_bjets, gt2_bjets])
ljets = ak.concatenate([ex0_ljets, ex1_ljets, ex2_ljets, gt2_ljets])

chi2_mask = (ak.num(bjets, axis=1) == 2) & (ak.num(ljets, axis=1) == 4)
bjets = ak.to_regular(bjets[chi2_mask])
ljets = ak.to_regular(ljets[chi2_mask])
print('bjets: ', ak.type(bjets))
print('ljets: ', ak.type(ljets))
print('N invalid chi2 events = ', ak.sum(~chi2_mask))


################################################
# 4) Build W and T combinations from bjets and ljets
w = ak.combinations(ljets, 2, axis=1, fields=["j1", "j2"])
w = ak.with_field(w, (w.j1 + w.j2).mass, "mass")

t = ak.cartesian({"w": w, "b": bjets}, axis=1)
t = ak.with_field(t, (t.w.j1 + t.w.j2 + t.b).mass, "mass")
t = ak.with_field(t, (t.w.j1 + t.w.j2 + t.b).pt, "pt")

tt = ak.combinations(t, 2, axis=1, fields=["t1", "t2"])


################################################
# 5) Build ch2 for two tops simultaneously
chi2 = (
    ( (tt.t1.mass - tt.t2.mass) / TOP_SIGMA )**2 
    + ( (tt.t1.w.mass - W_MASS) / W_SIGMA )**2
    + ( (tt.t2.w.mass - W_MASS) / W_SIGMA )**2
)


################################################
# 6) Select jets by minimizing chi2 and build dict
best_idx = ak.argmin(chi2, axis=1)
best_chi2 = ak.firsts(chi2[ak.local_index(chi2) == best_idx])
best_tt = ak.firsts(tt[ak.local_index(tt) == best_idx])

top_dict = {}
for i in range(2):
    top_dict[f'FRt{i+1}_mask'] = chi2_mask
    top_dict[f'FRt{i+1}_b'] = ak.where(chi2_mask, best_tt[f"t{i+1}"].b.index, -1)
    top_dict[f'FRt{i+1}_q1'] = ak.where(chi2_mask, best_tt[f"t{i+1}"].w.j1.index, -1)
    top_dict[f'FRt{i+1}_q2'] = ak.where(chi2_mask, best_tt[f"t{i+1}"].w.j2.index, -1)
    top_dict[f'FRt{i+1}_pt'] = ak.where(chi2_mask, best_tt[f"t{i+1}"].pt, -1)
    top_dict[f'FRt{i+1}_chi2'] = ak.where(chi2_mask, best_chi2, FILL_VALUE)
    

################################################
def chi2_to_prob(chi2):
    prob = np.exp(-chi2)
    prob = np.where(chi2 != FILL_VALUE, prob, 0)
    return prob


for i in range(2):
    print(f'top {i+1} \n', '-'*60)
    print(f'  chi2 - ', top_dict[f'FRt{i+1}_chi2'])
    print(f'  prob - ', chi2_to_prob(top_dict[f'FRt{i+1}_chi2']))

# Save out new h5 file
if SAVE_H5:
    out_filepath = os.path.join(DIRPATH, "tt_hadronic_chi2.h5")
    with h5py.File(out_filepath, 'a') as f:
        with h5py.File(file_path, 'r') as test_f:
            for jet_class in test_f['INPUTS'].keys():
                for variable in test_f['INPUTS'][jet_class].keys():
                    if f'INPUTS/{jet_class}/{variable}' not in f:
                        f[f'INPUTS/{jet_class}/{variable}'] = test_f[f'INPUTS/{jet_class}/{variable}'][:]

        for i in range(N_TOPS):
            f[f'TARGETS/FRt{i+1}/detection_probability'] = ak.to_numpy(top_dict[f'FRt{i+1}_mask'], allow_missing=False)
            f[f'TARGETS/FRt{i+1}/b'] = ak.to_numpy(top_dict[f'FRt{i+1}_b'], allow_missing=False)
            f[f'TARGETS/FRt{i+1}/q1'] = ak.to_numpy(top_dict[f'FRt{i+1}_q1'], allow_missing=False)
            f[f'TARGETS/FRt{i+1}/q2'] = ak.to_numpy(top_dict[f'FRt{i+1}_q2'], allow_missing=False)
            f[f'TARGETS/FRt{i+1}/assignment_probability'] = chi2_to_prob(ak.to_numpy(top_dict[f'FRt{i+1}_chi2'], allow_missing=False))


################################################
# Computes if χ² method found correct tops
def correct_mask(pred_b, pred_q1, pred_q2, top_idx=1):
    if top_idx == 1:
        return (
            (pred_b == tgt_t1_b)
            & (
                ( (pred_q1 == tgt_t1_q1) & (pred_q2 == tgt_t1_q2) ) 
                | ( (pred_q1 == tgt_t1_q2) & (pred_q2 == tgt_t1_q1) )
            )
        )
    elif top_idx == 2:
        return (
            (pred_b == tgt_t2_b)
            & (
                ( (pred_q1 == tgt_t2_q1) & (pred_q2 == tgt_t2_q2) ) 
                | ( (pred_q1 == tgt_t2_q2) & (pred_q2 == tgt_t2_q1) )
            )
        )

## Outputs ##
# Plot resolved baseline χ² distributions
if PLOT_CHI2_HISTS:
    # Plot Top χ² histograms
    for i in range(N_TOPS):
        valid_t = (tgt_t1_mask if i == 0 else tgt_t2_mask)
        chi2_vals_1 = ak.ravel(top_dict[f'FRt{i+1}_chi2'][valid_t])
        plt.figure()
        plt.hist(chi2_vals_1, range=(0, 500), bins=100)
        plt.xlabel(f"χ² (Top{i+1})")
        plt.ylabel("Frequency")
        plt.yscale('log')
        plt.title(f"Chi-Squared Distribution for Top{i+1} Candidates")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIRPATH, f"fully_resolved_chisq_top{i+1}.pdf"))

    # Plot Top χ² histograms
    for i in range(N_TOPS):
        correct_t = correct_mask(top_dict[f'FRt{i+1}_b'], top_dict[f'FRt{i+1}_q1'], top_dict[f'FRt{i+1}_q2'], top_idx=i+1)
        valid_t = (tgt_t1_mask if i == 0 else tgt_t2_mask)
        corr_chi2_t_vals = ak.ravel(top_dict[f'FRt{i+1}_chi2'][correct_t & valid_t])
        incorr_chi2_t_vals = ak.ravel(top_dict[f'FRt{i+1}_chi2'][~correct_t & valid_t])
        plt.figure()
        plt.hist([corr_chi2_t_vals, incorr_chi2_t_vals], range=(0, 500), bins=100, label=['Correct top assignment', 'Incorrect top assignment'], stacked=True)
        plt.xlabel(f"χ² (Top{i+1})")
        plt.ylabel("Frequency")
        plt.yscale('log')
        plt.title(f"Chi-Squared Distribution for Top{i+1} Candidates")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIRPATH, f"fully_resolved_chisqCorr_top{i+1}.pdf"))

# Plot resolved baseline ROC curve
if PLOT_ROCS:
    correct_t1 = correct_mask(top_dict[f'FRt{1}_b'], top_dict[f'FRt{1}_q1'], top_dict[f'FRt{1}_q2'], top_idx=1)
    correct_t2 = correct_mask(top_dict[f'FRt{2}_b'], top_dict[f'FRt{2}_q1'], top_dict[f'FRt{2}_q2'], top_idx=2)
    
    valid_t1 = tgt_t1_mask
    valid_t2 = tgt_t2_mask

    print(f"num valid t1 = {ak.sum(valid_t1)} out of {ak.num(valid_t1, axis=0)}")
    print(f"num valid t2 = {ak.sum(valid_t2)} out of {ak.num(valid_t2, axis=0)}")
    print(f"num correct and valid t1 = {ak.sum(correct_t1[valid_t1])} out of {ak.num(correct_t1[valid_t1], axis=0)}")
    print(f"num correct and valid t2 = {ak.sum(correct_t2[valid_t2])} out of {ak.num(correct_t2[valid_t2], axis=0)}")

    chi2_t1 = ak.to_numpy(top_dict[f'FRt{1}_chi2'][valid_t1], allow_missing=False)
    chi2_t2 = ak.to_numpy(top_dict[f'FRt{2}_chi2'][valid_t2], allow_missing=False)
    label_t1 = ak.to_numpy(correct_t1[valid_t1], allow_missing=False)
    label_t2 = ak.to_numpy(correct_t2[valid_t2], allow_missing=False)

    # === Plot ROC ===
    def plot_roc(chi2_vals, label, plotlabel):
        fpr, tpr, _ = roc_curve(label, chi2_vals)  # -chi2_vals
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
    plt.savefig(os.path.join(PLOT_DIRPATH, "fully_resolved_chisq_ROC.pdf"))
