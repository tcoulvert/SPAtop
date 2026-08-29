import copy
import os

import h5py
import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import awkward as ak
import vector as vec
vec.register_awkward()

N_TOPS = 2
TOP_MASS = 172.52  # GeV
TOP_SIGMA = 20.
W_MASS = 80.37  # GeV
W_SIGMA = 14.

FILL_VALUE = 1e5

PLOT_CHI2_HISTS = False
PLOT_ROCS = False
SAVE_H5 = False

FILEPATH = os.path.abspath(__file__)
DIRPATH = '/'.join(FILEPATH.split('/')[:-1])
PLOT_DIRPATH = os.path.join(DIRPATH, f"v10/SEQChi2_FR")
if not os.path.exists(PLOT_DIRPATH): os.makedirs(PLOT_DIRPATH)

file_path = "/storage/af/user/tsievert/topNet/fjTag_testing.h5"
################################################
# 1) Load arrays
with h5py.File(file_path, "r") as f:
    mask = f['INPUTS/Jets/MASK'][:]
    pt   = f['INPUTS/Jets/pt'][:]
    eta  = f['INPUTS/Jets/eta'][:]
    phi  = f['INPUTS/Jets/phi'][:]
    mass = f['INPUTS/Jets/mass'][:]
    btag = f['INPUTS/Jets/btag'][:]
    
    fj_mask = f['INPUTS/BoostedJets/MASK'][:]
    fj_pt   = f['INPUTS/BoostedJets/fj_pt'][:]
    fj_eta  = f['INPUTS/BoostedJets/fj_eta'][:]
    fj_phi  = f['INPUTS/BoostedJets/fj_phi'][:]
    fj_mass = f['INPUTS/BoostedJets/fj_mass'][:]
    fj_Ttag = f['INPUTS/BoostedJets/fj_Ttag'][:]
    fj_Wtag = f['INPUTS/BoostedJets/fj_Wtag'][:]

    # FR
    tgt_FRt1_b    = f['TARGETS/FRt1/b'][:]
    tgt_FRt1_q1   = f['TARGETS/FRt1/q1'][:]
    tgt_FRt1_q2   = f['TARGETS/FRt1/q2'][:]
    tgt_FRt1_mask = f["TARGETS/FRt1/MASK"][:]

    tgt_FRt2_b    = f['TARGETS/FRt2/b'][:]
    tgt_FRt2_q1   = f['TARGETS/FRt2/q1'][:]
    tgt_FRt2_q2   = f['TARGETS/FRt2/q2'][:]
    tgt_FRt2_mask = f["TARGETS/FRt2/MASK"][:]

    # SRqq
    tgt_SRqqt1_b    = f['TARGETS/SRqqt1/b'][:]
    tgt_SRqqt1_qr   = f['TARGETS/SRqqt1/qq'][:]
    tgt_SRqqt1_mask = f["TARGETS/SRqqt1/MASK"][:]

    tgt_SRqqt2_b    = f['TARGETS/SRqqt2/b'][:]
    tgt_SRqqt2_qq   = f['TARGETS/SRqqt2/qq'][:]
    tgt_SRqqt2_mask = f["TARGETS/SRqqt2/MASK"][:]

    # FB
    tgt_FBt1_bqq    = f['TARGETS/FBt1/bqq'][:]
    tgt_FBt1_mask = f["TARGETS/FBt1/MASK"][:]

    tgt_FBt2_bqq    = f['TARGETS/FBt2/bqq'][:]
    tgt_FBt2_mask = f["TARGETS/FBt2/MASK"][:]

################################################
# 2) Build jagged [events][jets] array
_jets_ = ak.from_regular(ak.zip({
    "pt": pt,
    "eta": eta,
    "phi": phi,
    "mass": mass,
    "btag": btag
}, with_name="Momentum4D"))
_jets_["index"] = ak.local_index(_jets_, axis=1)
evt_order = np.arange(ak.num(_jets_, axis=0))

_fjets_ = ak.from_regular(ak.zip({
    "pt": fj_pt,
    "eta": fj_eta,
    "phi": fj_phi,
    "mass": fj_mass,
    "Ttag": fj_Ttag,
    "Wtag": fj_Wtag
}, with_name="Momentum4D"))
_fjets_["index"] = ak.local_index(_fjets_, axis=1)

dataset = {}
n_selected_tops = ak.zeros_like(evt_order)
################################################
# 3) Fill Boosted events
fjets = ak.mask(_fjets_, fj_mask)
bqq_fjets = ak.drop_none(ak.mask(fjets, fjets['Ttag']))
selected_bqq = ak.pad_none(bqq_fjets[ak.argsort(bqq_fjets.pt, axis=1, ascending=False)], N_TOPS, axis=1, clip=True)

n_selected_tops = n_selected_tops + ak.sum(~ak.is_none(selected_bqq, axis=1), axis=1)
print(f"n_selected after Boosted = {np.unique(n_selected_tops, return_counts=True)}")
for i in range(N_TOPS):
    dataset[f'TARGETS/FBt{i+1}/detection_probability'] = (ak.sum(~ak.is_none(selected_bqq, axis=1)) > i)
    dataset[f'TARGETS/FBt{i+1}/bqq'] = ak.fill_none(selected_bqq["index"][:, i], -1)
    dataset[f'TARGETS/FBt{i+1}/assignment_probability'] = dataset[f'TARGETS/FBt{i+1}/detection_probability']

postFB_fj_mask = copy.deepcopy(fj_mask); postFB_j_mask = copy.deepcopy(mask)
for i in range(N_TOPS):
    postFB_fj_mask = np.logical_and(
        postFB_fj_mask, 
        np.logical_and(
            _fjets_["index"] != dataset[f'TARGETS/FBt{i+1}/bqq'],
            _fjets_.deltaR(_fjets_[_fjets_["index"] == dataset[f'TARGETS/FBt{i+1}/bqq']]) > 0.8
        )
    )
    postFB_j_mask = np.logical_and(
        postFB_j_mask, 
        _jets_.deltaR(_fjets_[_fjets_["index"] == dataset[f'TARGETS/FBt{i+1}/bqq']]) > 0.8
    )

################################################
# 4) Fill SRqq events
fjets = ak.mask(_fjets_, postFB_fj_mask)
jets = ak.mask(_jets_, postFB_j_mask)
qq_fjets = ak.drop_none(ak.mask(fjets, fjets['Wtag']))
b_jets = ak.drop_none(ak.mask(jets, jets['btag']))

t = ak.cartesian({'w': qq_fjets, 'b': b_jets}, axis=1)
t = ak.with_field(t, (t.w + t.b).mass, "mass")
t = ak.with_field(t, t.w.deltaR(t.b), "deltaR")
t = ak.with_field(t, ((t.mass - TOP_MASS) / TOP_SIGMA)**2, "chi2")

t_mask = (t["deltaR"] > 0.8)

selected_qq, selected_b = [], []
for i in range(N_TOPS):
    chi2_SRqq = ak.where(t_mask, t["chi2"], FILL_VALUE)
    best_idx = ak.argmin(chi2_SRqq, axis=1)
    best_chi2 = ak.firsts(chi2_SRqq[ak.local_index(chi2_SRqq) == best_idx])
    best_t = ak.firsts(t[ak.local_index(t) == best_idx])

    good_chi2 = ((N_TOPS - n_selected_tops) > i) & (best_chi2 != FILL_VALUE)
    selected_qq.append(ak.mask(best_t.w, good_chi2)); selected_b.append(ak.mask(best_t.b, good_chi2))

    t_mask = t_mask & (
        ((t.w["index"] != best_t.w["index"]) & (t.b["index"] != best_t.b["index"]))
        | ~good_chi2
    )
    t_mask = ak.fill_none(t_mask, [], axis=0)

selected_qq = ak.concatenate([selection[:, np.newaxis] for selection in selected_qq], axis=1)
selected_b = ak.concatenate([selection[:, np.newaxis] for selection in selected_b], axis=1)

n_selected_tops = n_selected_tops + ak.sum(~ak.is_none(selected_qq, axis=1), axis=1)
print(f"n_selected after Semi-Resolved = {np.unique(n_selected_tops, return_counts=True)}")
for i in range(N_TOPS):
    dataset[f'TARGETS/SRqqt{i+1}/detection_probability'] = (ak.sum(~ak.is_none(selected_qq, axis=1)) > i)
    dataset[f'TARGETS/SRqqt{i+1}/b'] = ak.fill_none(selected_b["index"][:, i], -1)
    dataset[f'TARGETS/SRqqt{i+1}/qq'] = ak.fill_none(selected_qq["index"][:, i], -1)
    dataset[f'TARGETS/SRqqt{i+1}/assignment_probability'] = dataset[f'TARGETS/SRqqt{i+1}/detection_probability']

postSR_fj_mask = copy.deepcopy(postFB_fj_mask); postSR_j_mask = copy.deepcopy(postFB_j_mask)
for i in range(N_TOPS):
    postSR_fj_mask = np.logical_and(
        postSR_fj_mask, 
        np.logical_and(
            _fjets_["index"] != dataset[f'TARGETS/SRqqt{i+1}/qq'],
            _fjets_.deltaR(_fjets_[_fjets_["index"] == dataset[f'TARGETS/SRqqt{i+1}/qq']]) > 0.8
        )
    )
    postSR_j_mask = np.logical_and(
        postSR_j_mask, 
        np.logical_and(
            _jets_["index"] != dataset[f'TARGETS/SRqqt{i+1}/b'],
            _jets_.deltaR(_fjets_[_fjets_["index"] == dataset[f'TARGETS/SRqqt{i+1}/qq']]) > 0.8
        )
    )

################################################
# 5) Split jets
# 5a) events with exactly 0 btagged jets
jets = ak.mask(_jets_, postSR_j_mask)
ex0_bjet_candidates = (ak.sum(jets["btag"], axis=1) == 0)
ex0_jets = jets[ex0_bjet_candidates][ak.argsort(jets[ex0_bjet_candidates].pt, axis=1, ascending=False)]
ex0_evt_order = evt_order[ex0_bjet_candidates]

ex0_bjets = ex0_jets[:, :2]
ex0_bjets = ex0_bjets[ak.argsort(ex0_bjets.pt, axis=1, ascending=False)]
ex0_ljets = ex0_jets[:, 2:6]
ex0_ljets = ex0_ljets[ak.argsort(ex0_ljets.pt, axis=1, ascending=False)]


# 5b) events with exactly 1 btagged jets
ex1_bjet_candidates = (ak.sum(jets["btag"], axis=1) == 1)
ex1_jets = jets[ex1_bjet_candidates][ak.argsort(jets[ex1_bjet_candidates].btag, axis=1, ascending=False)]
ex1_evt_order = evt_order[ex1_bjet_candidates]

ex1_bjets = ak.concatenate([
    ak.singletons(ex1_jets[:, 0]), ak.singletons(ex1_jets[:, 1:][ak.argsort(ex1_jets[:, 1:].pt, axis=1, ascending=False)][:, 0])
], axis=1)
ex1_bjets = ex1_bjets[ak.argsort(ex1_bjets.pt, axis=1, ascending=False)]
ex1_ljets = ex1_jets[:, 1:][ak.argsort(ex1_jets[:, 1:].pt, axis=1, ascending=False)][:, 1:5]
ex1_ljets = ex1_ljets[ak.argsort(ex1_ljets.pt, axis=1, ascending=False)]


# 5c) events with exactly 2 btagged jets
ex2_bjet_candidates = (ak.sum(jets["btag"], axis=1) == 2)
ex2_jets = jets[ex2_bjet_candidates][ak.argsort(jets[ex2_bjet_candidates].btag, axis=1, ascending=False)]
ex2_evt_order = evt_order[ex2_bjet_candidates]

ex2_bjets = ex2_jets[:, :2]
ex2_bjets = ex2_bjets[ak.argsort(ex2_bjets.pt, axis=1, ascending=False)]
ex2_ljets = ex2_jets[:, 2:6]
ex2_ljets = ex2_ljets[ak.argsort(ex2_ljets.pt, axis=1, ascending=False)]


# 5d) events with more than 2 btagged jets
gt2_bjet_candidates = (ak.sum(jets["btag"], axis=1) > 2)
gt2_jets = jets[gt2_bjet_candidates][ak.argsort(jets[gt2_bjet_candidates].btag, axis=1, ascending=False)]
gt2_evt_order = evt_order[gt2_bjet_candidates]

gt2_bjets = gt2_jets[gt2_jets["btag"] == 1][ak.argsort(gt2_jets[gt2_jets["btag"] == 1].pt, axis=1, ascending=False)][:, :2]
gt2_bjets = gt2_bjets[ak.argsort(gt2_bjets.pt, axis=1, ascending=False)]
gt2_ljets = ak.concatenate([
    gt2_jets[gt2_jets["btag"] == 1][ak.argsort(gt2_jets[gt2_jets["btag"] == 1].pt, axis=1, ascending=False)][:, 2:],
    gt2_jets[gt2_jets["btag"] == 0]
], axis=1)[:, :4]
gt2_ljets = gt2_ljets[ak.argsort(gt2_ljets.pt, axis=1, ascending=False)]


# 3e) merge different n bTag categories and require regular arrays for chi2
evt_reorder = ak.argsort(ak.concatenate([ex0_evt_order, ex1_evt_order, ex2_evt_order, gt2_evt_order]))
bjets = ak.concatenate([ex0_bjets, ex1_bjets, ex2_bjets, gt2_bjets])[evt_reorder]
ljets = ak.concatenate([ex0_ljets, ex1_ljets, ex2_ljets, gt2_ljets])[evt_reorder]

chi2_mask = (ak.num(bjets, axis=1) == N_TOPS) & (ak.num(ljets, axis=1) == 2*N_TOPS)
bjets = ak.mask(bjets, chi2_mask)
ljets = ak.mask(ljets, chi2_mask)
print('bjets: ', ak.type(bjets))
print('ljets: ', ak.type(ljets))
print('N invalid chi2 events = ', ak.sum(~chi2_mask))


################################################
# Perform Chi2
@nb.njit
def expand_chosen(mask, chosen_var, fill_value, builder):
    chosen_idx = 0
    for ismasked in mask:
        if ismasked: builder.append(chosen_var[chosen_idx]); chosen_idx += 1
        else: builder.append(fill_value)
    return builder

selected_b, selected_q1, selected_q2 = [], [], []
for i in range(N_TOPS):
    w = ak.combinations(ljets, 2, axis=1, fields=["j1", "j2"])
    w = ak.with_field(w, (w.j1 + w.j2).mass, "mass")

    t = ak.cartesian({"w": w, "b": bjets}, axis=1)
    t = ak.with_field(t, (t.w.j1 + t.w.j2 + t.b).mass, "mass")
    t = ak.with_field(t, ak.min([t.w.j1.deltaR(t.w.j2), t.w.j1.deltaR(t.b), t.w.j2.deltaR(t.b)]), "mindeltaR")
    t = ak.with_field(t, ((t.w.mass - W_MASS) / W_SIGMA )**2 + ((t.mass - TOP_MASS) / TOP_SIGMA)**2, "chi2")

    t_mask = (
        (t.b["index"] != t.w.j1["index"]) & (t.b["index"] != t.w.j2["index"]) & (t.w.j1["index"] != t.w.j2["index"])
        & (t["mindeltaR"])
    )
    print('Any tops have overlapping jets (should be False)? ', ak.any(~t_mask))

    chi2_FR = ak.where(t_mask, t["chi2"], FILL_VALUE)
    best_idx = ak.argmin(chi2_FR, axis=1)
    best_chi2 = ak.firsts(chi2_FR[ak.local_index(chi2_FR) == best_idx])
    best_t = ak.firsts(t[ak.local_index(t) == best_idx])

    good_chi2 = ((N_TOPS - n_selected_tops) > i) & (best_chi2 != FILL_VALUE) & (best_chi2 < 45)
    selected_b.append(ak.mask(best_t.b, good_chi2))
    selected_q1.append(ak.mask(best_t.w.j1, good_chi2)); selected_q2.append(ak.mask(best_t.w.j2, good_chi2))

    # n_events, n_ts = ak.num(t, axis=0), ak.num(t, axis=1)[0]
    # random_idxs = np.random.choice(n_ts, size=n_events)
    # random_t, random_chi2 = ak.firsts(t[ak.local_index(t) == random_idxs], axis=1), ak.firsts(good_chi2[ak.local_index(good_chi2) == random_idxs], axis=1)

    bjets, ljets = ak.from_regular(bjets), ak.from_regular(ljets)
    bjets = ak.to_regular(bjets[bjets.index != best_t.b.index])
    ljets = ak.to_regular(ljets[(ljets.index != best_t.w.j1.index) & (ljets.index != best_t.w.j2.index)])

selected_b = ak.concatenate([selection[:, np.newaxis] for selection in selected_b], axis=1)
selected_q1 = ak.concatenate([selection[:, np.newaxis] for selection in selected_q1], axis=1)
selected_q2 = ak.concatenate([selection[:, np.newaxis] for selection in selected_q2], axis=1)

n_selected_tops = n_selected_tops + ak.sum(~ak.is_none(selected_q1, axis=1), axis=1)
print(f"n_selected after Fully-Resolved = {np.unique(n_selected_tops, return_counts=True)}")
for i in range(N_TOPS):
    dataset[f'TARGETS/FRt{i+1}/detection_probability'] = (ak.sum(~ak.is_none(selected_b, axis=1)) > i)
    dataset[f'TARGETS/FRt{i+1}/b'] = ak.fill_none(selected_b["index"][:, i], -1)
    dataset[f'TARGETS/FRt{i+1}/q1'] = ak.fill_none(selected_q1["index"][:, i], -1)
    dataset[f'TARGETS/FRt{i+1}/q2'] = ak.fill_none(selected_q2["index"][:, i], -1)
    dataset[f'TARGETS/FRt{i+1}/assignment_probability'] = dataset[f'TARGETS/FRt{i+1}/detection_probability']

################################################
## Outputs ##
################################################
# Transforms χ² to probability for analysis
def chi2_to_prob(chi2):
    prob = np.exp(-chi2)
    prob = np.where(chi2 != FILL_VALUE, prob, 0)
    return prob

################################################
# Save out new h5 file
if SAVE_H5:
    out_filepath = os.path.join(DIRPATH, f"tt_hadronic_{'SPANET' if SPANET_CHI2_METHOD else 'SEQ'}chi2.h5")
    if os.path.exists(out_filepath): os.remove(out_filepath)
    with h5py.File(out_filepath, 'a') as f:
        with h5py.File(file_path, 'r') as test_f:
            for jet_class in test_f['INPUTS'].keys():
                for variable in test_f['INPUTS'][jet_class].keys():
                    if f'INPUTS/{jet_class}/{variable}' not in f:
                        f[f'INPUTS/{jet_class}/{variable}'] = test_f[f'INPUTS/{jet_class}/{variable}'][:]

        for i in range(N_TOPS):
            f[f'TARGETS/FRt{i+1}/detection_probability'] = ak.to_numpy(top_dict[f'FRt{i+1}_mask'] & (top_dict[f'FRt{i+1}_chi2'] < 45), allow_missing=False)
            f[f'TARGETS/FRt{i+1}/b'] = ak.to_numpy(top_dict[f'FRt{i+1}_b'], allow_missing=False)
            f[f'TARGETS/FRt{i+1}/q1'] = ak.to_numpy(top_dict[f'FRt{i+1}_q1'], allow_missing=False)
            f[f'TARGETS/FRt{i+1}/q2'] = ak.to_numpy(top_dict[f'FRt{i+1}_q2'], allow_missing=False)
            f[f'TARGETS/FRt{i+1}/assignment_probability'] = chi2_to_prob(ak.to_numpy(top_dict[f'FRt{i+1}_chi2'], allow_missing=False))

################################################
# Computes if χ² method found correct tops
def correct_mask(pred_b, pred_q1, pred_q2, top_idx=1):
    if top_idx == 1:
        return (
            (pred_b == tgt_FRt1_b)
            & (
                ( (pred_q1 == tgt_FRt1_q1) & (pred_q2 == tgt_FRt1_q2) ) 
                | ( (pred_q1 == tgt_FRt1_q2) & (pred_q2 == tgt_FRt1_q1) )
            )
        )
    elif top_idx == 2:
        return (
            (pred_b == tgt_FRt2_b)
            & (
                ( (pred_q1 == tgt_FRt2_q1) & (pred_q2 == tgt_FRt2_q2) ) 
                | ( (pred_q1 == tgt_FRt2_q2) & (pred_q2 == tgt_FRt2_q1) )
            )
        )

################################################
# Plot resolved baseline χ² distributions
if PLOT_CHI2_HISTS:
    # Plot Top χ² histograms
    for i in range(N_TOPS):
        correct_t = correct_mask(top_dict[f'FRt{i+1}_b'], top_dict[f'FRt{i+1}_q1'], top_dict[f'FRt{i+1}_q2'], top_idx=i+1)
        valid_t = (tgt_FRt1_mask if i == 0 else tgt_FRt2_mask)
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
        plt.savefig(os.path.join(PLOT_DIRPATH, f"fully_resolved_{'SPANET' if SPANET_CHI2_METHOD else 'SEQ'}chisq_top{i+1}.pdf"))

    # Plot Top χ² histograms
    for i in range(N_TOPS):
        correct_t = correct_mask(rand_dict[f'FRt{i+1}_b'], rand_dict[f'FRt{i+1}_q1'], rand_dict[f'FRt{i+1}_q2'], top_idx=i+1)
        valid_t = (tgt_FRt1_mask if i == 0 else tgt_FRt2_mask)
        corr_chi2_t_vals = ak.ravel(rand_dict[f'FRt{i+1}_chi2'][correct_t & valid_t])
        incorr_chi2_t_vals = ak.ravel(rand_dict[f'FRt{i+1}_chi2'][~correct_t & valid_t])
        plt.figure()
        plt.hist([corr_chi2_t_vals, incorr_chi2_t_vals], range=(0, 500), bins=100, label=['Correct top assignment', 'Incorrect top assignment'], stacked=True)
        plt.xlabel(f"χ² (Top{i+1})")
        plt.ylabel("Frequency")
        plt.yscale('log')
        plt.title(f"Chi-Squared Distribution for Randomly chosen Top{i+1} Candidates")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIRPATH, f"fully_resolved_{'SPANET' if SPANET_CHI2_METHOD else 'SEQ'}chisqRand_top{i+1}.pdf"))

################################################
# Plot resolved baseline ROC curve
if PLOT_ROCS:
    correct_t1 = correct_mask(top_dict[f'FRt{1}_b'], top_dict[f'FRt{1}_q1'], top_dict[f'FRt{1}_q2'], top_idx=1)
    correct_t2 = correct_mask(top_dict[f'FRt{2}_b'], top_dict[f'FRt{2}_q1'], top_dict[f'FRt{2}_q2'], top_idx=2)
    chi2_t1 = ak.to_numpy(top_dict[f'FRt{1}_chi2'][tgt_FRt1_mask], allow_missing=False)
    chi2_t2 = ak.to_numpy(top_dict[f'FRt{2}_chi2'][tgt_FRt2_mask], allow_missing=False)
    label_t1 = ak.to_numpy(correct_t1[tgt_FRt1_mask], allow_missing=False)
    label_t2 = ak.to_numpy(correct_t2[tgt_FRt2_mask], allow_missing=False)

    correct_t1_rand = correct_mask(rand_dict[f'FRt{1}_b'], rand_dict[f'FRt{1}_q1'], rand_dict[f'FRt{1}_q2'], top_idx=1)
    correct_t2_rand = correct_mask(rand_dict[f'FRt{2}_b'], rand_dict[f'FRt{2}_q1'], rand_dict[f'FRt{2}_q2'], top_idx=2)
    chi2_t1_rand = ak.to_numpy(rand_dict[f'FRt{1}_chi2'][tgt_FRt1_mask], allow_missing=False)
    chi2_t2_rand = ak.to_numpy(rand_dict[f'FRt{2}_chi2'][tgt_FRt2_mask], allow_missing=False)
    label_t1_rand = ak.to_numpy(correct_t1_rand[tgt_FRt1_mask], allow_missing=False)
    label_t2_rand = ak.to_numpy(correct_t2_rand[tgt_FRt2_mask], allow_missing=False)

    print(f"num valid t1 = {ak.sum(tgt_FRt1_mask)} out of {ak.num(tgt_FRt1_mask, axis=0)}")
    print(f"num valid t2 = {ak.sum(tgt_FRt2_mask)} out of {ak.num(tgt_FRt2_mask, axis=0)}")
    print(f"num correct and valid t1 = {ak.sum(correct_t1[tgt_FRt1_mask])} out of {ak.num(correct_t1[tgt_FRt1_mask], axis=0)}")
    print(f"num correct and valid t2 = {ak.sum(correct_t2[tgt_FRt2_mask])} out of {ak.num(correct_t2[tgt_FRt2_mask], axis=0)}")
    print(f"num correct and valid random t1 = {ak.sum(correct_t1_rand[tgt_FRt1_mask])} out of {ak.num(correct_t1_rand[tgt_FRt1_mask], axis=0)}")
    print(f"num correct and valid random t2 = {ak.sum(correct_t2_rand[tgt_FRt2_mask])} out of {ak.num(correct_t2_rand[tgt_FRt2_mask], axis=0)}")

    # === Plot ROC ===
    def plot_roc(chi2_vals, label, plotlabel):
        fpr, tpr, _ = roc_curve(label, chi2_vals)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{plotlabel} (AUC = {roc_auc:.3f})")

    plt.figure(figsize=(7, 6))
    plot_roc(chi2_t1, label_t1, "Top1")
    plot_roc(chi2_t2, label_t2, "Top2")
    plot_roc(chi2_t1_rand, label_t1_rand, "Random Top1")
    plot_roc(chi2_t2_rand, label_t2_rand, "Random Top2")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Chi² Discriminator")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIRPATH, f"fully_resolved_{'SPANET' if SPANET_CHI2_METHOD else 'SEQ'}chisq_ROC.pdf"))
